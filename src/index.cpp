#include "index.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>

#include "exceptions.h" 
#include "parameters.h"

#include <random>
#include <sys/stat.h>
#include <sstream>

#include <iostream>
#include <ctime>
#include <iomanip>

#include <fstream>
#include <vector>
#include <map>
#include <queue>

#include <chrono>
#include <float.h>

namespace efanna2e {

template <typename T>
Index<T>::Index(const size_t dimension,
                    const size_t max_num_points,
                    Metric metric,
                    DynamicIndex<T> *initializer)
    : DynamicIndex<T>(dimension, max_num_points)
{
  metric_ = metric;
  unified_store_.init_distance(metric);
  std::cout << "distance metric: " << metric_ << std::endl;
}

template <typename T>
Index<T>::~Index() {
}

// ----- main functions -----
template <typename T>
void Index<T>::Init(const Parameters &parameters) {
  std::cout << "max nd : " << this->max_nd_ << std::endl;
  const unsigned range = parameters.Get<unsigned>("R");
  width = range;

  // [1] DATA STORE: align this->max_nd_ vectors on data store
  unified_store_.init_datastore(this->max_nd_, this->dimension_);

  // [2] GRAPH STORE: align range neighbor ids of this->max_nd_ vectors on graph store
  unified_store_.init_graphstore(this->max_nd_, parameters);

  // [3] TAG TO LOCATION: reserve this->max_nd_ tag->location mapping
  // tag is made by user, and location is set automatically (linearly without delete)
  this->tag_to_location_.reserve(this->max_nd_);
  this->location_to_tag_.reserve(this->max_nd_);


}


template <typename T>
void Index<T>::BatchInsert(const std::vector<unsigned>& tags,
                          const std::vector<const T*>& vectors,
                          const Parameters& parameters) {

  int num_inserts = static_cast<int>(tags.size());

  std::cout << "insert " << num_inserts << " vectors..." << std::endl;

#pragma omp parallel for schedule(dynamic, 16) default(none) shared(num_inserts, vectors, tags, parameters)
  for (int i = 0; i < num_inserts; ++i) {
    Insert(tags[i], vectors[i], parameters);
  }
}

template <typename T>
void Index<T>::Insert(unsigned tag,
                      const T *vector,
                      const Parameters &parameters) {
  unsigned location;
  bool is_first = false;
  {
    std::unique_lock<std::shared_mutex> lock(tag_lock_);
    location = unified_store_.reserve_location();
    this->tag_to_location_[tag] = location;
    this->location_to_tag_.set(location, tag);
    this->nd_++;
    if (this->nd_ == 1) is_first = true;
  }

  unified_store_.set_vector(location, vector);

  if (is_first) { // first point
    return;
  }

  std::vector<Neighbor> expanded_nodes;
  NeighborPriorityQueue best_L_nodes;
  tsl::robin_set<uint32_t> inserted_into_pool_rs;

  get_neighbors(location, 
                vector, 
                parameters, 
                inserted_into_pool_rs, 
                best_L_nodes, 
                expanded_nodes);

  std::vector<Neighbor> result;

  sync_prune(vector, parameters, expanded_nodes, result);

  std::vector<unsigned> neighbors;
  for (const auto& neighbor : result) {
    if (neighbor.id == location) continue;
    neighbors.push_back(neighbor.id);
  }

  unified_store_.SetNeighbors(location, neighbors);

  InterInsert(location, vector, parameters, result);
}

template <typename T>
void Index<T>::BatchInsert_crng(const std::vector<unsigned>& tags,
                          const std::vector<const T*>& vectors,
                          const Parameters& parameters) {

  int num_inserts = static_cast<int>(tags.size());

  bool is_delete = !unified_store_.is_deleted_set_empty(); // empty then false

  std::cout << "rnn insert (lazy delete) " << num_inserts << " vectors..." << std::endl;

  // Initialize the entry point once before parallel CONDA inserts.
  int start_idx = 0;
  if (!is_delete && num_inserts > 0) {
    Insert_crng(tags[0], vectors[0], parameters, false);
    start_idx = 1;
    is_delete = true;
  }

#pragma omp parallel for schedule(dynamic, 16) default(none) shared(num_inserts, vectors, tags, parameters, is_delete, start_idx)
  for (int i = start_idx; i < num_inserts; ++i) {
    Insert_crng(tags[i], vectors[i], parameters, is_delete);
  }
}

template <typename T>
void Index<T>::Insert_crng(unsigned tag,
                      const T *vector,
                      const Parameters &parameters,
                      bool is_delete) {
  unsigned location;
  bool is_first = false;
  {
    std::unique_lock<std::shared_mutex> lock(tag_lock_);
    location = unified_store_.reserve_location();
    this->tag_to_location_[tag] = location;
    this->location_to_tag_.set(location, tag);
    this->nd_++;
    if (this->nd_ == 1) is_first = true;
  }

  unified_store_.set_vector(location, vector);

  if (is_first) { // first point
    ep_ = location;
    return;
  }

  std::vector<Neighbor> expanded_nodes;
  NeighborPriorityQueue best_L_nodes;
  tsl::robin_set<uint32_t> inserted_into_pool_rs;

  if (is_delete) {
    get_neighbors_lazy(location, 
      vector, 
      parameters, 
      inserted_into_pool_rs, 
      best_L_nodes, 
      expanded_nodes);
  } else {
    get_neighbors(location, 
      vector, 
      parameters, 
      inserted_into_pool_rs, 
      best_L_nodes, 
      expanded_nodes);
  }

  std::sort(expanded_nodes.begin(), expanded_nodes.end());

  // do crng based pruning
  const unsigned range = parameters.Get<unsigned>("R");

  // keep minimum distance(from result node) at occlude_factor
  std::vector<float> occlude_factor(expanded_nodes.size(), FLT_MAX);
  std::vector<unsigned> occlude_id(expanded_nodes.size());
  std::vector<unsigned> pruned_id(expanded_nodes.size(), UINT_MAX);

  std::vector<Neighbor> result;
  unsigned maxid = 0;

  for (size_t i=0; i<expanded_nodes.size() && result.size() < range; ++i) {
    auto p = expanded_nodes[i];
    // [1] check existence of the path result -> p
    if (occlude_factor[i] == FLT_MAX) {
      // [1-1] no way to result->p (2-hop path).
      // add the edge (result->p)
      result.push_back(p);
      maxid = i;
    } else {
      // [1-2] exist 2-hop path (result-> ? ->p)
      // [2] compare the distance
      if (occlude_factor[i] < p.distance) {
        // [2-1] d(result, p) < d(loc, p)
        // p is occluded by result. no need to add the edge
        pruned_id[i] = occlude_id[i];
        continue;
      } else {
        // [2-2] d(result, p) > p(loc, p)
        // add edge loc -> p and remove result -> p
        result.push_back(p);
        maxid = i;
        unsigned result_loc = occlude_id[i];
        // remove p from the neighbor of result_loc;
        unified_store_.DeleteEdge(result_loc, p.id);
        unified_store_.AddEdge(result_loc, location);

      }
    }

    // it's insert edge (loc->p) case
    // Check k-hop connectivity for occlusion detection
    unsigned hop_count = parameters.Get<unsigned>("hop_count", 2); // default: 2-hop
    
    if (hop_count == 1) {
      // 1-hop: No occlusion detection, skip neighbor checking
      continue;
    } else if (hop_count == 2) {
      // 2-hop: Original CONDA behavior - check direct neighbors of p
      std::vector<unsigned> neighbors = unified_store_.GetNeighbors(p.id);
      tsl::robin_set<unsigned> nset(neighbors.begin(), neighbors.end());
      
      for (size_t j=i+1; j<expanded_nodes.size(); ++j) {
        auto q = expanded_nodes[j];
        if (q.distance > occlude_factor[j]) {
          continue;
        }
        if (nset.find(q.id) != nset.end()) {
          float djk = unified_store_.get_distance(p.id, q.id);
          if (djk < occlude_factor[j]) {
            occlude_factor[j] = djk; // keep minimum djk
            occlude_id[j] = p.id;
          }
        }
      }
    } else if (hop_count >= 3) {
      // 3-hop: Check neighbors + neighbors of neighbors
      tsl::robin_set<unsigned> khop_neighbors;
      std::vector<unsigned> neighbors = unified_store_.GetNeighbors(p.id);
      
      // Add 1-hop neighbors
      for (auto n1 : neighbors) {
        khop_neighbors.insert(n1);
        
        // Add 2-hop neighbors (neighbors of neighbors)
        std::vector<unsigned> n1_neighbors = unified_store_.GetNeighbors(n1);
        for (auto n2 : n1_neighbors) {
          khop_neighbors.insert(n2);
        }
      }
      
      for (size_t j=i+1; j<expanded_nodes.size(); ++j) {
        auto q = expanded_nodes[j];
        if (q.distance > occlude_factor[j]) {
          continue;
        }
        if (khop_neighbors.find(q.id) != khop_neighbors.end()) {
          float djk = unified_store_.get_distance(p.id, q.id);
          if (djk < occlude_factor[j]) {
            occlude_factor[j] = djk; // keep minimum djk
            occlude_id[j] = p.id;
          }
        }
      }
    }
  }

  std::vector<unsigned> neighbors;
  for (const auto& neighbor : result) {
    if (neighbor.id == location) continue;
    neighbors.push_back(neighbor.id);
  }
 
  unified_store_.SetNeighbors(location, neighbors);

  InterInsert_with_expand(location, vector, parameters, expanded_nodes, pruned_id, maxid);
}

template <typename T>
void Index<T>::MarkDelete(unsigned tag) {
  if (this->tag_to_location_.find(tag) == this->tag_to_location_.end()) {
    throw std::runtime_error("Tag not found");
  }

  const auto location = this->tag_to_location_[tag];

  unified_store_.add_to_deleted_set(location);
  this->location_to_tag_.erase(location);
  this->tag_to_location_.erase(tag);
}

template <typename T>
void Index<T>::ConsolidateDelete(const Parameters &parameters) {
  tsl::robin_set<unsigned> old_deleted_set = unified_store_.copy_deleted_set();
  std::vector<unsigned> old_deleted_vector(old_deleted_set.begin(), old_deleted_set.end());

  std::cout << "deleted node : " << old_deleted_set.size() << std::endl;

  if (old_deleted_set.find(ep_) != old_deleted_set.end()) {
    auto pos = this->location_to_tag_.find_first();
    if (pos.is_valid()) {
      ep_ = pos._key;
    }
  }

  // [1] collect all neighbors' loc (need to update)
  std::unordered_set<unsigned> all_neighbors;

  unsigned max_location = unified_store_.get_max_location();

#pragma omp parallel for schedule(dynamic, 1) default(none) shared(all_neighbors, old_deleted_set, unified_store_, max_location)
  for (unsigned loc = 0; loc < max_location; ++loc) {
    if (unified_store_.is_in_empty_set(loc)) continue;
    if (old_deleted_set.find(loc) != old_deleted_set.end()) continue;

    bool need_to_update = false;

    std::vector<unsigned> neighbors = unified_store_.GetNeighbors(loc);
    for (auto neighbor : neighbors) {
      if (old_deleted_set.find(neighbor) != old_deleted_set.end()) {
        need_to_update = true;
        break;
      }
    }
    if (need_to_update) {
      #pragma omp critical
      {
        all_neighbors.insert(loc);
      }
    }
  }

  std::cout << "need to update : " << all_neighbors.size() << std::endl; // check log

  // [2] update neighbor's connection
  std::vector<unsigned> all_neighbors_vector(all_neighbors.begin(), all_neighbors.end());

#pragma omp parallel for schedule(dynamic, 1) default(none) shared(all_neighbors_vector, old_deleted_set, parameters)
  for (size_t idx = 0; idx < all_neighbors_vector.size(); ++idx) {
    unsigned loc = all_neighbors_vector[idx];
    RefineDelete(loc, old_deleted_set, parameters);
  }

#pragma omp parallel for schedule(dynamic, 1) default(none) shared(old_deleted_set, old_deleted_vector, unified_store_)
  for (size_t i = 0; i < old_deleted_vector.size(); ++i) {
    unsigned deleted = old_deleted_vector[i];
    unified_store_.ClearNeighbors(deleted);
    unified_store_.clear_vector(deleted);

    // [4] insert to empty_locs_
    #pragma omp critical
    {
      unified_store_.add_to_empty_set(deleted);
    }
  }

  this->nd_ -= old_deleted_set.size();
  unified_store_.erase_deleted_subset(old_deleted_set);
}

template <typename T>
void Index<T>::RefineDelete(unsigned loc,
                            const tsl::robin_set<unsigned> &old_deleted_set,
                            const Parameters &parameters) {
  const T *vector = unified_store_.get_vector(loc);
  unsigned range = parameters.Get<unsigned>("R");

  // collect new candidate pool
  tsl::robin_set<unsigned> new_set;
  std::vector<unsigned> neighbors = unified_store_.GetNeighbors(loc);

  for (auto neighbor : neighbors) {
    if (old_deleted_set.find(neighbor) == old_deleted_set.end()) {
      // add undeleted 1-hop neighbor
      if (neighbor != loc) {
        new_set.insert(neighbor);
      }
    } else {
      // add 2-hop neighbors (neighbors of deleted neighbor)
      std::vector<unsigned> neighbors_of_deleted = unified_store_.GetNeighbors(neighbor);
      for (auto neighbor_of_deleted : neighbors_of_deleted) {
        if (old_deleted_set.find(neighbor_of_deleted) == old_deleted_set.end()) {
          if (neighbor_of_deleted != loc) {
            new_set.insert(neighbor_of_deleted);
          }
        }
      }
    }
  }

  if (new_set.size() <= range) {
    unified_store_.SetNeighbors(loc, new_set);
  } else {
    std::vector<Neighbor> new_neighbors;
    new_neighbors.reserve(new_set.size());

    for (auto id : new_set) {
        float dist = unified_store_.get_distance(loc, id);
        new_neighbors.emplace_back(id, dist, true);
    }

    std::vector<Neighbor> result;
    sync_prune(vector, parameters, new_neighbors, result);
    unified_store_.SetNeighbors(loc, result);
  }
}

template <typename T>
void Index<T>::BatchSearch(const std::vector<const T*>& queries,
                            size_t k,
                            const Parameters &parameters,
                            std::vector<std::vector<unsigned>>& results) {
  int num_queries = static_cast<int>(queries.size());

#pragma omp parallel for schedule(dynamic, 1) default(none) shared(num_queries, queries, k, parameters, results)
  for (int i = 0; i < num_queries; ++i) {
    Search(queries[i], k, parameters, results[i].data());
  }
}

template <typename T>
void Index<T>::Search(const T *query, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  NeighborPriorityQueue best_L_nodes;
  tsl::robin_set<uint32_t> inserted_into_pool_rs;

  auto is_not_visited = [&inserted_into_pool_rs](const unsigned id) {
    return inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
  };

  std::vector<unsigned> init_ids;

  std::vector<unsigned> ep_neighbors = unified_store_.GetNeighbors(ep_);
  for (unsigned i=0; i < ep_neighbors.size() && i < L; ++i) {
    init_ids.push_back(ep_neighbors[i]);
  }

  best_L_nodes.reserve(L);

  for (auto id : init_ids) {
    if (is_not_visited(id)) {

      inserted_into_pool_rs.insert(id);
      float dist = unified_store_.get_distance(id, query);
      best_L_nodes.insert(Neighbor(id, dist));

    }
  }

  while (best_L_nodes.has_unexpanded_node()) {
    auto best_candidate = best_L_nodes.closest_unexpanded();
    auto n = best_candidate.id;

    std::vector<unsigned> neighbors = unified_store_.GetNeighbors(n);
    for (unsigned neighbor : neighbors) {
      if (is_not_visited(neighbor)) {
        inserted_into_pool_rs.insert(neighbor);
        float dist = unified_store_.get_distance(neighbor, query);
        best_L_nodes.insert(Neighbor(neighbor, dist));
      }
    }
  }

  size_t res_num = 0;
  for (size_t i=0; i < best_L_nodes.size(); ++i) {
    unsigned tag;
    if (this->location_to_tag_.try_get(best_L_nodes[i].id, tag)) {
      indices[res_num] = tag;
      res_num++;
      if (res_num == K) {
        break;
      }
    }
  }
}

template <typename T>
void Index<T>::BatchSearch(const std::vector<const T*>& queries,
                            size_t k,
                            unsigned L,
                            std::vector<std::vector<unsigned>>& results) {
  int num_queries = static_cast<int>(queries.size());

#pragma omp parallel for schedule(dynamic, 1) default(none) shared(num_queries, queries, k, L, results)
  for (int i = 0; i < num_queries; ++i) {
    Search(queries[i], k, L, results[i].data());
  }
}

template <typename T>
void Index<T>::Search(const T *query, size_t K,
                      unsigned L, unsigned *indices) {

  NeighborPriorityQueue best_L_nodes;
  unsigned max_location = unified_store_.get_max_location();
  boost::dynamic_bitset<uint32_t> inserted_into_pool_bs(max_location);

  auto is_not_visited = [&inserted_into_pool_bs](const unsigned id) {
    return inserted_into_pool_bs[id] == 0;
  };

  std::vector<unsigned> init_ids;

  std::vector<unsigned> ep_neighbors = unified_store_.GetNeighbors(ep_);
  for (unsigned i=0; i < ep_neighbors.size() && i < L; ++i) {
    init_ids.push_back(ep_neighbors[i]);
  }

  best_L_nodes.reserve(L);

  for (auto id : init_ids) {
    if (is_not_visited(id)) {

      inserted_into_pool_bs[id] = 1;
      float dist = unified_store_.get_distance(id, query);

      best_L_nodes.insert(Neighbor(id, dist));
    }
  }

  while (best_L_nodes.has_unexpanded_node()) {
    auto best_candidate = best_L_nodes.closest_unexpanded();
    auto n = best_candidate.id;

    unified_store_.ForEachNeighbor(n, [&](unsigned neighbor){
      if (is_not_visited(neighbor)) {
        inserted_into_pool_bs[neighbor] = 1;
        float dist = unified_store_.get_distance(neighbor, query);
        best_L_nodes.insert(Neighbor(neighbor, dist));
      }
    });
  }

  size_t res_num = 0;
  for (size_t i=0; i < best_L_nodes.size(); ++i) {
    unsigned tag;
    if (this->location_to_tag_.try_get(best_L_nodes[i].id, tag)) {
      indices[res_num] = tag; 
      res_num++;
      if (res_num == K) {
        break;
      }
    }
  }
}

// with distance
template <typename T>
void Index<T>::Search(const T *query, size_t K,
                          unsigned L, unsigned *indices, float *distances) {
  NeighborPriorityQueue best_L_nodes;
  tsl::robin_set<uint32_t> inserted_into_pool_rs;

  auto is_not_visited = [&inserted_into_pool_rs](const unsigned id) {
    return inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
  };

  std::vector<unsigned> init_ids;

  std::vector<unsigned> ep_neighbors = unified_store_.GetNeighbors(ep_);
  for (unsigned i = 0; i < ep_neighbors.size() && i < L; ++i) {
    init_ids.push_back(ep_neighbors[i]);
  }

  best_L_nodes.reserve(L);

  for (auto id : init_ids) {
    if (is_not_visited(id)) {
      inserted_into_pool_rs.insert(id);
      float dist = unified_store_.get_distance(id, query);
      best_L_nodes.insert(Neighbor(id, dist));
    }
  }

  while (best_L_nodes.has_unexpanded_node()) {
    auto best_candidate = best_L_nodes.closest_unexpanded();
    auto n = best_candidate.id;

    std::vector<unsigned> neighbors = unified_store_.GetNeighbors(n);
    for (unsigned neighbor : neighbors) {
      if (is_not_visited(neighbor)) {
        inserted_into_pool_rs.insert(neighbor);
        float dist = unified_store_.get_distance(neighbor, query);
        best_L_nodes.insert(Neighbor(neighbor, dist));
      }
    }
  }

  size_t res_num = 0;
  for (size_t i = 0; i < best_L_nodes.size(); ++i) {
    unsigned tag;
    const auto &cand = best_L_nodes[i];  // id, distance
    if (this->location_to_tag_.try_get(cand.id, tag)) {
      indices[res_num]   = tag;
      distances[res_num] = cand.distance;
      ++res_num;
      if (res_num == K) break;
    }
  }

  for (; res_num < K; ++res_num) {
    indices[res_num]   = std::numeric_limits<unsigned>::max();
    distances[res_num] = std::numeric_limits<float>::infinity();
  }
}

template <typename T>
void Index<T>::BatchSearch(const std::vector<const T*>& queries,
                               size_t k, unsigned L,
                               std::vector<std::vector<unsigned>>& indices,
                               std::vector<std::vector<float>>& distances) {
  const int num_queries = static_cast<int>(queries.size());

  if ((int)indices.size() != num_queries) indices.resize(num_queries);
  if ((int)distances.size() != num_queries) distances.resize(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    if (indices[i].size() != k)   indices[i].assign(k, std::numeric_limits<unsigned>::max());
    if (distances[i].size() != k) distances[i].assign(k, std::numeric_limits<float>::infinity());
  }

#pragma omp parallel for schedule(dynamic, 1) default(none) shared(num_queries, queries, k, L, indices, distances)
  for (int i = 0; i < num_queries; ++i) {
    Search(queries[i], k, L, indices[i].data(), distances[i].data());
  }
}





template <typename T>
void Index<T>::ConsolidateDelete_conda(const Parameters &parameters) {
  tsl::robin_set<unsigned> old_deleted_set = unified_store_.get_deleted_set();
  std::vector<unsigned> old_deleted_vector(old_deleted_set.begin(), old_deleted_set.end());

#pragma omp parallel for schedule(dynamic, 1) default(none) shared(old_deleted_set, old_deleted_vector, unified_store_)
  for (size_t i = 0; i < old_deleted_vector.size(); ++i) {
    unsigned deleted = old_deleted_vector[i];
    unified_store_.ClearNeighbors(deleted);
    unified_store_.fill_max_vector(deleted);

    #pragma omp critical
    {
      unified_store_.add_to_empty_set(deleted);
    }
  }

  this->nd_ -= old_deleted_set.size();
  if (old_deleted_set.find(ep_) != old_deleted_set.end()) {
    auto pos = this->location_to_tag_.find_first();
    if (pos.is_valid()) {
      ep_ = pos._key;
    }
  }

  unified_store_.clear_deleted_set();
}

// ----- helper functions -----
template <typename T>
void Index<T>::get_neighbors(unsigned location,
                            const T *vector,
                            const Parameters &parameters,
                            tsl::robin_set<unsigned> &inserted_into_pool_rs,
                            NeighborPriorityQueue &best_L_nodes,
                            std::vector<Neighbor> &expanded_nodes) {

  unsigned L = parameters.Get<unsigned>("L");

  auto is_not_visited = [&inserted_into_pool_rs](const unsigned id) {
    return inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
  };

  inserted_into_pool_rs.insert(location); // do not consider itself

  std::vector<unsigned> init_ids;

  // get init ids
  init_ids.push_back(ep_);

  std::vector<unsigned> ep_neighbors = unified_store_.GetNeighbors(ep_);
  for (unsigned i=0; i < ep_neighbors.size() && (i+1) < L; ++i) {
    if (ep_neighbors[i] == location) continue;
    init_ids.push_back(ep_neighbors[i]);
  }

  best_L_nodes.reserve(L);

  for (auto id : init_ids) {
    if (is_not_visited(id)) {
      inserted_into_pool_rs.insert(id);
      float dist = unified_store_.get_distance(id, vector);
      best_L_nodes.insert(Neighbor(id, dist));
    }
  }

  while (best_L_nodes.has_unexpanded_node()) {
    auto best_candidate = best_L_nodes.closest_unexpanded();
    auto n = best_candidate.id;
    expanded_nodes.emplace_back(best_candidate);

    std::vector<unsigned> neighbors = unified_store_.GetNeighbors(n);
    for (unsigned neighbor : neighbors) {
      if (is_not_visited(neighbor)) {
        inserted_into_pool_rs.insert(neighbor);
        float dist = unified_store_.get_distance(neighbor, vector);
        Neighbor nn(neighbor, dist);
        best_L_nodes.insert(nn);
      }
    }
  }
}

}  // namespace efanna2e

template class efanna2e::Index<float>;
template class efanna2e::Index<int8_t>;
template class efanna2e::Index<uint8_t>;
