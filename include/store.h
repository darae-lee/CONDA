#include <vector>
#include <shared_mutex>
#include <memory>
#include <algorithm>
#include <atomic> 
#include "neighbor.h"
#include "parameters.h"
#include "distance.h"
#include <thread>
#include <omp.h>
#include <unordered_set>
#include "natural_number_set.h"
#include <limits>

namespace efanna2e {

template <typename T>
class GraphDataStore {
public:
	void init_graphstore(size_t num_nodes, const Parameters& params) {
		range = params.Get<unsigned>("R");

		graph_store_.resize(num_nodes);
		graph_locks_.resize(num_nodes);
		for (size_t i = 0; i < num_nodes; ++i) {
			graph_store_[i].reserve(range);
			graph_locks_[i] = std::make_unique<std::shared_mutex>();
		}

		empty_locs_.reserve(num_nodes);
		std::cout << "init graphstore size of " << graph_store_.size() << std::endl; 
	}

	void extend_stores(unsigned max_nd_) {
		unsigned new_max = max_nd_ * 2;
		graph_store_.resize(new_max);
		graph_locks_.resize(new_max);
		for (size_t i = max_nd_; i < new_max; ++i) {
			graph_store_[i].reserve(range);
			graph_locks_[i] = std::make_unique<std::shared_mutex>();
		}
		empty_locs_.reserve(new_max);
	}

	void init_datastore(size_t num, size_t dim) {
		data_store_ = (T*)std::aligned_alloc(32, num * dim * sizeof(T));
		dimension_ = dim;
		std::cout << "init dataset dimension of " << dimension_ << std::endl; 
	}

	void init_distance(Metric metric) {
		switch (metric) {
			case L2:
				distance_ = new DistanceL2<T>();
				break;
			case INNER_PRODUCT:
				distance_ = new DistanceInnerProduct<T>();
				break;
			case FAST_L2:
				distance_ = new DistanceFastL2<T>();
				break;
			case COSINE:
				distance_ = new DistanceCosine<T>();
				break;
			default:
				throw std::invalid_argument("Unsupported metric type");
		}
	}

	size_t get_compare_call() const {
		return DistanceBase::compare_count.load();
	}

	void reset_compare_count() {
	    DistanceBase::reset_compare_count();
	}


	void SetNeighbors(unsigned location, std::vector<unsigned> neighbors) {
		std::unique_lock<std::shared_mutex> lock(*graph_locks_[location]);
		graph_store_[location] = std::move(neighbors);
	}

	void SetNeighbors(unsigned location, tsl::robin_set<unsigned> neighbors) {
		std::vector<unsigned> neighbor_vector(neighbors.begin(), neighbors.end());
		SetNeighbors(location, std::move(neighbor_vector));
	}

	void SetNeighbors(unsigned location, const std::vector<Neighbor>& neighbors) {
		std::vector<unsigned> neighbor_ids;
		neighbor_ids.reserve(neighbors.size());

		for (const auto& neighbor : neighbors) {
			neighbor_ids.push_back(neighbor.id);
		}
		
		SetNeighbors(location, std::move(neighbor_ids));
	}

	void SetNeighbors(unsigned location, const std::vector<SimpleNeighbor>& neighbors) {
		std::vector<unsigned> neighbor_ids;
		neighbor_ids.reserve(neighbors.size());

		for (const auto& neighbor : neighbors) {
			neighbor_ids.push_back(neighbor.id);
		}

		SetNeighbors(location, std::move(neighbor_ids));
	}

	void ClearNeighbors(unsigned location) {
		std::unique_lock<std::shared_mutex> lock(*graph_locks_[location]);
		graph_store_[location].clear();
	}

// *** InterInsert
// less then inrange -> use AddEdge
	void AddEdge(unsigned from, unsigned to) {
		std::unique_lock<std::shared_mutex> lock(*graph_locks_[from]);
		graph_store_[from].push_back(to);
	}

	void DeleteEdge(unsigned from, unsigned to) {
		std::unique_lock<std::shared_mutex> lock(*graph_locks_[from]);
		auto& neighbors = graph_store_[from];
		auto it = std::remove(neighbors.begin(), neighbors.end(), to);
		if (it != neighbors.end()) {
			neighbors.erase(it, neighbors.end());
		}
	}

	// GraphStore Method
	std::vector<unsigned> GetNeighbors(unsigned node) const {
		std::shared_lock<std::shared_mutex> lock(*graph_locks_[node]);
		return graph_store_[node];
	}

	template <typename F>
	void ForEachNeighbor(unsigned node, F&& f) const {
	  std::shared_lock<std::shared_mutex> lock(*graph_locks_[node]);
	  const auto& nbrs = graph_store_[node];
	  for (unsigned v : nbrs) f(v);
	}

// *** DataStore
	void set_vector(unsigned location, const T* vector) {
		std::memset(data_store_ + location * dimension_, 0, dimension_ * sizeof(T));
		std::memcpy(data_store_ + location * dimension_, vector, dimension_ * sizeof(T));
	}

	const T* get_vector(unsigned location) const {
		return data_store_ + location * (size_t)dimension_;
	}

	void clear_vector(unsigned location) {
    	std::fill(data_store_ + location * dimension_, data_store_ + (location + 1) * dimension_, 0.0f);
	}

	void fill_max_vector(unsigned location) {
		std::fill(
			data_store_ + location * dimension_,
			data_store_ + (location + 1) * dimension_,
			std::numeric_limits<T>::max()
		);
	}

	float get_distance(unsigned node1, unsigned node2) const {
		const T* vec1 = get_vector(node1);
		const T* vec2 = get_vector(node2);
		return distance_->compare(vec1, vec2, (unsigned)dimension_);
	}

	float get_distance(unsigned node, const T* query) const {
		const T* vec = get_vector(node);
		return distance_->compare(vec, query, dimension_);
	}

	float get_distance(const T* query, unsigned node) const {
		const T* vec = get_vector(node);
		return distance_->compare(vec, query, dimension_);
	}

	float get_distance(const T* vector1, const T* vector2) const {
		return distance_->compare(vector1, vector2, dimension_);
	}

	size_t get_dimension() {
		return dimension_;
	}

	void add_to_deleted_set(unsigned loc) {
		std::unique_lock<std::shared_mutex> lock(delete_lock_);
		deleted_set_.insert(loc);
	}

	void swap_deleted_set(tsl::robin_set<unsigned>& target_set) {
		std::unique_lock<std::shared_mutex> lock(delete_lock_);
		std::swap(deleted_set_, target_set);
	}

	const tsl::robin_set<unsigned>& copy_deleted_set() const {
		std::shared_lock<std::shared_mutex> lock(delete_lock_);
		return deleted_set_;
	}

	const tsl::robin_set<unsigned>& get_deleted_set() const {
		std::shared_lock<std::shared_mutex> lock(delete_lock_);
		return deleted_set_;
	}

	bool is_deleted_set_empty() const {
		return deleted_set_.empty();
	}

	void clear_deleted_set() {
		std::unique_lock<std::shared_mutex> lock(delete_lock_);
		deleted_set_.clear();
	}
	void erase_deleted_subset(const tsl::robin_set<unsigned>& subset) {
		std::unique_lock<std::shared_mutex> lock(delete_lock_);
		for (unsigned id : subset) {
		  deleted_set_.erase(id);
		}
	}
	void add_to_empty_set(unsigned loc) {
		std::unique_lock<std::shared_mutex> lock(empty_lock_);
		empty_locs_.insert(loc);
	}
	bool is_in_empty_set(unsigned loc) {
		std::shared_lock<std::shared_mutex> lock(empty_lock_);
		return empty_locs_.is_in_set(loc);
	}
	bool is_empty_set_empty() {
		std::shared_lock<std::shared_mutex> lock(empty_lock_);
		return empty_locs_.is_empty();
	}
	unsigned reserve_location() {
		if (empty_locs_.is_empty()) {
			return max_location_++;
		}
		unsigned new_loc = empty_locs_.pop_any();
		return new_loc;
	}
	
	unsigned get_max_location() {
		return max_location_;
	}

private:
	// *** Parameter
	Parameters parameters;
	unsigned range;

	// *** GraphStore
	std::vector<std::vector<unsigned>> graph_store_;
	std::vector<std::unique_ptr<std::shared_mutex>> graph_locks_;

	// *** DataStore
	T *data_store_ = nullptr;
	size_t dimension_;
	Distance<T>* distance_ = nullptr;

	tsl::robin_set<unsigned> deleted_set_;
	natural_number_set<unsigned> empty_locs_;
	size_t max_location_ = 0;

	mutable std::shared_mutex delete_lock_;
	mutable std::shared_mutex empty_lock_;
};

}