#ifndef INDEX_TOP_H
#define INDEX_TOP_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "dynamic_index.h"
#include <cassert>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>
#include "tsl/robin_set.h"

#include "store.h"

namespace efanna2e {

template <typename T>
class Index : public DynamicIndex<T> {
 public:
  explicit Index(
      const size_t dimension,
      const size_t max_num_points,
      Metric metric,
      DynamicIndex<T> *initializer);

  virtual ~Index();

  virtual void Init(const Parameters &parameters) override;

  virtual void Insert(unsigned tag, const T *vector, const Parameters &parameters) override;
  virtual void BatchInsert(const std::vector<unsigned>& tags, const std::vector<const T*>& vectors, const Parameters& parameters) override;
  virtual void MarkDelete(unsigned tag) override;
  virtual void ConsolidateDelete(const Parameters &parameters) override;

  virtual void Search(
      const T *query,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;
  virtual void BatchSearch(
      const std::vector<const T*>& queries,
      size_t k,
      const Parameters &parameters,
      std::vector<std::vector<unsigned>>& results) override;

    void Insert_crng(unsigned tag,
        const T *vector,
        const Parameters &parameters,
        bool is_delete);
    void BatchInsert_crng(const std::vector<unsigned>& tags, const std::vector<const T*>& vectors, const Parameters& parameters);

  void Search(
      const T *query,
      size_t k,
      unsigned L,
      unsigned *indices);
  void BatchSearch(
      const std::vector<const T*>& queries,
      size_t k,
      unsigned L,
      std::vector<std::vector<unsigned>>& results);

    // with distance
    void Search(
        const T *query,
        size_t k,
        unsigned L,
        unsigned *indices,
        float *distances);
    void BatchSearch(
        const std::vector<const T*>& queries,
        size_t k,
        unsigned L,
        std::vector<std::vector<unsigned>>& indices,
        std::vector<std::vector<float>>& distances);

    void ConsolidateDelete_conda(const Parameters &parameters);

    void Process_ip(unsigned tag,
                const Parameters &parameters,
                const tsl::robin_set<unsigned> &deleted_set);
    void Process_crng(unsigned tag,
        const Parameters &parameters,
        const tsl::robin_set<unsigned> &deleted_set);
    void MultiProcess_ip(const Parameters &parameters);
    void MultiProcess_crng(const Parameters &parameters);
    void ConsolidateDelete_ip(const Parameters &parameters);

  protected:
    DynamicIndex<T> *initializer_ = nullptr;
    void get_neighbors(unsigned location,
        const T *vector,
        const Parameters &parameters,
        tsl::robin_set<unsigned> &inserted_into_pool_rs,
        NeighborPriorityQueue &best_L_nodes,
        std::vector<Neighbor> &expanded_nodes);
    void get_neighbors_lazy(unsigned location,
        const T *vector,
        const Parameters &parameters,
        tsl::robin_set<unsigned> &inserted_into_pool_rs,
        NeighborPriorityQueue &best_L_nodes,
        std::vector<Neighbor> &expanded_nodes);
    void get_neighbors(unsigned location,
        const T *vector,
        unsigned L,
        tsl::robin_set<unsigned> &inserted_into_pool_rs,
        NeighborPriorityQueue &best_L_nodes,
        std::vector<Neighbor> &expanded_nodes);
    void get_neighbors_from_loc(unsigned location,
        const T *vector,
        unsigned L,
        tsl::robin_set<unsigned> &inserted_into_pool_rs,
        NeighborPriorityQueue &best_L_nodes,
        std::vector<Neighbor> &expanded_nodes);
    void get_neighbors(unsigned location,
        const T *vector,
        const Parameters &parameters,
        tsl::robin_set<unsigned> &inserted_into_pool_rs,
        NeighborPriorityQueue &best_L_nodes);
    void sync_prune(const T *vector,
        const Parameters &parameters,
        std::vector<Neighbor> &pool,
        std::vector<Neighbor> &result);
    void sync_prune(const T *vector,
        const Parameters &parameters,
        std::vector<SimpleNeighbor> &pool,
        std::vector<SimpleNeighbor> &result);
    void sync_prune(const T *vector,
        const Parameters &parameters,
        std::vector<SimpleNeighbor> &pool,
        std::vector<SimpleNeighbor> &result,
        unsigned location,
        bool &is_connect);
    void sync_prune_crng(const T *vector,
        const Parameters &parameters,
        std::vector<Neighbor> &pool,
        std::vector<Neighbor> &result);
    void sync_prune(const T *vector,
        const Parameters &parameters,
        NeighborPriorityQueue &pool,
        std::vector<Neighbor> &result);
    void InterInsert(unsigned location,
        const T *vector,
        const Parameters &parameters,
        const std::vector<Neighbor> &result);

    size_t reserve_location();

    void RefineDelete(unsigned loc,
                    const tsl::robin_set<unsigned>& old_deleted_set,
                    const Parameters &parameters);
                    
    void InterInsert_with_expand(unsigned location,
        const T *vector,
        const Parameters &parameters,
        std::vector<Neighbor> &result,
        std::vector<unsigned> &pruned_id,
        unsigned maxid);

    GraphDataStore<T> unified_store_;
    Metric metric_;

    // mutex
    std::shared_mutex tag_lock_;

  private:
    unsigned width; // max degree on graph
    unsigned ep_ = 0;
};
}

#endif //INDEX_TOP_H
