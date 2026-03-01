#ifndef DYNAMIC_INDEX_H
#define DYNAMIC_INDEX_H

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include "distance.h"
#include "parameters.h"
#include <mutex>
#include <shared_mutex>

#include "tsl/sparse_map.h"
#include "tsl/robin_map.h"

#include "natural_number_map.h"

namespace efanna2e {

template <typename T>
class DynamicIndex {
 public:
  explicit DynamicIndex(const size_t dimension, const size_t max_num_points);
  virtual ~DynamicIndex();


  unsigned get_tag_by_loc(unsigned loc) const;
  unsigned get_loc_by_tag(unsigned tag) const;

  virtual void Init(const Parameters &parameters) = 0;
  virtual void Insert(unsigned tag, const T *vector, const Parameters &parameters) = 0;
  virtual void BatchInsert(const std::vector<unsigned>& tags, const std::vector<const T*>& vectors, const Parameters& parameters) = 0;
  virtual void MarkDelete(unsigned tag) = 0;
  virtual void ConsolidateDelete(const Parameters &parameters) = 0;
  virtual void Search(const T *query, size_t k, const Parameters &parameters, unsigned *indices) = 0;
  virtual void BatchSearch(const std::vector<const T*>& queries, size_t k, const Parameters &parameters, std::vector<std::vector<unsigned>>& results) = 0;

  inline size_t GetDimension() const { return dimension_; }
  inline size_t GetMaxSizeOfData() const { return max_nd_; }
  inline size_t GetSizeOfData() const { return nd_; }

 protected:
  const size_t dimension_;
  tsl::sparse_map<unsigned, unsigned> tag_to_location_;
  natural_number_map<unsigned, unsigned> location_to_tag_;

  size_t max_nd_; 
  size_t nd_ = 0; 
};

}  // namespace efanna2e

#endif  // DYNAMIC_INDEX_H