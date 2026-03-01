#include "dynamic_index.h"
#include "util.h"

namespace efanna2e {

template <typename T>
DynamicIndex<T>::DynamicIndex(const size_t dimension, const size_t max_num_points)
    : dimension_(calculate_aligned_dimension(dimension)), max_nd_(max_num_points) {
}

template <typename T>
DynamicIndex<T>::~DynamicIndex() {
}



template <typename T>
unsigned DynamicIndex<T>::get_tag_by_loc(unsigned loc) const {
    unsigned tag;
    if (this->location_to_tag_.try_get(loc, tag)) {
        return tag;
    } else {
        throw std::runtime_error("Location not found in location_to_tag map");
    }
}

template <typename T>
unsigned DynamicIndex<T>::get_loc_by_tag(unsigned tag) const {
    auto it = this->tag_to_location_.find(tag);
    if (it != this->tag_to_location_.end()) {
        return it->second;
    } else {
        throw std::runtime_error("Tag not found in tag_to_location map");
    }
}

}  // namespace efanna2e

// Template instantiations
template class efanna2e::DynamicIndex<float>;
template class efanna2e::DynamicIndex<int8_t>;
template class efanna2e::DynamicIndex<uint8_t>;
