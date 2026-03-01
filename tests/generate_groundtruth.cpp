#include <distance.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

enum class MetricMode { L2, IP };

template <typename T>
struct Matrix {
  std::vector<T> data;
  uint32_t num = 0;
  uint32_t dim = 0;
};

struct Candidate {
  float distance;
  uint32_t id;
};

struct WorseCandidateFirst {
  bool operator()(const Candidate& a, const Candidate& b) const {
    if (a.distance != b.distance) {
      return a.distance < b.distance;
    }
    return a.id < b.id;
  }
};

inline bool is_better(const Candidate& lhs, const Candidate& rhs) {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.id < rhs.id;
}

template <typename T>
Matrix<T> load_headered_vectors(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open vector file: " + path);
  }

  Matrix<T> mat;
  in.read(reinterpret_cast<char*>(&mat.num), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&mat.dim), sizeof(uint32_t));
  if (!in) {
    throw std::runtime_error("Failed to read header from: " + path);
  }

  const size_t count = static_cast<size_t>(mat.num) * mat.dim;
  mat.data.resize(count);
  in.read(reinterpret_cast<char*>(mat.data.data()),
          static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) {
    throw std::runtime_error("Failed to read payload from: " + path);
  }
  return mat;
}

void write_ivecs(const std::string& path,
                 const std::vector<uint32_t>& ids,
                 uint32_t rows,
                 uint32_t k) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open GT file for writing: " + path);
  }

  for (uint32_t row = 0; row < rows; ++row) {
    out.write(reinterpret_cast<const char*>(&k), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(ids.data() + static_cast<size_t>(row) * k),
              static_cast<std::streamsize>(k * sizeof(uint32_t)));
  }
}

template <typename T>
const efanna2e::Distance<T>& get_distance_impl(MetricMode metric) {
  static const efanna2e::DistanceL2<T> l2;
  static const efanna2e::DistanceInnerProduct<T> ip;
  return (metric == MetricMode::L2)
             ? static_cast<const efanna2e::Distance<T>&>(l2)
             : static_cast<const efanna2e::Distance<T>&>(ip);
}

MetricMode parse_metric(const std::string& s) {
  if (s == "L2") {
    return MetricMode::L2;
  }
  if (s == "IP") {
    return MetricMode::IP;
  }
  throw std::runtime_error("Unsupported metric: " + s);
}

void ensure_directory(const std::string& path) {
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec) {
    throw std::runtime_error("Failed to create directory: " + path);
  }
}

template <typename T>
void compute_gt_for_active_set(const Matrix<T>& base,
                               const Matrix<T>& queries,
                               const std::vector<uint32_t>& active_ids,
                               MetricMode metric,
                               uint32_t k,
                               const std::string& gt_path) {
  if (active_ids.size() < k) {
    throw std::runtime_error("Active point count is smaller than K for " + gt_path);
  }

  const auto& distance = get_distance_impl<T>(metric);
  std::vector<uint32_t> gt(static_cast<size_t>(queries.num) * k);

#pragma omp parallel for schedule(dynamic, 1)
  for (int qi = 0; qi < static_cast<int>(queries.num); ++qi) {
    const T* query = queries.data.data() + static_cast<size_t>(qi) * queries.dim;
    std::priority_queue<Candidate, std::vector<Candidate>, WorseCandidateFirst> heap;

    for (uint32_t tag : active_ids) {
      const T* vec = base.data.data() + static_cast<size_t>(tag) * base.dim;
      Candidate cand{distance.compare(query, vec, base.dim), tag};

      if (heap.size() < k) {
        heap.push(cand);
      } else if (is_better(cand, heap.top())) {
        heap.pop();
        heap.push(cand);
      }
    }

    std::vector<Candidate> best;
    best.reserve(k);
    while (!heap.empty()) {
      best.push_back(heap.top());
      heap.pop();
    }
    std::sort(best.begin(), best.end(), [](const Candidate& a, const Candidate& b) {
      if (a.distance != b.distance) {
        return a.distance < b.distance;
      }
      return a.id < b.id;
    });

    uint32_t* out = gt.data() + static_cast<size_t>(qi) * k;
    for (uint32_t i = 0; i < k; ++i) {
      out[i] = best[i].id;
    }
  }

  write_ivecs(gt_path, gt, queries.num, k);
}

template <typename T>
int generate_single_gt(const std::string& base_path,
                       const std::string& query_path,
                       const std::string& gt_path,
                       MetricMode metric,
                       uint32_t k) {
  const Matrix<T> base = load_headered_vectors<T>(base_path);
  const Matrix<T> queries = load_headered_vectors<T>(query_path);
  if (base.dim != queries.dim) {
    throw std::runtime_error("Base/query dimension mismatch");
  }

  std::vector<uint32_t> active_ids(base.num);
  for (uint32_t i = 0; i < base.num; ++i) {
    active_ids[i] = i;
  }
  compute_gt_for_active_set(base, queries, active_ids, metric, k, gt_path);
  return 0;
}

void parse_runbook(const std::string& runbook_path,
                   const std::string& dataset_key,
                   int& step_size,
                   std::vector<std::pair<int, YAML::Node>>& operations) {
  YAML::Node root = YAML::LoadFile(runbook_path);
  if (!root[dataset_key]) {
    throw std::runtime_error("Dataset key not found in runbook: " + dataset_key);
  }
  YAML::Node dataset = root[dataset_key];
  step_size = dataset["step_size"].as<int>();
  operations.reserve(step_size);
  for (int i = 1; i <= step_size; ++i) {
    operations.emplace_back(i, dataset[i]);
  }
}

template <typename T>
int generate_streaming_gt(const std::string& base_path,
                          const std::string& query_path,
                          const std::string& gt_dir,
                          MetricMode metric,
                          uint32_t k,
                          const std::string& runbook_path,
                          const std::string& dataset_key) {
  const Matrix<T> base = load_headered_vectors<T>(base_path);
  const Matrix<T> queries = load_headered_vectors<T>(query_path);
  if (base.dim != queries.dim) {
    throw std::runtime_error("Base/query dimension mismatch");
  }

  std::vector<std::pair<int, YAML::Node>> operations;
  int step_size = 0;
  parse_runbook(runbook_path, dataset_key, step_size, operations);

  std::vector<uint32_t> active_ids;
  active_ids.reserve(base.num);
  std::vector<int64_t> positions(base.num, -1);

  ensure_directory(gt_dir);

  for (const auto& op_entry : operations) {
    const int step_id = op_entry.first;
    const YAML::Node& step_node = op_entry.second;
    const std::string op = step_node["operation"].as<std::string>();
    if (op == "insert") {
      const uint32_t start = step_node["start"].as<uint32_t>();
      const uint32_t end = step_node["end"].as<uint32_t>();
      if (end > base.num) {
        throw std::runtime_error("Insert range exceeds base vector count");
      }
      for (uint32_t tag = start; tag < end; ++tag) {
        if (positions[tag] != -1) {
          continue;
        }
        positions[tag] = static_cast<int64_t>(active_ids.size());
        active_ids.push_back(tag);
      }
    } else if (op == "delete") {
      const uint32_t start = step_node["start"].as<uint32_t>();
      const uint32_t end = step_node["end"].as<uint32_t>();
      if (end > base.num) {
        throw std::runtime_error("Delete range exceeds base vector count");
      }
      for (uint32_t tag = start; tag < end; ++tag) {
        const int64_t pos = positions[tag];
        if (pos == -1) {
          continue;
        }
        const uint32_t tail_tag = active_ids.back();
        active_ids[static_cast<size_t>(pos)] = tail_tag;
        positions[tail_tag] = pos;
        active_ids.pop_back();
        positions[tag] = -1;
      }
    } else if (op == "search") {
      const std::string gt_path =
          gt_dir + "/step" + std::to_string(step_id) + ".gt100";
      std::cout << "Generating GT for step " << step_id
                << " with " << active_ids.size()
                << " active points -> " << gt_path << '\n';
      compute_gt_for_active_set(base, queries, active_ids, metric, k, gt_path);
    } else {
      throw std::runtime_error("Unsupported runbook operation: " + op);
    }
  }

  return 0;
}

template <typename T>
int dispatch(const std::string& base_path,
             const std::string& query_path,
             const std::string& out_path,
             const std::string& dtype,
             MetricMode metric,
             uint32_t k,
             int argc,
             char** argv) {
  if (argc == 7) {
    return generate_single_gt<T>(base_path, query_path, out_path, metric, k);
  }
  return generate_streaming_gt<T>(base_path, query_path, out_path, metric, k, argv[7], argv[8]);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7 && argc != 9) {
    std::cerr << "Usage:\n"
              << "  " << argv[0]
              << " <base_file> <query_file> <gt_file> <dtype> <metric> <k>\n"
              << "  " << argv[0]
              << " <base_file> <query_file> <gt_dir> <dtype> <metric> <k> <runbook_yaml> <dataset_key>\n"
              << "  dtype: float|u8|i8\n"
              << "  metric: L2|IP\n";
    return 1;
  }

  try {
    const std::string base_path = argv[1];
    const std::string query_path = argv[2];
    const std::string out_path = argv[3];
    const std::string dtype = argv[4];
    const MetricMode metric = parse_metric(argv[5]);
    const uint32_t k = static_cast<uint32_t>(std::stoul(argv[6]));

    if (dtype == "float") {
      return dispatch<float>(base_path, query_path, out_path, dtype, metric, k, argc, argv);
    }
    if (dtype == "u8") {
      return dispatch<uint8_t>(base_path, query_path, out_path, dtype, metric, k, argc, argv);
    }
    if (dtype == "i8") {
      return dispatch<int8_t>(base_path, query_path, out_path, dtype, metric, k, argc, argv);
    }

    throw std::runtime_error("Unsupported dtype: " + dtype);
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << '\n';
    return 1;
  }
}
