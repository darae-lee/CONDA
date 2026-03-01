#include <index.h>
#include <util.h>

#include <yaml-cpp/yaml.h>
#include <vector>
#include <map>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>

// ======== Metric alias (assumes efanna2e::Metric enum) ========
using efanna2e::L2;
using efanna2e::INNER_PRODUCT;

// ================== I/O helpers (templated) ====================

// bigann-like loader: header [uint32 num][uint32 dim] then contiguous matrix of T
template <typename T>
void load_bigann_headered(const char* filename, T*& data, unsigned& num, unsigned& dim) {
  std::ifstream in(filename, std::ios::binary | std::ios::ate);
  if (!in.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    throw std::runtime_error("File open failed");
  }
  std::streamsize file_size = in.tellg();
  in.seekg(0, std::ios::beg);

  in.read(reinterpret_cast<char*>(&num), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
  if (!in) throw std::runtime_error("Failed reading (num, dim) header");

  const size_t expected = 8 + static_cast<size_t>(num) * dim * sizeof(T);
  if (static_cast<size_t>(file_size) != expected) {
    std::cerr << "Warning: file size mismatch! expected " << expected
              << " bytes, got " << static_cast<size_t>(file_size) << " bytes.\n";
  }

  data = new T[static_cast<size_t>(num) * dim];
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(static_cast<size_t>(num) * dim * sizeof(T)));
  if (!in) throw std::runtime_error("Failed reading matrix payload");
  in.close();

  std::cout << "Loaded bigann-headered: num=" << num << " dim=" << dim
            << " elem_size=" << sizeof(T) << " bytes\n";
}

// unify loader using yaml dataset_type
template <typename T>
void load_vectors(const std::string& dataset_type,
                  const char* data_path,
                  const char* query_path,
                  T*& data, unsigned& points_num, unsigned& dim,
                  T*& query, unsigned& query_num, unsigned& query_dim) {
  load_bigann_headered<T>(data_path,  data,  points_num, dim);
  load_bigann_headered<T>(query_path, query, query_num,  query_dim);

  if (dim != query_dim) {
    std::cerr << "Warning: data dim(" << dim << ") != query dim(" << query_dim << ")\n";
  }
}

// ================= YAML parsing ===================
void parse_yaml(const std::string& yaml_file,
                const std::string& dataset,
                int& max_pts,
                int& step_size,
                std::string& dataset_type,
                std::vector<std::pair<int, YAML::Node>>& operations) {
  YAML::Node config = YAML::LoadFile(yaml_file);
  if (!config[dataset]) {
    throw std::runtime_error("Dataset key not found in YAML: " + dataset);
  }

  YAML::Node steps = config[dataset];
  max_pts      = steps["max_pts"].as<int>();
  step_size    = steps["step_size"].as<int>();
  dataset_type = steps["type"].as<std::string>();

  int derived_max_pts = 0;
  for (int i = 1; i <= step_size; i++) {
    YAML::Node step = steps[i];
    operations.emplace_back(i, step);
    if (step["operation"].as<std::string>() == "insert" && step["end"]) {
      derived_max_pts = std::max(derived_max_pts, step["end"].as<int>());
    }
  }

  if (derived_max_pts > max_pts) max_pts = derived_max_pts;
}








// ============== result saver (unchanged) =================
void save_result(const std::string& filename,
                 std::vector<std::vector<unsigned>>& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

// ================== enums & parsers ==================
enum class IndexMode { FV, IV, CONDA };
enum class MetricMode { L2, IP };

static inline IndexMode parse_index_mode(const std::string& s) {
  if (s == "FV")    return IndexMode::FV;
  if (s == "IV")    return IndexMode::IV;
  if (s == "CONDA") return IndexMode::CONDA;
  throw std::runtime_error("Unknown index mode: " + s);
}

static inline MetricMode parse_metric(const std::string& s) {
  if (s == "L2") return MetricMode::L2;
  if (s == "IP") return MetricMode::IP;
  throw std::runtime_error("Unknown metric: " + s);
}

// ================== main runner ==================
template <typename T>
int run_once(int argc, char** argv) {
  // argv:
  //  0: prog
  //  1: yaml_path
  //  2: dataset(key)
  //  3: data_path
  //  4: query_path
  //  5: buildL
  //  6: R
  //  7: C
  //  8: alpha
  //  9: searchL
  // 10: searchK
  // 11: save_folder
  // 12: dtype    (parsed outside, here we just consume)
  // 13: index    (FV|IV|CONDA)
  // 14: metric  (L2|IP)

  if (argc != 15) {
    std::cout << "Usage:\n  " << argv[0]
              << " yaml_path dataset data_path query_path "
              << "buildL R C alpha searchL searchK save_folder "
              << "dtype index metric\n"
              << "  dtype:  float|u8|i8\n"
              << "  index:  FV|IV|CONDA\n"
              << "  metric: L2|IP\n";
    return -1;
  }

  std::string yaml_path = argv[1];
  std::string dataset   = argv[2];
  const char* data_path  = argv[3];
  const char* query_path = argv[4];
  unsigned build_L  = (unsigned)atoi(argv[5]);
  unsigned R        = (unsigned)atoi(argv[6]);
  unsigned C        = (unsigned)atoi(argv[7]);
  float alpha       = (float)atof(argv[8]);
  unsigned search_L = (unsigned)atoi(argv[9]);
  unsigned search_K = (unsigned)atoi(argv[10]);
  std::string save_folder = argv[11];
  std::string dtype_str   = argv[12];
  std::string index_str   = argv[13];
  std::string metric_str  = argv[14];

  // Make folder if needed
  struct stat info;
  if (stat(save_folder.c_str(), &info) != 0) {
    if (mkdir(save_folder.c_str(), 0777) != 0) {
      std::cerr << "Error: Could not create directory " << save_folder << std::endl;
      return -1;
    }
  } else if (!(info.st_mode & S_IFDIR)) {
    std::cerr << "Error: " << save_folder << " is not a directory\n";
    return -1;
  }

  // YAML parse
  std::vector<std::pair<int, YAML::Node>> operations;
  int max_pts = 0, step_size = 0;
  std::string dataset_type;
  parse_yaml(yaml_path, dataset, max_pts, step_size, dataset_type, operations);

  // Load vectors
  T* data_load  = nullptr; unsigned points_num = 0, dim = 0;
  T* query_load = nullptr; unsigned query_num = 0, query_dim = 0;
  load_vectors<T>(dataset_type, data_path, query_path,
                  data_load, points_num, dim,
                  query_load, query_num, query_dim);

  // Parse modes
  IndexMode  mode = parse_index_mode(index_str);
  MetricMode mm   = parse_metric(metric_str);

  auto metric = (mm == MetricMode::L2)
                  ? efanna2e::L2
                  : efanna2e::INNER_PRODUCT;

  efanna2e::Index<T> index(dim, (size_t)max_pts, metric, nullptr);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L", build_L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<unsigned>("L_search", search_L);
  paras.Set<unsigned>("hop_count", 2);

  std::cout << "alpha: " << alpha
            << "  index: " << index_str
            << "  metric: " << metric_str
            << "  dtype: "  << dtype_str << std::endl;

  index.Init(paras);

  // Prepare data vectors
  std::vector<std::vector<T>> data_vectors(points_num, std::vector<T>(dim));
  for (size_t i = 0; i < points_num; ++i) {
    std::memcpy(data_vectors[i].data(), data_load + i * dim, dim * sizeof(T));
  }
  std::vector<const T*> vectors;
  vectors.reserve(points_num);
  for (const auto& v : data_vectors) vectors.push_back(v.data());

  // Prepare query vectors
  std::vector<std::vector<T>> query_vectors(query_num, std::vector<T>(dim));
  for (size_t i = 0; i < query_num; ++i) {
    std::memcpy(query_vectors[i].data(), query_load + i * dim, dim * sizeof(T));
  }
  std::vector<const T*> queries;
  queries.reserve(query_num);
  for (const auto& v : query_vectors) queries.push_back(v.data());

  for (const auto& op : operations) {
    const int step_id = op.first;
    const YAML::Node& params = op.second;
    const std::string op_name = params["operation"].as<std::string>();

    if (op_name == "search") {
      std::cout << step_id << " : search\n";

      std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(search_K));
      
      auto start = std::chrono::high_resolution_clock::now();
      index.BatchSearch(queries, search_K, search_L, res);
      auto end = std::chrono::high_resolution_clock::now();
      
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "Search execution time (Ls = " << search_L << "): "
                << elapsed.count() << " seconds\n";
      
      std::string filename =
          save_folder + "/" + std::to_string(step_id) + "-Ls" +
          std::to_string(search_L) + ".res";
      save_result(filename, res);
      
    } else if (op_name == "insert") {
      unsigned starti = params["start"].as<int>();
      unsigned endi   = params["end"].as<int>();
      std::cout << step_id << " : insert " << starti << " ~ " << endi << std::endl;

      std::vector<unsigned> tags;
      std::vector<const T*> batch_vectors;
      tags.reserve(endi - starti);
      batch_vectors.reserve(endi - starti);
      for (unsigned i = starti; i < endi; ++i) {
        tags.push_back(i);
        batch_vectors.push_back(vectors[i]);
      }

      // index.Load(save_folder.c_str(), paras);

      auto start = std::chrono::high_resolution_clock::now();
      // Insert dispatch
      if (mode == IndexMode::CONDA) {
        index.BatchInsert_crng(tags, batch_vectors, paras);
      } else {
        index.BatchInsert(tags, batch_vectors, paras);
      }
      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> elapsed = end - start;
      std::cout << "BatchInsert execution time: " << elapsed.count() << " seconds\n";
      // index.Save(save_folder.c_str());

    } else if (op_name == "delete") {
      unsigned starti = params["start"].as<int>();
      unsigned endi   = params["end"].as<int>();
      std::cout << step_id << " : delete " << starti << " ~ " << endi << std::endl;

      auto start = std::chrono::high_resolution_clock::now();

      for (unsigned i = starti; i < endi; ++i) {
        index.MarkDelete(i); // tag
      }

      // Delete dispatch
      if (mode == IndexMode::FV) {
        index.ConsolidateDelete(paras);
      } else if (mode == IndexMode::IV) {
        index.MultiProcess_ip(paras);
        index.ConsolidateDelete_ip(paras);
      } else /* CONDA */ {
        index.MultiProcess_crng(paras);
        index.ConsolidateDelete_conda(paras);
      }

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "Delete execution time: " << elapsed.count() << " seconds\n";
    }
  }

  // cleanup raw buffers
  delete[] data_load;
  delete[] query_load;
  return 0;
}

// ===================== dtype dispatcher ======================
int main(int argc, char** argv) {
  if (argc != 15) {
    std::cout << "Usage:\n  " << argv[0]
              << " yaml_path dataset data_path query_path "
              << "buildL R C alpha searchL searchK save_folder "
              << "dtype index metric\n";
    return -1;
  }

  const std::string dtype = argv[12];

  try {
    if (dtype == "float") return run_once<float>(argc, argv);
    if (dtype == "u8")    return run_once<uint8_t>(argc, argv);
    if (dtype == "i8")    return run_once<int8_t>(argc, argv);
    throw std::runtime_error("Unknown dtype: " + dtype);
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << std::endl;
    return -1;
  }
}
