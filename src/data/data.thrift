namespace cpp2 recsys.knn_server

namespace py recsys.knn_server

struct EmbeddingSearchResult {
  1: i64 id;
  2: double distance;
}

enum EmbeddingDataType {
  Float64_t = 1,
  Float32_t = 2,
}