include "src/data/data.thrift"

namespace py recsys.knn_server
namespace cpp2 recsys.knn_server


struct QueryRequest {
  1: binary raw_query_vector;
  2: data.EmbeddingDataType type;
  3: i64 library_id;
  4: i64 embedding_dim;
  5: i64 n_closest;
  6: bool include_vectors = false;
}

enum ResponseStatus { 
  StatusOk = 1,
  StatusFailed = 2
}

struct QueryResponse {
  1: ResponseStatus status;
  2: i64 micros_passed;
  3: list<data.EmbeddingSearchResult> results;
}

service KnnSearchService {
    QueryResponse search(1: QueryRequest request);
}