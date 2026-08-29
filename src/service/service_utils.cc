#include "src/service/service_utils.h"

#include <folly/Expected.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "src/data/consts.h"
#include "src/data/data_utils.h"
#include "src/data/structs.h"
#include "src/search.h"
#include "src/service/gen-cpp2/service_types.h"

namespace recsys::knn_server {
namespace {

template <typename T>
folly::fbvector<T> ConvertQueryVectorFromBytes(const QueryRequest& request) {
  folly::fbvector<T> query_vec(request.embedding_dim_ref().value());
  std::memcpy(query_vec.data(), request.raw_query_vector_ref()->data(),
              request.raw_query_vector_ref()->size());

  return query_vec;
}

template <typename T>
void PerformSearchAndPopulateResponseHelper(const MemoryArena& arena,
                                            const QueryRequest& request,
                                            QueryResponse& mutable_response) {
  folly::fbvector<T> query_vector = ConvertQueryVectorFromBytes<T>(request);
  auto start = std::chrono::high_resolution_clock::now();

  folly::fbvector<EmbeddingSearchResult> res =
      FindNClosest(arena, query_vector, request.n_closest_ref().value(),
                   std::thread::hardware_concurrency());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;

  mutable_response.status_ref() = ResponseStatus::StatusOk;
  mutable_response.milli_seconds_passed_ref() = duration.count();
  mutable_response.results_ref() =
      std::vector<EmbeddingSearchResult>(res.begin(), res.end());
}

}  // namespace

folly::Expected<folly::Unit, SearchRequestError> ValidateRequestDataSize(
    const QueryRequest& request) {
  size_t expected_bytes = GetTypeSize(request.type_ref().value()) *
                          request.embedding_dim_ref().value();

  if (expected_bytes != request.raw_query_vector_ref()->size()) {
    return folly::makeUnexpected(SearchRequestError::InvalidRawDataSizeError);
  }

  return folly::unit;
}

folly::Expected<folly::Unit, SearchRequestError> ValidateRequestSearchLibrary(
    const QueryRequest& request) {
  // TODO: check that request.library exists. This will require storing
  // available libraries in some way and checking if the library exists. I will
  // problably need to make some sort of config file for this and parse it.

  // temp placeholder to use request and keep Werror from complaning
  if (*request.library_id_ref() == 0) {
    return folly::makeUnexpected(SearchRequestError::InvalidSearchLibraryError);
  }

  return folly::unit;
}

folly::Expected<folly::Unit, SearchRequestError> ValidateSearchRequest(
    const QueryRequest& request) {
  return ValidateRequestDataSize(request).then(
      [&request](
          folly::Unit) -> folly::Expected<folly::Unit, SearchRequestError> {
        return ValidateRequestSearchLibrary(request);
      });
}

void PerformSearchAndPopulateResponse(const MemoryArena& arena,
                                      const QueryRequest& request,
                                      QueryResponse& mutable_response) {
  EmbeddingDataType arena_type = arena.GetArenaEmbeddingType();
  switch (arena_type) {
    case EmbeddingDataType::Float32_t:
      PerformSearchAndPopulateResponseHelper<float>(arena, request,
                                                    mutable_response);
      break;
    case EmbeddingDataType::Float64_t:
      PerformSearchAndPopulateResponseHelper<double>(arena, request,
                                                     mutable_response);
      break;
  }
}

}  // namespace recsys::knn_server