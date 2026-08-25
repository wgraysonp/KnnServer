#ifndef RECSYS_ENGINE_DISTANCE_H_
#define RECSYS_ENGINE_DISTANCE_H_

#include <folly/FBVector.h>

#include <cstddef>

#include "src/data/gen-cpp2/data_types.h"

namespace recsys::knn_server {

// Computes the euclidean distance to the the query vector for all active
// ids with indicices in [start, end) using neon intrinsics. Results are
// stored in batch_results

void ComputeAllDistancesInBatch(
    EmbeddingSearchResult* batch_results,
    const folly::fbvector<unsigned long>& active_ids, const float* arena_base,
    const float* query_vector, size_t embedding_dim, size_t start, size_t end);

void ComputeAllDistancesInBatch(
    EmbeddingSearchResult* batch_results,
    const folly::fbvector<unsigned long>& active_ids, const double* arena_base,
    const double* query_vector, size_t embedding_dim, size_t start, size_t end);

}  // namespace recsys::knn_server
#endif  // RECSYS_ENGINE_DISTANCE_H_