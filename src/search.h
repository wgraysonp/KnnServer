#ifndef RECSYS_ENGINE_SEARCH_H_
#define RECSYS_ENGINE_SEARCH_H_

#include <folly/FBVector.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <queue>

#include "src/arena.h"
#include "src/data/structs.h"
#include "src/data/types.h"
#include "src/distance.h"
#include "src/threads/threadpool.h"

namespace recsys {

// The main KNN search functions. These find the n_closest embeddings
// to the query_vector among the active embeddings in the arena. The function
// is overloaded for the possible datatypes of the embeddings stored in the
// arena

folly::fbvector<EmbeddingSearchResult> FindNClosest(
    const MemoryArena& arena, const folly::fbvector<float>& query_vector,
    size_t n_closest, size_t n_workers = 4);

folly::fbvector<EmbeddingSearchResult> FindNClosest(
    const MemoryArena& arena, const folly::fbvector<double>& query_vector,
    size_t n_closest, size_t n_workers = 4);

}  // namespace recsys
#endif  // RECSYS_ENGINE_SEARCH_H_