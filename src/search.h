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

struct CompareResult {
  bool operator()(const EmbeddingSearchResult& a,
                  const EmbeddingSearchResult& b) {
    return a.dist < b.dist;
  }
};

using KnnPriorityQueue =
    std::priority_queue<EmbeddingSearchResult,
                        folly::fbvector<EmbeddingSearchResult>, CompareResult>;

template <typename T>
void ComputeAllDistancesInBatch(
    EmbeddingSearchResult* batch_results,
    const folly::fbvector<unsigned long>& active_ids, const T* arena_base,
    const T* query_vector, size_t embedding_dim, size_t start, size_t end);

void UpdateNClosestInChunkWithNewDistances(EmbeddingSearchResult* batch_results,
                                           KnnPriorityQueue& mutable_queue,
                                           size_t n_closest);

template <typename T>
folly::fbvector<EmbeddingSearchResult> FindNClosestInChunk(
    const folly::fbvector<unsigned long>& active_ids, const T* arena_base,
    const T* query_vector, size_t n_closest, size_t embedding_dim,
    size_t chunk_start, size_t chunk_end) {
  KnnPriorityQueue closest_n_queue;
  EmbeddingSearchResult batch_results[BATCH_SIZE];
  size_t batch_end;
  size_t batch_start;

  for (batch_start = chunk_start; batch_start < chunk_end;
       batch_start += BATCH_SIZE) {
    batch_end = std::min(batch_start + BATCH_SIZE, active_ids.size());

    ComputeAllDistancesInBatch<T>(batch_results, active_ids, arena_base,
                                  query_vector, embedding_dim, batch_start,
                                  batch_end);
    UpdateNClosestInChunkWithNewDistances(batch_results, closest_n_queue,
                                          n_closest);
  }
  folly::fbvector<EmbeddingSearchResult> chunk_results;

  while (!closest_n_queue.empty()) {
    EmbeddingSearchResult res = closest_n_queue.top();
    chunk_results.push_back(std::move(res));
    closest_n_queue.pop();
  }
  return chunk_results;
}

template <typename T>
folly::fbvector<EmbeddingSearchResult> ComputeInitialCanidiateEmbeddings(
    const MemoryArena& arena, const folly::fbvector<T>& query_vector,
    size_t n_closest, size_t n_workers) {
  const folly::fbvector<unsigned long>& active_ids = arena.GetActiveIds();

  const T* query_data = query_vector.data();
  const T* arena_base = arena.GetArenaView<T>();

  size_t total_items = active_ids.size();
  size_t embedding_dim = arena.GetEmbeddingDim();

  ThreadPool bundle = ThreadPool(n_workers);

  size_t embeddings_per_worker = total_items / n_workers;
  size_t remainder = total_items % n_workers;
  size_t start = 0;
  folly::fbvector<std::future<folly::fbvector<EmbeddingSearchResult>>>
      search_results;

  for (size_t i = 0; i < n_workers; ++i) {
    size_t chunk_size = embeddings_per_worker + (i < remainder ? 1 : 0);
    size_t end = start + chunk_size;
    search_results.emplace_back(
        bundle.Add([&arena_base, &embedding_dim, &query_data, &n_closest,
                    &active_ids, start, end]() {
          return FindNClosestInChunk(active_ids, arena_base, query_data,
                                     n_closest, embedding_dim, start, end);
        }));
    start = end;
  }

  folly::fbvector<EmbeddingSearchResult> return_vec;

  for (auto& fut : search_results) {
    auto res = fut.get();
    return_vec.insert(return_vec.end(), std::make_move_iterator(res.begin()),
                      std::make_move_iterator(res.end()));
  }
  return return_vec;
}

template <typename T>
folly::fbvector<EmbeddingSearchResult> FindNClosest(
    const MemoryArena& arena, const folly::fbvector<T>& query_vector,
    size_t n_closest, size_t n_workers = 4) {
  folly::fbvector<EmbeddingSearchResult> candidates =
      ComputeInitialCanidiateEmbeddings(arena, query_vector, n_closest,
                                        n_workers);

  std::nth_element(
      candidates.begin(), candidates.begin() + n_closest, candidates.end(),
      [](const EmbeddingSearchResult& a, const EmbeddingSearchResult& b) {
        return a.dist < b.dist;
      });
  candidates.resize(n_closest);
  return candidates;
}
}  // namespace recsys
#endif  // RECSYS_ENGINE_SEARCH_H_