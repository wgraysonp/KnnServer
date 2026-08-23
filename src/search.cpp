#include "src/search.h"

#include <folly/FBVector.h>
#include <folly/futures/Future.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <queue>

#include "src/arena.h"
#include "src/data/structs.h"
#include "src/data/types.h"
#include "src/distance.h"
#include "src/threads/custom_folly_pool.h"
#include "src/threads/threadpool.h"

namespace recsys {
namespace {

struct CompareResult {
  bool operator()(const EmbeddingSearchResult& a,
                  const EmbeddingSearchResult& b) {
    if (a.dist != b.dist) return a.dist < b.dist;

    return a.id < b.id;
  }
};

using KnnPriorityQueue =
    std::priority_queue<EmbeddingSearchResult,
                        folly::fbvector<EmbeddingSearchResult>, CompareResult>;

void UpdateNClosestInChunkWithNewDistances(EmbeddingSearchResult* batch_results,
                                           KnnPriorityQueue& mutable_queue,
                                           const size_t batch_length,
                                           const size_t n_closest) {
  double worst_distance = std::numeric_limits<double>::min();
  size_t n_processed = 0;

  for (size_t res_idx = 0; res_idx < batch_length; ++res_idx) {
    if (n_processed < n_closest) {
      mutable_queue.push(batch_results[res_idx]);
      n_processed++;
      worst_distance = std::max(worst_distance, batch_results[res_idx].dist);
      continue;
    }
    if (batch_results[res_idx].dist < worst_distance) {
      mutable_queue.pop();
      mutable_queue.push(batch_results[res_idx]);
      worst_distance = mutable_queue.top().dist;
    }
  }
}

template <typename T>
folly::fbvector<EmbeddingSearchResult> FindNClosestInChunk(
    const folly::fbvector<unsigned long>& active_ids, const T* arena_base,
    const T* query_vector, const size_t n_closest, const size_t embedding_dim,
    const size_t chunk_start, const size_t chunk_end) {
  KnnPriorityQueue closest_n_queue;
  EmbeddingSearchResult batch_results[BATCH_SIZE];
  size_t batch_end;
  size_t batch_start;
  size_t batch_length;

  for (batch_start = chunk_start; batch_start < chunk_end;
       batch_start += BATCH_SIZE) {
    batch_end = std::min(batch_start + BATCH_SIZE, active_ids.size());
    batch_length = batch_end - batch_start;

    ComputeAllDistancesInBatch(batch_results, active_ids, arena_base,
                               query_vector, embedding_dim, batch_start,
                               batch_end);
    UpdateNClosestInChunkWithNewDistances(batch_results, closest_n_queue,
                                          batch_length, n_closest);
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
    const size_t n_closest, const size_t n_workers) {
  const folly::fbvector<unsigned long>& active_ids = arena.GetActiveIds();

  const T* query_data = query_vector.data();
  const T* arena_base = arena.GetArenaView<T>();

  const size_t total_items = active_ids.size();
  const size_t embedding_dim = arena.GetEmbeddingDim();

  FollyPool bundle(n_workers);

  const size_t embeddings_per_worker = total_items / n_workers;
  const size_t remainder = total_items % n_workers;
  size_t start = 0;
  folly::fbvector<folly::Future<folly::fbvector<EmbeddingSearchResult>>>
      search_results_futures;

  for (size_t i = 0; i < n_workers; ++i) {
    size_t chunk_size = embeddings_per_worker + (i < remainder ? 1 : 0);
    size_t end = start + chunk_size;
    search_results_futures.emplace_back(
        folly::via(&bundle, [&arena_base, &embedding_dim, &query_data,
                             &n_closest, &active_ids, start, end]() {
          return FindNClosestInChunk(active_ids, arena_base, query_data,
                                     n_closest, embedding_dim, start, end);
        }));
    start = end;
  }

  return folly::collect(search_results_futures).via(&bundle)
      .thenValue([](std::vector<folly::fbvector<EmbeddingSearchResult>>&&
                        search_results) {
        folly::fbvector<EmbeddingSearchResult> return_vec;
        for (auto& res : search_results) {
          return_vec.insert(return_vec.end(),
                            std::make_move_iterator(res.begin()),
                            std::make_move_iterator(res.end()));
        }
        return return_vec;
      }).get();
}

template <typename T>
folly::fbvector<EmbeddingSearchResult> FindNClosestImplementation(
    const MemoryArena& arena, const folly::fbvector<T>& query_vector,
    const size_t n_closest, const size_t n_workers) {
  size_t effective_n_closest =
      std::min(n_closest, arena.GetTotalActiveEmbeddings());

  folly::fbvector<EmbeddingSearchResult> candidates =
      ComputeInitialCanidiateEmbeddings(arena, query_vector,
                                        effective_n_closest, n_workers);

  std::nth_element(
      candidates.begin(), candidates.begin() + effective_n_closest,
      candidates.end(),
      [](const EmbeddingSearchResult& a, const EmbeddingSearchResult& b) {
        return a.dist < b.dist;
      });
  candidates.resize(effective_n_closest);
  std::sort(candidates.begin(), candidates.end(), CompareResult());
  return candidates;
}
}  // namespace

folly::fbvector<EmbeddingSearchResult> FindNClosest(
    const MemoryArena& arena, const folly::fbvector<float>& query_vector,
    const size_t n_closest, const size_t n_workers) {
  return FindNClosestImplementation<float>(arena, query_vector, n_closest,
                                           n_workers);
}

folly::fbvector<EmbeddingSearchResult> FindNClosest(
    const MemoryArena& arena, const folly::fbvector<double>& query_vector,
    const size_t n_closest, const size_t n_workers) {
  return FindNClosestImplementation<double>(arena, query_vector, n_closest,
                                            n_workers);
}

}  // namespace recsys