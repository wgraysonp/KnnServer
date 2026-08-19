#ifndef RECSYS_ENGINE_SEARCH_H_
#define RECSYS_ENGINE_SEARCH_H_

#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

#include "src/arena.h"
#include "src/distance.h"
#include "src/threads/threadpool.h"

namespace recsys {

template <typename T>
struct KnnResult {
  const T* embedding;
  unsigned long id;
  double dist;
};

template <typename T>
struct CompareResult {
  bool operator()(const KnnResult<T>& a, const KnnResult<T>& b) {
    return a.dist < b.dist;
  }
};

template <typename T>
using KnnPriorityQueue =
    std::priority_queue<KnnResult<T>, std::vector<KnnResult<T>>,
                        CompareResult<T>>;

template <typename T>
void ComputeAllDistancesInBatch(KnnResult<T>* batch_results,
                                const std::vector<unsigned long>& active_ids,
                                const T* arena_base, const T* query_vector,
                                size_t embedding_dim, size_t start, size_t end);

template <typename T>
void UpdateNClosestInChunkWithNewDistances(KnnResult<T>* batch_results,
                                           KnnPriorityQueue<T>& mutable_queue,
                                           size_t n_closest) {
  for (size_t res_idx = 0; res_idx < BATCH_SIZE; ++res_idx) {
    if (mutable_queue.size() < n_closest) {
      mutable_queue.push(batch_results[res_idx]);
      continue;
    }
    if (batch_results[res_idx].dist < mutable_queue.top().dist) {
      mutable_queue.pop();
      mutable_queue.push(batch_results[res_idx]);
    }
  }
}

template <typename T>
std::vector<KnnResult<T>> FindNClosestInChunk(
    const std::vector<unsigned long>& active_ids, const T* arena_base,
    const T* query_vector, size_t n_closest, size_t embedding_dim,
    size_t chunk_start, size_t chunk_end);

template <typename T>
std::vector<KnnResult<T>> ComputeInitialCanidiateEmbeddings(
    const MemoryArena& arena, const std::vector<T>& query_vector,
    size_t n_closest) {
  const std::vector<unsigned long>& active_ids = arena.GetActiveIds();

  const T* query_data = query_vector.data();
  const T* arena_base = arena.GetArenaView<T>().data();

  size_t total_items = active_ids.size();
  size_t embedding_dim = arena.GetEmbeddingDim();

  ThreadPool bundle = ThreadPool();
  size_t n_workers = bundle.GetPoolSize();

  size_t embeddings_per_worker = total_items / n_workers;
  size_t remainder = total_items % n_workers;
  size_t start = 0;
  std::vector<std::future<std::vector<KnnResult<T>>>> search_results;

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

  std::vector<KnnResult<T>> return_vec;

  for (auto& fut : search_results) {
    auto res = fut.get();
    return_vec.insert(return_vec.end(), std::make_move_iterator(res.begin()),
                      std::make_move_iterator(res.end()));
  }
  return return_vec;
}

template <typename T>
std::vector<KnnResult<T>> FindNClosest(const MemoryArena& arena,
                                       const std::vector<T>& query_vector,
                                       size_t n_closest) {
  std::vector<KnnResult<T>> candidates =
      ComputeInitialCanidiateEmbeddings(arena, query_vector, n_closest);

  std::nth_element(candidates.begin(), candidates.begin() + n_closest,
                   candidates.end(),
                   [](const KnnResult<T>& a, const KnnResult<T>& b) {
                     return a.dist < b.dist;
                   });
  candidates.resize(n_closest);
  return candidates;
}
}  // namespace recsys
#endif  // RECSYS_ENGINE_SEARCH_H_