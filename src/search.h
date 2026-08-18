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
std::vector<KnnResult<T>> ComputeInitialCanidiateEmbeddings(
    const MemoryArena& arena, const std::vector<T>& query_vector,
    size_t n_closest) {
  const auto& active_ids = arena.GetActiveIds();
  size_t total_items = active_ids.size();
  const T* arena_base = arena.GetArenaView<T>().data();
  size_t embedding_dim = arena.GetEmbeddingDim();
  const T* query_data = query_vector.data();

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
          std::priority_queue<KnnResult<T>, std::vector<KnnResult<T>>,
                              CompareResult<T>>
              closest_n_queue;
          size_t processed_embeddings = 0;
          for (size_t j = start; j < end; ++j) {
            unsigned long id = active_ids[j];
            const T* embedding = arena_base + id * embedding_dim;
            // TODO(grayson) - Fix this to load the query data into the
            // registers
            //  first before the loop executes.
            double distance = NeonComputeSquaredEuclideanDistance<T>(
                query_data, embedding, embedding_dim);
            if (processed_embeddings < n_closest) {
              closest_n_queue.push(KnnResult<T>{embedding, id, distance});
              processed_embeddings++;
              continue;
            }
            if (distance < closest_n_queue.top().dist) {
              closest_n_queue.pop();
              closest_n_queue.push(KnnResult<T>{embedding, id, distance});
            }
          }

          std::vector<KnnResult<T>> query_result;
          while (!closest_n_queue.empty()) {
            KnnResult<T> res = closest_n_queue.top();
            query_result.push_back(std::move(res));
            closest_n_queue.pop();
          }
          return query_result;
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