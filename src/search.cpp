#include <memory>
#include <queue>
#include <vector>

#include "src/arena.h"

namespace recsys {

template<typename T>
struct KnnResult {
  Item<T>* item;
  double dist;
};

template<typename T>
struct CompareResult {
  bool operator()(const KnnResult<T>& a, const KnnResult<T>& b){
    return a.dist < b.dist;
  }
};


template<typename T>
double ComputeSquaredEuclideanDistance(const T* a, const T*b, size_t dim){
  double sum = 0.0;
  for (size_t i = 0; i < dim; ++i){
    // cast to double to prevent possible overflow if T is int
    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

template <typename T>
std::vector<unsigned long> FindNClosest(const MemoryArena<T>& arena, 
   const std::vector<T>& query_vector, size_t n_closest){
  const auto& catalog = arena.GetCatalog();
  std::priority_queue<KnnResult, std::vector<KnnResult>, CompareResult> closest_n_queue;
  
  const T* query_data = query_vector.data();
  size_t total_items = catalog.size();
  size_t i = 0;
  for (; i < n_closest; ++i){
    double distance = ComputeSquaredEuclideanDistance<T>(query_data, catalog[i].embedding, EMBEDDING_DIM);
    closest_n_queue.push(KnnResult{catalog[i].id, distances});
  }
  for (; i < total_items; ++i){
    double distance = ComputeSquaredEuclideanDistance<T>(query_data, catalog[i].embedding, EMBEDDING_DIM);
    if (distances < closest_n_queue.top().dist){
      closest_n_queue.pop();
      closest_n_queue.push(KnnResult{catalog[i].id, distance});
    }
  }

  std::vector<unsigned long> query_result;
  while (!closest_n_queue.empty()){
    query_result.push_back(closest_n_queue.top().id);
    closest_n_queue.pop();
  }
  return query_result;
}
} // namespace recys