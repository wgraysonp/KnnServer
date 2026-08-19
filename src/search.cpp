#include "src/search.h"

#include <arm_neon.h>

#include <memory>
#include <queue>
#include <vector>

#include "src/arena.h"
#include "src/consts.h"

namespace recsys {

template <>
double ComputeSquaredEuclideanDistance<float>(const float* a, const float* b,
                                              size_t dim) {
  double sum = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    // cast to double to prevent possible overflow if T is int
    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

template <>
void ComputeAllDistancesInBatch<float>(
    KnnResult<float>* batch_results,
    const std::vector<unsigned long>& active_ids, const float* arena_base,
    const float* query_vector, size_t embedding_dim, size_t start, size_t end) {
  int processed_count = 0;

  for (unsigned long i = start; i < std::min(end, active_ids.size());
       i += NUM_NEON_PIPELINES) {
    float32x4_t out0 = vmovq_n_f32(0.0);
    float32x4_t out1 = vmovq_n_f32(0.0);
    float32x4_t out2 = vmovq_n_f32(0.0);
    float32x4_t out3 = vmovq_n_f32(0.0);

    const float* search_0 = arena_base + active_ids[i] * embedding_dim;
    const float* search_1 = arena_base + active_ids[i + 1] * embedding_dim;
    const float* search_2 = arena_base + active_ids[i + 2] * embedding_dim;
    const float* search_3 = arena_base + active_ids[i + 3] * embedding_dim;

    for (size_t idx = 0; idx < embedding_dim; idx += 4) {
      float32x4_t query = vld1q_f32(&query_vector[idx]);

      float32x4_t s0 = vld1q_f32(&search_0[idx]);
      float32x4_t s1 = vld1q_f32(&search_1[idx]);
      float32x4_t s2 = vld1q_f32(&search_2[idx]);
      float32x4_t s3 = vld1q_f32(&search_3[idx]);

      float32x4_t sub0 = vsubq_f32(query, s0);
      float32x4_t sub1 = vsubq_f32(query, s1);
      float32x4_t sub2 = vsubq_f32(query, s2);
      float32x4_t sub3 = vsubq_f32(query, s3);

      out0 = vfmaq_f32(out0, sub0, sub0);
      out1 = vfmaq_f32(out1, sub1, sub1);
      out2 = vfmaq_f32(out2, sub2, sub2);
      out3 = vfmaq_f32(out3, sub3, sub3);
    }

    batch_results[processed_count] = KnnResult<float>{
        search_0, active_ids[i], static_cast<double>(vaddvq_f32(out0))};
    batch_results[processed_count + 1] = KnnResult<float>{
        search_1, active_ids[i + 1], static_cast<double>(vaddvq_f32(out1))};
    batch_results[processed_count + 2] = KnnResult<float>{
        search_2, active_ids[i + 2], static_cast<double>(vaddvq_f32(out2))};
    batch_results[processed_count + 3] = KnnResult<float>{
        search_3, active_ids[i + 3], static_cast<double>(vaddvq_f32(out3))};

    processed_count += 4;
  }
}

template <>
std::vector<KnnResult<float>> FindNClosestInChunk<float>(
    const std::vector<unsigned long>& active_ids, const float* arena_base,
    const float* query_vector, size_t n_closest, size_t embedding_dim,
    size_t chunk_start, size_t chunk_end) {
  std::priority_queue<KnnResult<float>, std::vector<KnnResult<float>>,
                      CompareResult<float>>
      closest_n_queue;

  KnnResult<float> batch_results[BATCH_SIZE];

  for (size_t batch_start = chunk_start; batch_start < chunk_end;
       batch_start += BATCH_SIZE) {
    size_t batch_end = batch_start + BATCH_SIZE;
    ComputeAllDistancesInBatch<float>(batch_results, active_ids, arena_base,
                                      query_vector, embedding_dim, batch_start,
                                      batch_end);

    for (size_t res_idx = 0; res_idx < BATCH_SIZE; ++res_idx) {
      if (closest_n_queue.size() < n_closest) {
        closest_n_queue.push(batch_results[res_idx]);
        continue;
      }
      if (batch_results[res_idx].dist < closest_n_queue.top().dist) {
        closest_n_queue.pop();
        closest_n_queue.push(batch_results[res_idx]);
      }
    }
    batch_start = batch_end;
  }
  std::vector<KnnResult<float>> chunk_results;

  while (!closest_n_queue.empty()) {
    KnnResult<float> res = closest_n_queue.top();
    chunk_results.push_back(std::move(res));
    closest_n_queue.pop();
  }
  return chunk_results;
}

}  // namespace recsys