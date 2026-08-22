#include "src/search.h"

#include <arm_neon.h>
#include <folly/FBVector.h>

#include <algorithm>
#include <memory>
#include <queue>

#include "src/arena.h"

namespace recsys {

void ComputeAllDistancesInBatch(
    EmbeddingSearchResult* batch_results,
    const folly::fbvector<unsigned long>& active_ids, const float* arena_base,
    const float* query_vector, size_t embedding_dim, size_t start, size_t end) {
  int processed_count = 0;
  size_t remainder = (end - start) % 4;
  unsigned long i;

  // Handle remainder when the batch size is not divisible by 4
  for (i = start; i < start + remainder; ++i) {
    const float* search_vec = arena_base + active_ids[i] * embedding_dim;
    double dist = ComputeSquaredEuclideanDistance<float>(
        search_vec, query_vector, embedding_dim);
    batch_results[processed_count] = EmbeddingSearchResult{active_ids[i], dist};
    processed_count++;
  }

  // Compute distances for the majority of the batch using neon pipelines
  for (; i < end; i += NUM_NEON_PIPELINES) {
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

    batch_results[processed_count] = EmbeddingSearchResult{
        active_ids[i], static_cast<double>(vaddvq_f32(out0))};
    batch_results[processed_count + 1] = EmbeddingSearchResult{
        active_ids[i + 1], static_cast<double>(vaddvq_f32(out1))};
    batch_results[processed_count + 2] = EmbeddingSearchResult{
        active_ids[i + 2], static_cast<double>(vaddvq_f32(out2))};
    batch_results[processed_count + 3] = EmbeddingSearchResult{
        active_ids[i + 3], static_cast<double>(vaddvq_f32(out3))};

    processed_count += 4;
  }
}

void ComputeAllDistancesInBatch(
    EmbeddingSearchResult* batch_results,
    const folly::fbvector<unsigned long>& active_ids, const double* arena_base,
    const double* query_vector, size_t embedding_dim, size_t start,
    size_t end) {
  int processed_count = 0;
  size_t remainder = (end - start) % 4;
  unsigned long i;

  // Handle remainder when the batch size is not divisible by 4
  for (i = start; i < start + remainder; ++i) {
    const double* search_vec = arena_base + active_ids[i] * embedding_dim;
    double dist = ComputeSquaredEuclideanDistance<double>(
        search_vec, query_vector, embedding_dim);
    batch_results[processed_count] = EmbeddingSearchResult{active_ids[i], dist};
    processed_count++;
  }

  // Compute distances for the majority of the batch using neon pipelines
  // NUM_NEON_PIPELINES = 4 on Apple Silicon M2
  for (; i < end; i += NUM_NEON_PIPELINES) {
    float64x2_t out0 = vmovq_n_f64(0.0);
    float64x2_t out1 = vmovq_n_f64(0.0);
    float64x2_t out2 = vmovq_n_f64(0.0);
    float64x2_t out3 = vmovq_n_f64(0.0);

    const double* search_0 = arena_base + active_ids[i] * embedding_dim;
    const double* search_1 = arena_base + active_ids[i + 1] * embedding_dim;
    const double* search_2 = arena_base + active_ids[i + 2] * embedding_dim;
    const double* search_3 = arena_base + active_ids[i + 3] * embedding_dim;

    for (size_t idx = 0; idx < embedding_dim; idx += 2) {
      float64x2_t query = vld1q_f64(&query_vector[idx]);

      float64x2_t s0 = vld1q_f64(&search_0[idx]);
      float64x2_t s1 = vld1q_f64(&search_1[idx]);
      float64x2_t s2 = vld1q_f64(&search_2[idx]);
      float64x2_t s3 = vld1q_f64(&search_3[idx]);

      float64x2_t sub0 = vsubq_f64(query, s0);
      float64x2_t sub1 = vsubq_f64(query, s1);
      float64x2_t sub2 = vsubq_f64(query, s2);
      float64x2_t sub3 = vsubq_f64(query, s3);

      out0 = vfmaq_f64(out0, sub0, sub0);
      out1 = vfmaq_f64(out1, sub1, sub1);
      out2 = vfmaq_f64(out2, sub2, sub2);
      out3 = vfmaq_f64(out3, sub3, sub3);
    }

    batch_results[processed_count] = EmbeddingSearchResult{
        active_ids[i], static_cast<double>(vaddvq_f64(out0))};
    batch_results[processed_count + 1] = EmbeddingSearchResult{
        active_ids[i + 1], static_cast<double>(vaddvq_f64(out1))};
    batch_results[processed_count + 2] = EmbeddingSearchResult{
        active_ids[i + 2], static_cast<double>(vaddvq_f64(out2))};
    batch_results[processed_count + 3] = EmbeddingSearchResult{
        active_ids[i + 3], static_cast<double>(vaddvq_f64(out3))};

    processed_count += 4;
  }
}

void UpdateNClosestInChunkWithNewDistances(EmbeddingSearchResult* batch_results,
                                           KnnPriorityQueue& mutable_queue,
                                           size_t batch_length,
                                           size_t n_closest) {
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

}  // namespace recsys