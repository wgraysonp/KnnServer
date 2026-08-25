#include "src/distance.h"

#include <arm_neon.h>
#include <folly/FBVector.h>

#include <cstddef>

#include "src/data/consts.h"
#include "src/data/gen-cpp2/data_types.h"

namespace recsys::knn_server {
namespace {

double NeonComputeSquaredEuclideanDistance(const float* a, const float* b,
                                           size_t dim) {
  float32x4_t out0 = vmovq_n_f32(0.0);
  float32x4_t out1 = vmovq_n_f32(0.0);
  float32x4_t out2 = vmovq_n_f32(0.0);
  float32x4_t out3 = vmovq_n_f32(0.0);

  for (size_t i = 0; i < dim; i += 16) {
    float32x4_t vec_a0 = vld1q_f32(&a[i]);
    float32x4_t vec_a1 = vld1q_f32(&a[i + 4]);
    float32x4_t vec_a2 = vld1q_f32(&a[i + 8]);
    float32x4_t vec_a3 = vld1q_f32(&a[i + 12]);

    float32x4_t vec_b0 = vld1q_f32(&b[i]);
    float32x4_t vec_b1 = vld1q_f32(&b[i + 4]);
    float32x4_t vec_b2 = vld1q_f32(&b[i + 8]);
    float32x4_t vec_b3 = vld1q_f32(&b[i + 12]);

    float32x4_t sub0 = vsubq_f32(vec_a0, vec_b0);
    float32x4_t sub1 = vsubq_f32(vec_a1, vec_b1);
    float32x4_t sub2 = vsubq_f32(vec_a2, vec_b2);
    float32x4_t sub3 = vsubq_f32(vec_a3, vec_b3);

    out0 = vfmaq_f32(out0, sub0, sub0);
    out1 = vfmaq_f32(out1, sub1, sub1);
    out2 = vfmaq_f32(out2, sub2, sub2);
    out3 = vfmaq_f32(out3, sub3, sub3);
  }
  float32x4_t total_acc = vaddq_f32(out0, out1);
  total_acc = vaddq_f32(total_acc, out2);
  total_acc = vaddq_f32(total_acc, out3);

  // horizontal add all lanes
  float sum_sq_diff = vaddvq_f32(total_acc);

  return static_cast<double>(sum_sq_diff);
}

double NeonComputeSquaredEuclideanDistance(const double* a, const double* b,
                                           size_t dim) {
  float64x2_t out0 = vmovq_n_f64(0.0);
  float64x2_t out1 = vmovq_n_f64(0.0);
  float64x2_t out2 = vmovq_n_f64(0.0);
  float64x2_t out3 = vmovq_n_f64(0.0);

  for (size_t i = 0; i < dim; i += 8) {
    float64x2_t vec_a0 = vld1q_f64(&a[i]);
    float64x2_t vec_a1 = vld1q_f64(&a[i + 2]);
    float64x2_t vec_a2 = vld1q_f64(&a[i + 4]);
    float64x2_t vec_a3 = vld1q_f64(&a[i + 6]);

    float64x2_t vec_b0 = vld1q_f64(&b[i]);
    float64x2_t vec_b1 = vld1q_f64(&b[i + 2]);
    float64x2_t vec_b2 = vld1q_f64(&b[i + 4]);
    float64x2_t vec_b3 = vld1q_f64(&b[i + 6]);

    float64x2_t sub0 = vsubq_f64(vec_a0, vec_b0);
    float64x2_t sub1 = vsubq_f64(vec_a1, vec_b1);
    float64x2_t sub2 = vsubq_f64(vec_a2, vec_b2);
    float64x2_t sub3 = vsubq_f64(vec_a3, vec_b3);

    out0 = vfmaq_f64(out0, sub0, sub0);
    out1 = vfmaq_f64(out1, sub1, sub1);
    out2 = vfmaq_f64(out2, sub2, sub2);
    out3 = vfmaq_f64(out3, sub3, sub3);
  }
  float64x2_t total_acc = vaddq_f64(out0, out1);
  total_acc = vaddq_f64(total_acc, out2);
  total_acc = vaddq_f64(total_acc, out3);

  // horizontal add all lanes
  float sum_sq_diff = vaddvq_f64(total_acc);

  return static_cast<double>(sum_sq_diff);
}
}  // namespace

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
    double dist = NeonComputeSquaredEuclideanDistance(search_vec, query_vector,
                                                      embedding_dim);

    batch_results[processed_count].id_ref() = active_ids[i];
    batch_results[processed_count].distance_ref() = dist;

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

    batch_results[processed_count].id_ref() = active_ids[i];
    batch_results[processed_count].distance_ref() =
        static_cast<double>(vaddvq_f32(out0));

    batch_results[processed_count + 1].id_ref() = active_ids[i + 1];
    batch_results[processed_count + 1].distance_ref() =
        static_cast<double>(vaddvq_f32(out1));

    batch_results[processed_count + 2].id_ref() = active_ids[i + 2];
    batch_results[processed_count + 2].distance_ref() =
        static_cast<double>(vaddvq_f32(out2));

    batch_results[processed_count + 3].id_ref() = active_ids[i + 3];
    batch_results[processed_count + 3].distance_ref() =
        static_cast<double>(vaddvq_f32(out3));

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
    double dist = NeonComputeSquaredEuclideanDistance(search_vec, query_vector,
                                                      embedding_dim);

    batch_results[processed_count].id_ref() = active_ids[i];
    batch_results[processed_count].distance_ref() = dist;

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

    batch_results[processed_count].id_ref() = active_ids[i];
    batch_results[processed_count].distance_ref() =
        static_cast<double>(vaddvq_f64(out0));

    batch_results[processed_count + 1].id_ref() = active_ids[i + 1];
    batch_results[processed_count + 1].distance_ref() =
        static_cast<double>(vaddvq_f64(out1));

    batch_results[processed_count + 2].id_ref() = active_ids[i + 2];
    batch_results[processed_count + 2].distance_ref() =
        static_cast<double>(vaddvq_f64(out2));

    batch_results[processed_count + 3].id_ref() = active_ids[i + 3];
    batch_results[processed_count + 3].distance_ref() =
        static_cast<double>(vaddvq_f64(out3));

    processed_count += 4;
  }
}

}  // namespace recsys::knn_server