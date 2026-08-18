#include "src/distance.h"

#include <arm_neon.h>

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
double NeonComputeSquaredEuclideanDistance<float>(const float* a,
                                                  const float* b, size_t dim) {
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
}  // namespace recsys