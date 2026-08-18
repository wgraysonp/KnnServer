#include "src/search.h"

#include <memory>
#include <queue>
#include <vector>

#include "src/arena.h"

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

}  // namespace recsys