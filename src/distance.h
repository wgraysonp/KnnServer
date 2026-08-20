#ifndef RECSYS_ENGINE_DISTANCE_H_
#define RECSYS_ENGINE_DISTANCE_H_

#include <cstddef>

namespace recsys {

template <typename T>
double ComputeSquaredEuclideanDistance(const T* a, const T* b, size_t dim);

template <typename T>
double NeonComputeSquaredEuclideanDistance(const T* a, const T* b, size_t dim);

}  // namespace recsys
#endif  // RECSYS_ENGINE_DISTANCE_H_