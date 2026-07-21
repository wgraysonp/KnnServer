#ifndef RECSYS_ENGINE_SEARCH_H_
#define RECSYS_ENGINE_SEARCH_H_

#include <memory>
#include <queue>
#include <vector>

#include "src/arena.h"

namespace recsys {

float ComputeSquaredEuclideanDistance(const float* a, const float* b, size_t dim);

std::<vector> FindNClosest(const MemoryArena& arena, 
   const std::vector<float>& query_vector, size_t n_closest);

} // namespace recys
#endif //RECSYS_ENGINE_SEARCH_H_