#include "src/arena.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace recsys;

int main(){
    std::cout << 
      "Initializing memory arena via Cmake configuration..." << std::endl;

    size_t item_count = 10000000;
    auto arena = 
      MemoryArena<float>::CreateForObjects(item_count*EMBEDDING_DIM);
    std::vector<Item<float>> catalog;
    catalog.reserve(item_count);

    for (size_t i = 0; i < item_count; ++i){
      if (!arena.Allocate(EMBEDDING_DIM)) break;
    }
    return 0;
}