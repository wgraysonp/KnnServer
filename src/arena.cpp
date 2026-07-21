#include "src/arena.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <sys/mman.h>
#include <stdexcept>

namespace recsys{
template<typename T>
MemoryArena<T>::MemoryArena(size_t total_bytes): 
  capacity_(total_bytes), offset_(0) {
  raw_ptr_ = mmap(nullptr, capacity_, 
    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw_ptr_ == MAP_FAILED) {
    throw std::runtime_error("mmap allocation failed");
  }
}

template<typename T>
MemoryArena<T>::~MemoryArena(){
  if (raw_ptr_ && raw_ptr_ != MAP_FAILED){
    munmap(raw_ptr_, capacity_);
  }
}

template<typename T>
void* MemoryArena<T>::AllocateBytes(size_t bytes, size_t alignment){
  uintptr_t current_address = reinterpret_cast<uintptr_t>(raw_ptr_) + offset_;
  size_t padding = 0;
  if (current_address % alignment != 0) {
    //TODO Update to use bit shifts instead of modular arithmetic
    padding = alignment - (current_address % alignment);
  }
  if (offset_ + padding + bytes > capacity_) {
    std::cerr << 
    "[WARNING] Not enough memory left. Returning nullptr" << std::endl;
    return nullptr;
  }
  offset_ += padding;
  void* allocated_ptr = reinterpret_cast<void*>(
    reinterpret_cast<uintptr_t>(raw_ptr_) + offset_);
  offset_ += bytes;
  return allocated_ptr;
}
template class MemoryArena<float>;
template class MemoryArena<double>;
template class MemoryArena<int>;
} // namespace recsys