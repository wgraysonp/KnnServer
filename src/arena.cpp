#include "src/arena.h"

#include <folly/Expected.h>
#include <sys/mman.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>

#include "src/consts.h"
#include "src/macros.h"

namespace recsys {
MemoryArena::MemoryArena(size_t capacity, size_t embedding_dim,
                         EmbeddingDataType type)
    : type_(type), capacity_(capacity), embedding_dim_(embedding_dim) {
  if (embedding_dim % 16 != 0) {
    throw std::invalid_argument("Embedding dimesion must be a multple of 16");
  }

  folly::Expected<size_t, InvalidArgmentError> embedding_entry_size =
      GetTypeSize(type_);
  if (!embedding_entry_size) {
    throw std::invalid_argument("Invalid embedding datatype");
  }

  type_size_ = *embedding_entry_size;
  size_t total_bytes = capacity_ * embedding_dim_ * type_size_;

  arena_base_ptr_ = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (arena_base_ptr_ == MAP_FAILED) {
    throw std::runtime_error("mmap allocation failed");
  }

  is_id_active_.assign(capacity_, 0);
}

MemoryArena::~MemoryArena() {
  size_t total_bytes = capacity_ * embedding_dim_ * type_size_;
  if (arena_base_ptr_ && arena_base_ptr_ != MAP_FAILED) {
    munmap(arena_base_ptr_, total_bytes);
  }
}

folly::Expected<folly::Unit, AllocError> MemoryArena::SetEntry(
    unsigned long index, std::span<const float> embedding) {
  return ValidateAddedEmbedding<float>(index, embedding)
      .then([this, index, embedding](folly::Unit) {
        std::copy(
            embedding.begin(), embedding.end(),
            static_cast<float*>(arena_base_ptr_) + index * embedding_dim_);
        if (is_id_active_[index] == 0) {
          active_ids_.push_back(index);
          is_id_active_[index] = 1;
        }
        return folly::unit;
      });
}

folly::Expected<folly::Unit, AllocError> MemoryArena::SetEntry(
    unsigned long index, std::span<const double> embedding) {
  return ValidateAddedEmbedding<double>(index, embedding)
      .then([this, index, embedding](folly::Unit) {
        std::copy(
            embedding.begin(), embedding.end(),
            static_cast<double*>(arena_base_ptr_) + index * embedding_dim_);
        if (is_id_active_[index] == 0) {
          active_ids_.push_back(index);
          is_id_active_[index] = 1;
        }
        return folly::unit;
      });
}

folly::Expected<std::unique_ptr<MemoryArena>, StartupError>
MemoryArena::MakeArena(size_t capacity, size_t embedding_dim,
                       EmbeddingDataType type) {
  try {
    return std::unique_ptr<MemoryArena>(
        new MemoryArena(capacity, embedding_dim, type));
  } catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return folly::makeUnexpected(StartupError::MMAPFailure);
  } catch (const std::invalid_argument& e){
    std::cerr << "Error: " << e.what() << std::endl;
    return folly::makeUnexpected(StartupError::ArenaInvalidArgument);
  } catch (...) {
    std::cerr << "Unknown startup error.." << std::endl;
    return folly::makeUnexpected(StartupError::Unknown);
  }
}

folly::Expected<size_t, InvalidArgmentError> MemoryArena::GetTypeSize(
    EmbeddingDataType type) {
  switch (type) {
    case EmbeddingDataType::Float32_t:
      return 4;
    case EmbeddingDataType::Float64_t:
      return 8;
    default:
      return folly::makeUnexpected(InvalidArgmentError::EnumTypeUndefined);
  }
}
}  // namespace recsys