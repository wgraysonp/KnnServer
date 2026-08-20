#ifndef RECSYS_ENGINE_ARENA_H_
#define RECSYS_ENGINE_ARENA_H_

#include <folly/Expected.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <new>
#include <span>
#include <vector>

#include "src/data/structs.h"
#include "src/data/types.h"

namespace recsys {

class MemoryArena {
 public:
  ~MemoryArena();
  MemoryArena(const MemoryArena&) = delete;
  MemoryArena& operator=(const MemoryArena&) = delete;
  MemoryArena(MemoryArena&&) noexcept = default;
  MemoryArena& operator=(MemoryArena&&) noexcept = default;

  size_t GetCapacity() const { return capacity_; }
  size_t GetEmbeddingDim() const { return embedding_dim_; }
  const std::vector<unsigned long>& GetActiveIds() const { return active_ids_; }

  template <typename T>
  std::span<const T> GetArenaView() const {
    return std::span(static_cast<T*>(arena_base_ptr_),
                     capacity_ * embedding_dim_);
  }

  folly::Expected<folly::Unit, AllocError> SetEntry(
      unsigned long index, std::span<const float> embedding);
  folly::Expected<folly::Unit, AllocError> SetEntry(
      unsigned long index, std::span<const double> embedding);

  static folly::Expected<std::unique_ptr<MemoryArena>, StartupError> MakeArena(
      size_t capacity, size_t embedding_dim, EmbeddingDataType type);

 private:
  MemoryArena(size_t capacity, size_t embedding_dim, EmbeddingDataType type);
  static size_t GetTypeSize(EmbeddingDataType type);

  template <typename T>
  folly::Expected<folly::Unit, AllocError> ValidateAddedEmbedding(
      size_t index, std::span<const T> embedding) {
    if (sizeof(T) != type_size_) {
      return folly::makeUnexpected(AllocError::TypeDisagreesWithEmbeddingType);
    }

    if (embedding.size() != embedding_dim_) {
      std::cerr << "Embedding dimension in correct."
                << "Expected " << embedding_dim_ << " But got "
                << embedding.size() << std::endl;
      return folly::makeUnexpected(AllocError::IncorrectEmbeddingDim);
    }
    if (index >= capacity_) {
      std::cerr << "No memory allocated for item " << index << std::endl;
      return folly::makeUnexpected(AllocError::NotFound);
    }
    return folly::unit;
  }

  std::vector<uint8_t> is_id_active_;
  std::vector<unsigned long> active_ids_;
  EmbeddingDataType type_;
  void* arena_base_ptr_;
  size_t capacity_;
  size_t embedding_dim_;
  size_t type_size_;
};

}  // namespace recsys

#endif  // RECSYS_ENGINE_ARENA_H_