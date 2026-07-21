#ifndef RECSYS_ENGINE_ARENA_H_
#define RECSYS_ENGINE_ARENA_H_

#include <cstddef>
#include <new>
#include <vector>

namespace recsys {

constexpr size_t EMBEDDING_DIM = 128;

template <typename T>
struct Item {
    unsigned long id;
    const T* embedding;
};

template <typename T>
class MemoryArena {
  public:
    ~MemoryArena();
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&&) noexcept = default;
    MemoryArena& operator=(MemoryArena&&) noexcept = default;

    static MemoryArena<T> CreateForObjects(size_t n_objects){
      size_t total_bytes = n_objects*sizeof(T);
      return MemoryArena<T>(total_bytes);
    }

    bool Allocate(size_t n_objects, size_t alignment=16){
      size_t bytes = n_objects*sizeof(T);
      void* mem = AllocateBytes(bytes, alignment);
      if (!mem) {
        return false;
      }
      catalog_.push_back(Item<T>{item_count_++, static_cast<T*>(mem)});
      return true;
    }
    size_t bytes_used() const { return offset_; }
    size_t capacity() const { return capacity_; }
    size_t n_items() const { return item_count_; }
    const std::vector<Item<T>>& GetCatalog() const { return catalog_;}

  private:
    std::vector<Item<T>> catalog_;
    void* raw_ptr_;
    size_t capacity_;
    size_t offset_;
    size_t item_count_;
    
    MemoryArena(size_t total_bytes);
    void* AllocateBytes(size_t bytes, size_t alignment = 16);
};
} // namespace recsys

#endif // RECSYS_ENGINE_ARENA_H_