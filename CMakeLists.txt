cmake_minimum_required(VERSION 3.15)

project(RecSysEngine LANGUAGES CXX)

# require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=apple-m2")
set(CMAKE_CXX_FLAGES_DEBUG "-mcpu=apple-m2 -Wall -Wextra -g")

add_library(arena_memory
  src/arena.cpp
)

target_include_directories(arena_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(recsys_app apps/main.cpp)

target_include_directories(recsys_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(recsys_app PRIVATE arena_memory)