cmake_minimum_required(VERSION 3.30)
project(RecSysEngine LANGUAGES CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Boost_NO_BOOST_CMAKE OFF)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

find_package(folly REQUIRED)
find_package(Threads REQUIRED)

if(APPLE)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
endif()

#add_library(folly /opt/homebrew/include/folly)
add_library(threads src/threads/threadpool.cpp)
add_library(arena_memory src/arena.cpp)
add_library(distance src/distance.cpp)

foreach(target arena_memory distance threads)
    target_compile_options(${target} PRIVATE "-mcpu=apple-m2")
    target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:-Wall -Wextra -g>)
    target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O2>)
endforeach()

target_include_directories(arena_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(distance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(threads PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(recsys_app apps/main.cpp)
target_include_directories(recsys_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(recsys_app PRIVATE
          arena_memory
          distance
          threads
          Folly::folly
          Threads::Threads
 )

option(BUILD_TESTING "Build the test suite" ON)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
