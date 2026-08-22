cmake_minimum_required(VERSION 3.30)
project(RecSysEngine LANGUAGES CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_SiTANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Boost_NO_BOOST_CMAKE OFF)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

find_package(folly CONFIG REQUIRED)
find_package(Threads REQUIRED)

# if(APPLE)
#     include_directories(/opt/homebrew/include)
#     link_directories(/opt/homebrew/lib)
# endif()

add_subdirectory(src/data)
add_subdirectory(src/service)

add_library(threads src/threads/threadpool.cpp)
target_include_directories(threads PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(arena_memory src/arena.cpp)
target_link_libraries(arena_memory 
        PUBLIC 
            service_module
            data_module
        PRIVATE
            Folly::folly
)
target_include_directories(arena_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(distance src/distance.cpp)
target_include_directories(distance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(search src/search.cpp)
target_link_libraries(search 
        PUBLIC
            arena_memory
        PRIVATE
            service_module
            data_module
            threads
            Folly::folly
)
target_include_directories(search PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

foreach(target arena_memory distance threads search)
    target_compile_options(${target} PRIVATE "-mcpu=apple-m2")
    target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:-Wall -Wextra -Werror -g>)
    target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O2>)
endforeach()


add_executable(recsys_app apps/main.cpp)
target_link_libraries(recsys_app PRIVATE
          search
          arena_memory
          distance
          threads
          service_module
          data_module
          Folly::folly
          Folly::folly_deps
          Threads::Threads
 )
 target_include_directories(recsys_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

option(BUILD_TESTING "Build the test suite" ON)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
