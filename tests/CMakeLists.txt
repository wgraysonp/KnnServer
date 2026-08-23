cmake_minimum_required(VERSION 3.15)
project(recsys_tests CXX)

# Get GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

add_executable(run_threadpool_tests threadpool_test.cpp)
target_include_directories(run_threadpool_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(run_threadpool_tests PRIVATE GTest::gtest_main)
target_link_libraries(run_threadpool_tests PRIVATE threads)
gtest_discover_tests(run_threadpool_tests DISCOVERY_MODE PRE_TEST)

add_executable(run_follypool_tests folly_threadpool_test.cpp)
target_include_directories(run_follypool_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(run_follypool_tests
            PRIVATE
              customfollythreads
              GTest::gtest_main
              Folly::folly
)
gtest_discover_tests(run_threadpool_tests DISCOVERY_MODE PRE_TEST)

add_executable(run_arena_tests arena_test.cpp)
target_include_directories(run_arena_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(run_arena_tests PRIVATE GTest::gtest_main)
target_link_libraries(run_arena_tests PRIVATE gmock_main)
target_link_libraries(run_arena_tests PRIVATE arena_memory)
gtest_discover_tests(run_arena_tests DISCOVERY_MODE PRE_TEST)

add_executable(run_search_tests search_test.cpp)
target_include_directories(run_search_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(run_search_tests PRIVATE GTest::gtest_main)
target_link_libraries(run_search_tests PRIVATE gmock_main)
target_link_libraries(run_search_tests PRIVATE search)
gtest_discover_tests(run_search_tests DISCOVERY_MODE PRE_TEST)