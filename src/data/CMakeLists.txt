cmake_minimum_required(VERSION 3.15)
project(recsys_tests CXX)

add_library(data_module)

target_include_directories(data_module PUBLIC ${CMAKE_SOURCE_DIR})

target_sources(data_module
    PRIVATE
    data_utils.cc
)