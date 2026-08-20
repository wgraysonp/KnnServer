cmake_minimum_required(VERSION 3.15)
project(recsys_tests CXX)

add_library(service_module)

target_include_directories(service_module PUBLIC ${CMAKE_SOURCE_DIR})

target_sources(service_module
    PRIVATE
    pipelines.cc
    service_utils.cc
)

target_compile_options(service_module PRIVATE "-mcpu=apple-m2")
target_compile_options(service_module PRIVATE $<$<CONFIG:Debug>:-Wall -Wextra -Werror -g>)
target_compile_options(service_module PRIVATE $<$<CONFIG:Release>:-O2>)