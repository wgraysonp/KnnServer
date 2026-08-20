add_library(data_module)

target_include_directories(data_module PUBLIC ${CMAKE_SOURCE_DIR})

target_sources(data_module
    PRIVATE
    data_utils.cc
)

target_link_libraries(data_module
    PUBLIC
        Folly::folly
)