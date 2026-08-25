add_library(service_module)

set(THRIFT_SERVICE_INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/service.thrift")
set(THRIFT_SERVICE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gen-cpp2")
set(GENERATED_THRIFT_SERVICE_SOURCES 
        "${THRIFT_SERVICE_OUTPUT_DIR}/KnnSearchService.cpp"
        "${THRIFT_SERVICE_OUTPUT_DIR}/KnnSearchServiceAsyncClient.cpp"
        "${THRIFT_SERVICE_OUTPUT_DIR}/service_constants.cpp"
        "${THRIFT_SERVICE_OUTPUT_DIR}/service_data.cpp"
        "${THRIFT_SERVICE_OUTPUT_DIR}/service_metadata.cpp"
        "${THRIFT_SERVICE_OUTPUT_DIR}/service_sinit.cpp"
        "${THRIFT_SERVICE_OUTPUT_DIR}/service_types_binary.cpp"
        "${THRIFT_SERVICE_OUTPUT_DIR}/service_types_compact.cpp"
        "${THRIFT_SERVICE_OUTPUT_DIR}/service_types_serialization.cpp"
        "${THRIFT_SERVICE_OUTPUT_DIR}/service_types.cpp"
)

add_custom_command(
    OUTPUT ${GENERATED_THRIFT_SERVICE_SOURCES}
    COMMAND ${THRIFT_COMPILER_BIN} 
             --gen "mstch_cpp2:include_prefix=src/service"
             -I "${CMAKE_SOURCE_DIR}"
             -out "${THRIFT_SERVICE_OUTPUT_DIR}" 
             "${THRIFT_SERVICE_INPUT_FILE}"
    DEPENDS "${THRIFT_SERVICE_INPUT_FILE}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    VERBATIM
)

target_include_directories(service_module 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}
    PRIVATE
        ${THRIFT_SERVICE_OUTPUT_DIR}
)


target_sources(service_module
    PRIVATE
    pipelines.cc
    service_utils.cc
    ${GENERATED_THRIFT_SERVICE_SOURCES}
)

target_link_libraries(service_module
            PUBLIC
                data_module
            PRIVATE
                ${THRIFT_CPP2_LIB}
                Folly::folly
)

target_compile_options(service_module PRIVATE "-mcpu=apple-m2")
target_compile_options(service_module PRIVATE $<$<CONFIG:Debug>:-Wall -Wextra  -g>)
target_compile_options(service_module PRIVATE $<$<CONFIG:Release>:-O2>)