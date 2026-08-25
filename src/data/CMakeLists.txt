add_library(data_module)

set(THRIFT_DATA_INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/data.thrift")
set(THRIFT_DATA_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gen-cpp2")
set(GENERATED_THRIFT_DATA_SOURCES 
        "${THRIFT_DATA_OUTPUT_DIR}/data_data.cpp"
        "${THRIFT_DATA_OUTPUT_DIR}/data_constants.cpp"
        "${THRIFT_DATA_OUTPUT_DIR}/data_metadata.cpp"
        "${THRIFT_DATA_OUTPUT_DIR}/data_sinit.cpp"
        "${THRIFT_DATA_OUTPUT_DIR}/data_types_binary.cpp"
        "${THRIFT_DATA_OUTPUT_DIR}/data_types_compact.cpp"
        "${THRIFT_DATA_OUTPUT_DIR}/data_types_serialization.cpp"
        "${THRIFT_DATA_OUTPUT_DIR}/data_types.cpp"
)

add_custom_command(
    OUTPUT ${GENERATED_THRIFT_DATA_SOURCES}
    COMMAND ${THRIFT_COMPILER_BIN} 
             --gen "mstch_cpp2:include_prefix=src/data"
             -I "${CMAKE_SOURCE_DIR}"                    
             -out "${THRIFT_DATA_OUTPUT_DIR}"
             "${THRIFT_DATA_INPUT_FILE}"               
    DEPENDS "${THRIFT_DATA_INPUT_FILE}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    VERBATIM
)

target_include_directories(data_module 
    PUBLIC 
      ${CMAKE_SOURCE_DIR}
    PRIVATE
      ${THRIFT_DATA_OUTPUT_DIR}
)

target_sources(data_module
    PRIVATE
    data_utils.cc
    ${GENERATED_THRIFT_DATA_SOURCES}
)

target_link_libraries(data_module
    PUBLIC
        ${THRIFT_CPP2_LIB}
        Folly::folly
)