#ifndef RECSYS_ENGINE_SERVICE_SERVICE_UTILS_H_
#define RECSYS_ENGINE_SERVICE_SERVICE_UTILS_H_

#include <folly/Expected.h>

#include <cstddef>
#include <cstdint>

#include "src/data/structs.h"
#include "src/data/types.h"

namespace recsys {

// Checks that the size of the raw bytes of the query vector sent in the
// request match the request datatype and embedding dimension.
folly::Expected<folly::Unit, SearchRequestError> ValidateRequestDataSize(
    const QueryRequest& request);

// Verifies that the requested library to search within exists and is valid.
folly::Expected<folly::Unit, SearchRequestError>
ValidateRequestSearchLibrary(const QueryRequest& request);

// Temporary implementation assuming the request consists only of a single
// float vector.
// TODO: This should be updated to handle a full http or protobuf request
folly::Expected<folly::Unit, SearchRequestError> ValidateSearchRequest(
    const QueryRequest& request);

}  // namespace recsys

#endif  // RECSYS_ENGINE_SERVICE_SERVICE_UTILS_H_