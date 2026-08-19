#ifndef RECSYS_ENGINE_SERVICE_SERVICE_UTILS_H_
#define RECSYS_ENGINE_SERVICE_SERVICE_UTILS_H_

#include <folly/Expected.h>

#include <cstddef>
#include <cstdint>

#include "src/consts.h"
#include "src/data/structs.h"

enum class RequestValidationError {
  RawDataSizeError,
  SearchLibraryNotFoundError,
  UnupportedDatatypeError
};

namespace recsys {

folly::Expected<folly::Unit, RequestValidationError> ValidateRequestDataSize(
    const QueryRequest& request);

}  // namespace recsys

#endif  // RECSYS_ENGINE_SERVICE_SERVICE_UTILS_H_