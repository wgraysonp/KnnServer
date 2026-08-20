#include "src/service/service_utils.h"

#include <folly/Expected.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/data/data_utils.h"
#include "src/data/structs.h"
#include "src/data/types.h"

namespace recsys {

folly::Expected<folly::Unit, RequestValidationError> ValidateRequestDataSize(
    const QueryRequest& request) {
  size_t expected_bytes = GetTypeSize(request.type) * request.embedding_dim;

  if (expected_bytes != request.query_vector.size()) {
    return folly::makeUnexpected(RequestValidationError::RawDataSizeError);
  }

  return folly::unit;
}

folly::Expected<folly::Unit, RequestValidationError>
ValidateRequestSearchLibrary(const QueryRequest& request) {
  // TODO: check that request.library exists. This will require storing
  // available libraries in some way and checking if the library exists. I will
  // problably need to make some sort of config file for this and parse it.

  return folly::unit;
}

}  // namespace recsys