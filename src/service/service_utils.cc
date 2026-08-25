#include "src/service/service_utils.h"

#include <folly/Expected.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/data/consts.h"
#include "src/data/data_utils.h"
#include "src/data/structs.h"
#include "src/service/gen-cpp2/service_types.h"

namespace recsys::knn_server {

folly::Expected<folly::Unit, SearchRequestError> ValidateRequestDataSize(
    const QueryRequest& request) {
  size_t expected_bytes = GetTypeSize(request.type_ref().value()) *
                          request.embedding_dim_ref().value();

  if (expected_bytes != request.raw_query_vector_ref()->size()) {
    return folly::makeUnexpected(SearchRequestError::InvalidRawDataSizeError);
  }

  return folly::unit;
}

folly::Expected<folly::Unit, SearchRequestError> ValidateRequestSearchLibrary(
    const QueryRequest& request) {
  // TODO: check that request.library exists. This will require storing
  // available libraries in some way and checking if the library exists. I will
  // problably need to make some sort of config file for this and parse it.

  // temp placeholder to use request and keep Werror from complaning
  if (*request.library_id_ref() == 0) {
    return folly::makeUnexpected(SearchRequestError::InvalidSearchLibraryError);
  }

  return folly::unit;
}

folly::Expected<folly::Unit, SearchRequestError> ValidateSearchRequest(
    const QueryRequest& request) {
  return ValidateRequestDataSize(request).then(
      [&request](
          folly::Unit) -> folly::Expected<folly::Unit, SearchRequestError> {
        return ValidateRequestSearchLibrary(request);
      });
}

}  // namespace recsys::knn_server