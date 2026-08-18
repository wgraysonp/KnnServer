#ifndef RECSYS_ENGINE_MACROS_H_
#define RECSYS_ENGINE_MACROS_H_

#define ASSIGN_OR_RETURN(lhs, expr)                      \
  auto&& _res = (expr);                                  \
  if (!_res) return folly::makeUnexpected(_res.error()); \
  lhs = std::move(*_res)

#define RETURN_IF_ERROR(expr) \
  auto&& _res = (expr);       \
  if (!_res) return folly::makeUnexpected(_res.error());

#define CHECK_OK(expr)                                                    \
  do {                                                                    \
    auto&& _res = (expr);                                                 \
    if (!_res) {                                                          \
      std::cerr << "CRITICAL FAILURE: " << static_cast<int>(_res.error()) \
                << "\n";                                                  \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

#define CHECK_OK_AND_ASSIGN(lhs, expr)                \
  do {                                                \
    auto&& _res = (expr);                             \
    if (!_res) return static_cast<int>(_res.error()); \
    lhs = std::move(*_res);                           \
  } while (0)

#endif  // RECSYS_ENGINE_MACROS_H_