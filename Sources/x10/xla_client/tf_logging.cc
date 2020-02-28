#include "tensorflow/compiler/xla/xla_client/tf_logging.h"

namespace xla {
namespace internal {

void ErrorGenerator::operator&(const std::basic_ostream<char>& oss) const {
  const ErrorSink& sink = dynamic_cast<const ErrorSink&>(oss);
  auto sink_str = sink.str();
  TF_LOG(ERROR) << sink_str;
  std::stringstream ess;
  ess << file_ << ":" << line_ << " : " << sink_str;
  TF_LOG(FATAL) << ess.str();
}

}  // namespace internal
}  // namespace xla
