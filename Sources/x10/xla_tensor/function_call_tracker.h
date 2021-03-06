#pragma once

namespace swift_xla {
namespace fn_tracker {

#define XLA_FN_TRACK(level) \
  torch_xla::fn_tracker::TrackFunction(__FUNCTION__, level)

void TrackFunction(const char* tag, int level);

}  // namespace fn_tracker
}  // namespace swift_xla
