// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/xla/xla_client/device.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace swift_xla {
namespace {

thread_local absl::optional<Device> g_current_device;

std::string DeviceTypeToString(DeviceType hw_type) {
  switch (hw_type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::GPU:
      return "GPU";
    case DeviceType::TPU:
      return "TPU";
    case DeviceType::REMOTE_TPU:
      return "REMOTE_TPU";
  }
  XLA_ERROR() << "Invalid device type";
}

void ParseDevice(const std::string& device_spec, Device* device) {
  if (device_spec.empty()) {
    std::string default_device_spec =
        xla::ComputationClient::Get()->GetDefaultDevice();
    XLA_CHECK(!default_device_spec.empty());
    return ParseDevice(default_device_spec, device);
  }
  if (device_spec[0] == ':') {
    std::string default_device_spec =
        xla::ComputationClient::Get()->GetDefaultDevice();
    auto pos = default_device_spec.find(':');
    XLA_CHECK_NE(pos, std::string::npos) << default_device_spec;
    return ParseDevice(default_device_spec.substr(0, pos) + device_spec,
                       device);
  }
  std::vector<std::string> device_spec_parts = absl::StrSplit(device_spec, ':');
  XLA_CHECK_EQ(device_spec_parts.size(), 2)
      << "Invalid device specification: " << device_spec;

  device->ordinal = std::stoi(device_spec_parts[1]);
  if (device_spec_parts[0] == "TPU") {
    device->hw_type = DeviceType::TPU;
  } else if (device_spec_parts[0] == "CPU") {
    device->hw_type = DeviceType::CPU;
  } else if (device_spec_parts[0] == "GPU") {
    device->hw_type = DeviceType::GPU;
  } else if (device_spec_parts[0] == "REMOTE_TPU") {
    device->hw_type = DeviceType::REMOTE_TPU;
  } else {
    XLA_ERROR() << "Invalid device specification: " << device_spec;
  }
}

}  // namespace

Device::Device(const std::string& device_spec) {
  ParseDevice(device_spec, this);
}

std::string Device::ToString() const {
  return absl::StrCat(DeviceTypeToString(hw_type), ":", ordinal);
}

const Device* GetDefaultDevice() {
  static const Device* default_device = new Device("");
  return default_device;
}

Device GetCurrentDevice() {
  if (!g_current_device) {
    g_current_device = *GetDefaultDevice();
  }
  return *g_current_device;
}

Device SetCurrentDevice(const Device& device) {
  Device current = GetCurrentDevice();
  g_current_device = device;
  TF_VLOG(2) << "New current device: " << device;
  return current;
}

}  // namespace swift_xla
