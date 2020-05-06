/*
 * Copyright 2020 TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef X10_XLA_CLIENT_DEVICE_H_
#define X10_XLA_CLIENT_DEVICE_H_

#include <iostream>
#include <string>

#include "tensorflow/compiler/xla/xla_client/util.h"

namespace swift_xla {

enum class DeviceType { CPU, GPU, TPU, REMOTE_TPU };

struct Device {
  Device() = default;
  explicit Device(const std::string& device_spec);
  Device(DeviceType hw_type, int ordinal)
      : hw_type(hw_type), ordinal(ordinal) {}

  bool operator==(const Device& other) const { return compare(other) == 0; }

  bool operator!=(const Device& other) const { return compare(other) != 0; }

  bool operator<(const Device& rhs) const { return compare(rhs) < 0; }

  int compare(const Device& rhs) const {
    if (hw_type != rhs.hw_type) {
      return hw_type < rhs.hw_type ? -1 : +1;
    }
    return ordinal < rhs.ordinal ? -1 : (ordinal > rhs.ordinal ? +1 : 0);
  }

  std::string ToString() const;

  friend std::ostream& operator<<(std::ostream& os, const Device& device) {
    os << device.ToString();
    return os;
  }

  size_t hash() const {
    return xla::util::StdHashCombine(xla::util::GetEnumValue(hw_type),
                                     ordinal + 1);
  }

  DeviceType hw_type = DeviceType::CPU;
  int ordinal = 0;
};

const Device* GetDefaultDevice();

Device GetCurrentDevice();

Device SetCurrentDevice(const Device& device);

static inline Device GetDeviceOrCurrent(const Device* device) {
  return device != nullptr ? *device : GetCurrentDevice();
}

}  // namespace swift_xla

#endif  // X10_XLA_CLIENT_DEVICE_H_
