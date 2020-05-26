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

#ifndef X10_DEVICE_WRAPPER_H_
#define X10_DEVICE_WRAPPER_H_

#include <stdbool.h>
#include <stddef.h>

#if !defined(XLA_API)
#define XLA_API
#endif

#ifdef __cplusplus
#include "tensorflow/compiler/xla/xla_client/device.h"

extern "C" {
#endif

enum DeviceType { CPU_DEVICE, GPU_DEVICE, TPU_DEVICE, REMOTE_TPU_DEVICE };

struct CDevice {
  enum DeviceType hw_type;
  int ordinal;
};

struct DeviceList {
  const struct CDevice* devices;
  size_t count;
#ifdef __cplusplus
  ~DeviceList() { delete[] devices; }
#endif
};

XLA_API void destroyDeviceList(struct DeviceList* device_list);

// Gets all available devices.
XLA_API struct DeviceList* getAllDevices();

// Gets the best default device.
XLA_API struct CDevice getDefaultDevice();

// Set device replication for cross-device gradient reduction.
XLA_API void setReplicationDevices(struct DeviceList* device_list);

// Get current device replication for cross-device gradient reduction.
XLA_API struct DeviceList* getReplicationDevices();

// Execute outstanding operations for all live tensors across the provided
// devices, in parallel.
XLA_API void syncLiveTensorsForDevices(struct DeviceList* device_list);

// Marks step and synchronizes a single device out of a list of devices.
// For use in a multi-threaded environment.
XLA_API void XLATensor_LazyTensorBarrier(const struct CDevice* device,
                                         struct DeviceList* device_list,
                                         bool wait);

#ifdef __cplusplus
}  // extern "C"

DeviceType ConvertDeviceType(swift_xla::DeviceType device_type);

swift_xla::DeviceType ConvertDeviceType(DeviceType device_type);

swift_xla::Device ConvertDevice(const CDevice& device);

CDevice ConvertDevice(const swift_xla::Device& device);

#endif

#endif  // X10_DEVICE_WRAPPER_H_
