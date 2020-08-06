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

#if defined(_WIN32)
#define XLA_API __declspec(dllexport)
#else
#define XLA_API __attribute__((__visibility__("default")))
#endif

#include "device_wrapper.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/tensor.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"

DeviceType ConvertDeviceType(swift_xla::DeviceType device_type) {
  switch (device_type) {
    case swift_xla::DeviceType::CPU: {
      return CPU_DEVICE;
    }
    case swift_xla::DeviceType::GPU: {
      return GPU_DEVICE;
    }
    case swift_xla::DeviceType::TPU: {
      return TPU_DEVICE;
    }
    case swift_xla::DeviceType::REMOTE_TPU: {
      return REMOTE_TPU_DEVICE;
    }
    default: {
      LOG(FATAL) << "Invalid device: " << static_cast<int>(device_type);
    }
  }
}

swift_xla::DeviceType ConvertDeviceType(DeviceType device_type) {
  switch (device_type) {
    case CPU_DEVICE: {
      return swift_xla::DeviceType::CPU;
    }
    case GPU_DEVICE: {
      return swift_xla::DeviceType::GPU;
    }
    case TPU_DEVICE: {
      return swift_xla::DeviceType::TPU;
    }
    case REMOTE_TPU_DEVICE: {
      return swift_xla::DeviceType::REMOTE_TPU;
    }
    default: {
      LOG(FATAL) << "Invalid device: " << device_type;
    }
  }
}

swift_xla::Device ConvertDevice(const CDevice& device) {
  return {ConvertDeviceType(device.hw_type), device.ordinal};
}

CDevice ConvertDevice(const swift_xla::Device& device) {
  return {ConvertDeviceType(device.hw_type), device.ordinal};
}

namespace {

DeviceList* DeviceListFromStrings(
    tensorflow::gtl::ArraySlice<const std::string> device_strings) {
  size_t device_count = device_strings.size();
  auto devices = std::make_unique<CDevice[]>(device_count);
  for (size_t device_index = 0; device_index < device_count; ++device_index) {
    const std::string& device_string = device_strings[device_index];
    swift_xla::Device device(device_string);
    devices[device_index].hw_type = ConvertDeviceType(device.hw_type);
    devices[device_index].ordinal = device.ordinal;
  }
  return new DeviceList{devices.release(), device_count};
}

std::vector<std::string> DeviceListToStrings(DeviceList* device_list) {
  std::vector<xla::string> device_strings;
  for (size_t device_index = 0; device_index < device_list->count;
       ++device_index) {
    const CDevice& device = device_list->devices[device_index];
    swift_xla::Device xla_device(ConvertDeviceType(device.hw_type),
                                 device.ordinal);
    device_strings.push_back(xla_device.ToString());
  }
  return device_strings;
}

}  // namespace

void destroyDeviceList(DeviceList* device_list) { delete device_list; }

DeviceList* getAllDevices() {
  return DeviceListFromStrings(xla::ComputationClient::Get()->GetAllDevices());
}

CDevice getDefaultDevice() {
  return ConvertDevice(xla::ComputationClient::Get()->GetDefaultDeviceStruct());
}

void setReplicationDevices(struct DeviceList* device_list) {
  const auto device_strings = DeviceListToStrings(device_list);
  xla::ComputationClient::Get()->SetReplicationDevices(device_strings);
}

struct DeviceList* getReplicationDevices() {
  return DeviceListFromStrings(
      xla::ComputationClient::Get()->GetReplicationDevices());
}

void syncLiveTensorsForDevices(struct DeviceList* device_list) {
  const auto device_strings = DeviceListToStrings(device_list);
  xla::util::MultiWait mwait(device_strings.size());
  for (size_t i = 0; i < device_strings.size(); ++i) {
    auto executor = [&, i]() {
      const CDevice& cdevice = device_list->devices[i];
      swift_xla::Device device(ConvertDeviceType(cdevice.hw_type),
                               cdevice.ordinal);
      swift_xla::XLATensor::SyncLiveTensorsGraph(/*device=*/&device,
                                                 /*devices=*/device_strings,
                                                 /*wait=*/true);
    };
    xla::env::ScheduleIoClosure(mwait.Completer(std::move(executor)));
  }
  mwait.Wait();
}

void XLATensor_LazyTensorBarrier(const struct CDevice* device,
                                 struct DeviceList* device_list, bool wait) {
  const auto device_strings = DeviceListToStrings(device_list);
  swift_xla::Device tmp_device;
  if (device) tmp_device = ConvertDevice(*device);
  const swift_xla::Device* converted_device = device ? &tmp_device : nullptr;
  swift_xla::XLATensor::SyncLiveTensorsGraph(/*device=*/converted_device,
                                             /*devices=*/device_strings,
                                             /*wait=*/wait);
  swift_xla::XLATensor::MarkStep(converted_device);
}
