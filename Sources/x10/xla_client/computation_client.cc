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

#include "tensorflow/compiler/xla/xla_client/computation_client.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/mesh_service.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace xla {

std::shared_ptr<ComputationClient::Computation> ComputationClient::Compile(
    XlaComputation computation, std::string compilation_device,
    std::vector<std::string> devices, const Shape* output_shape) {
  std::vector<CompileInstance> instances;
  instances.emplace_back(std::move(computation), output_shape);
  std::vector<std::shared_ptr<Computation>> results =
      GetX10Device(compilation_device)->Compile(devices, std::move(instances));
  return std::move(results[0]);
}

std::vector<std::string> ComputationClient::GetCompilationDevices(
    const std::string& device, absl::Span<const std::string> devices) {
  std::vector<std::string> compilation_devices;
  if (devices.empty()) {
    compilation_devices.push_back(device);
  } else {
    compilation_devices.insert(compilation_devices.end(), devices.begin(),
                               devices.end());
  }
  return compilation_devices;
}

int64 ComputationClient::GetDeviceOrdinal(const std::string& device) {
  auto pos = device.rfind(':');
  XLA_CHECK_NE(pos, std::string::npos) << device;
  return std::stoi(device.substr(pos + 1));
}

ComputationClient* ComputationClient::Get() {
  static ComputationClient* computation_client =
      ComputationClient::Create().release();
  return computation_client;
}

metrics::Metric* ComputationClient::TransferToServerMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("TransferToServerTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::TransferToServerTransformMetric() {
  static metrics::Metric* metric = new metrics::Metric(
      "TransferToServerTransformTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::TransferFromServerMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("TransferFromServerTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::CompileMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("CompileTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteReplicatedMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteReplicatedTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteParallelMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteParallelTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::ExecuteChainedMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ExecuteChainedTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::DeconstructTupleMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("DeconstructTupleTime", metrics::MetricFnTime);
  return metric;
}

metrics::Counter* ComputationClient::CreateDataHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter = new metrics::Counter("CreateDataHandles");
  return counter;
}

metrics::Counter* ComputationClient::ReleaseDataHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter = new metrics::Counter("ReleaseDataHandles");
  return counter;
}

metrics::Counter* ComputationClient::DestroyDataHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter = new metrics::Counter("DestroyDataHandles");
  return counter;
}

metrics::Metric* ComputationClient::ReleaseDataHandlesTimeMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ReleaseDataHandlesTime", metrics::MetricFnTime);
  return metric;
}

metrics::Counter* ComputationClient::CreateCompileHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter =
      new metrics::Counter("CreateCompileHandles");
  return counter;
}

metrics::Counter* ComputationClient::ReleaseCompileHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter =
      new metrics::Counter("ReleaseCompileHandles");
  return counter;
}

metrics::Counter* ComputationClient::DestroyCompileHandlesCounter() {
  // Do not change the name of the counter as xla_model.py references it.
  static metrics::Counter* counter =
      new metrics::Counter("DestroyCompileHandles");
  return counter;
}

metrics::Metric* ComputationClient::ReleaseCompileHandlesTimeMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("ReleaseCompileHandlesTime", metrics::MetricFnTime);
  return metric;
}

metrics::Metric* ComputationClient::InboundDataMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("InboundData", metrics::MetricFnBytes);
  return metric;
}

metrics::Metric* ComputationClient::OutboundDataMetric() {
  static metrics::Metric* metric =
      new metrics::Metric("OutboundData", metrics::MetricFnBytes);
  return metric;
}

int32_t ComputationClient::Device::mesh_id() const {
  TF_LOG(FATAL) << "Unsupported";
}

std::vector<std::string> ComputationClient::GetAllDevices() const {
  std::vector<std::string> out;
  auto tmp = GetAllDevicePointers();
  out.reserve(tmp.size());
  for (Device* device : tmp) {
    out.push_back(device->name());
  }
  return out;
}

void ComputationClient::AddDevice(std::unique_ptr<Device> device) {
  devices_.push_back(device.get());
  LOG(INFO) << "NAME: " << device->name();
  devices_by_name_[device->name()] = device.get();
  devices_owned_.push_back(std::move(device));
}

ComputationClient::Device* ComputationClient::GetDevice(
    const std::string& device_name) const {
  auto it = devices_by_name_.find(device_name);
  XLA_CHECK(it != devices_by_name_.end())
      << "Unable to find device: " << device_name;
  return it->second;
}

ComputationClient::Device* GetX10Device(const std::string& device) {
  return xla::ComputationClient::Get()->GetDevice(device);
}

ComputationClient::Device* GetX10Device(swift_xla::Device device_id) {
  return GetX10Device(device_id.ToString());
}

std::vector<Literal> ComputationClient::TransferFromServer(
    absl::Span<const DataPtr> handles) {
  if (handles.empty()) return {};
  TransferManager* transfer = handles[0]->device()->GetTransferManager();
  for (auto& handle : handles) {
    XLA_CHECK_EQ(transfer, handle->device()->GetTransferManager());
  }
  return transfer->TransferFromServerImpl(handles);
}

ComputationClient::DataPtr ComputationClient::Device::TransferToServer(
    xla::BorrowingLiteral literal, const xla::Shape& dest_shape) {
  TF_LOG(FATAL) << "Only supported for LocalClient";
}

std::map<std::string, Metric> ComputationClient::ReadMetrics() {
  return Get()->GetMetrics();
}

ComputationClient::Device* ComputationClient::DefaultDevice() {
  auto* client = Get();
  return client->GetDevice(client->GetDefaultDevice());
}

thread_local std::vector<std::string> g_replication_devices;  // NOLINT

void ComputationClient::SetReplicationDevices(
    std::vector<std::string> devices) {
  g_replication_devices = std::move(devices);
}

const std::vector<std::string>& ComputationClient::GetReplicationDevices() {
  return g_replication_devices;
}

swift_xla::Device ComputationClient::DefaultDeviceStruct() {
  return Get()->GetDefaultDeviceStruct();
}

std::vector<std::string> ComputationClient::AllDevices() {
  return Get()->GetAllDevices();
}

}  // namespace xla
