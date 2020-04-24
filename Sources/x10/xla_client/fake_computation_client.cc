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

#include "tensorflow/compiler/xla/xla_client/fake_computation_client.h"

#include <tuple>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace xla {

class FakeComputationClient::FakeData : public ComputationClient::Data {
 public:
  using Data::Data;

  virtual OpaqueHandle GetOpaqueHandle() {
    return reinterpret_cast<intptr_t>(this);
  }

  virtual void Assign(const Data& data) {}
  virtual bool HasValue() const { return true; }
};

struct FakeComputationClient::FakeComputation
    : public ComputationClient::Computation {
  using Computation::Computation;
};

using DataPtr = ComputationClient::DataPtr;
using ComputationPtr = ComputationClient::ComputationPtr;

DataPtr FakeComputationClient::CreateDataPlaceholder(std::string device,
                                                     Shape shape) {
  return std::make_shared<FakeData>(device, shape);
}

ComputationClient::DataPtr FakeComputationClient::TransferToServer(
    xla::BorrowingLiteral literal, const xla::Shape& dest_shape,
    const std::string& device) {
  return CreateDataPlaceholder(device, dest_shape);
}

std::vector<DataPtr> FakeComputationClient::TransferToServer(
    absl::Span<const TensorSource> tensors) {
  tensorflow::profiler::TraceMe trace("TransferToServer");
  std::vector<DataPtr> out;
  for (const auto& tensor : tensors) {
    out.push_back(CreateDataPlaceholder(tensor.device, tensor.shape));
  }
  return out;
}

std::vector<Literal> FakeComputationClient::TransferFromServer(
    absl::Span<const DataPtr> handles) {
  tensorflow::profiler::TraceMe trace("TransferFromServer");
  std::vector<Literal> out;
  for (const auto& handle : handles) {
    out.push_back(xla::Literal(handle->shape()));
  }
  return out;
}

std::vector<ComputationPtr> FakeComputationClient::Compile(
    std::vector<CompileInstance> instances) {
  metrics::TimedSection timed(CompileMetric());
  std::vector<ComputationPtr> out;
  for (auto& instance : instances) {
    auto program_shape =
        xla::ProgramShape(instance.computation.GetProgramShape().ValueOrDie());
    out.push_back(std::shared_ptr<FakeComputation>(
        new FakeComputation(std::move(instance.computation),
                            std::move(program_shape), instance.devices)));
  }
  return out;
}

std::vector<DataPtr> FakeComputationClient::ExecuteComputation(
    const Computation& computation, absl::Span<const DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  std::vector<DataPtr> out;
  for (const auto& shape :
       computation.program_shape().result().tuple_shapes()) {
    out.push_back(CreateDataPlaceholder(device, shape));
  }
  return out;
}

std::vector<std::vector<DataPtr>> FakeComputationClient::ExecuteReplicated(
    const Computation& computation,
    const std::vector<std::vector<DataPtr>>& arguments,
    absl::Span<const std::string> devices,
    const ExecuteReplicatedOptions& options) {
  TF_LOG(FATAL) << "Implement";
}

std::vector<std::vector<DataPtr>> FakeComputationClient::ExecuteParallel(
    absl::Span<const Computation* const> computations,
    const std::vector<std::vector<DataPtr>>& arguments,
    absl::Span<const std::string> devices,
    const ExecuteParallelOptions& options) {
  TF_LOG(FATAL) << "Implement";
}

std::vector<DataPtr> FakeComputationClient::ExecuteChained(
    absl::Span<const ExecuteChainedOp> ops, const std::string& device) {
  TF_LOG(FATAL) << "Implement";
}

std::vector<std::vector<DataPtr>> FakeComputationClient::DeconstructTuple(
    absl::Span<const DataPtr> tuples) {
  TF_LOG(FATAL) << "Implement";
}

std::map<std::string, Metric> FakeComputationClient::GetMetrics() const {
  TF_LOG(FATAL) << "Implement";
}

std::string FakeComputationClient::GetResourceDomain(
    const std::string& device) const {
  return "NopDomain";
}

std::string FakeComputationClient::GetDefaultDevice() const {
  return default_device_;
}

swift_xla::Device FakeComputationClient::GetDefaultDeviceStruct() const {
  return swift_xla::Device(swift_xla::DeviceType::CPU, 0);
}

size_t FakeComputationClient::GetNumDevices() const {
  return device_names_.size();
}

std::vector<std::string> FakeComputationClient::GetLocalDevices() const {
  TF_LOG(FATAL) << "Implement";
}

std::vector<std::string> FakeComputationClient::GetAllDevices() const {
  return device_names_;
  ;
}

void FakeComputationClient::SetReplicationDevices(
    std::vector<std::string> devices) {
  TF_LOG(FATAL) << "Implement";
}

const std::vector<std::string>& FakeComputationClient::GetReplicationDevices()
    const {
  static std::vector<std::string>* replicated_devices =
      new std::vector<std::string>;
  return *replicated_devices;
}

void FakeComputationClient::SetRngSeed(size_t seed) {
  TF_LOG(FATAL) << "Implement";
}

FakeComputationClient::FakeComputationClient() {
  for (int i = 0; i < 1; ++i) {
    device_names_.push_back(absl::StrCat("CPU:", i));
  }
  for (int i = 0; i < 16; ++i) {
    device_names_.push_back(absl::StrCat("TPU:", i));
  }
}

std::unique_ptr<ComputationClient> ComputationClient::Create() {
  return std::make_unique<FakeComputationClient>();
}

FakeComputationClient::~FakeComputationClient() {}

bool ComputationClient::IsLocal() { return true; }

}  // namespace xla
