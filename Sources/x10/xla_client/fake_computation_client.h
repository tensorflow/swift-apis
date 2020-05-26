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

#ifndef X10_XLA_CLIENT_FAKE_COMPUTATION_CLIENT_H_
#define X10_XLA_CLIENT_FAKE_COMPUTATION_CLIENT_H_

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/client/client_library.h"

namespace xla {

class FakeComputationClient : public ComputationClient {
 public:
  class FakeData;
  struct FakeComputation;

  FakeComputationClient();
  ~FakeComputationClient();

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  DataPtr TransferToServer(xla::BorrowingLiteral literal,
                           const xla::Shape& dest_shape,
                           const std::string& device_string) override;

  std::vector<Literal> TransferFromServer(
      absl::Span<const DataPtr> handles) override;

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      absl::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteParallelOptions& options) override;

  std::vector<DataPtr> ExecuteChained(absl::Span<const ExecuteChainedOp> ops,
                                      const std::string& device) override;

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      absl::Span<const DataPtr> tuples) override;

  std::string GetResourceDomain(const std::string& device) const override;

  std::string GetDefaultDevice() const override;

  swift_xla::Device GetDefaultDeviceStruct() const override;

  size_t GetNumDevices() const override;

  std::map<std::string, Metric> GetMetrics() const override;

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  void SetReplicationDevices(std::vector<std::string> devices) override;

  const std::vector<std::string>& GetReplicationDevices() const override;

  void SetRngSeed(size_t seed) override;

 private:
  std::string default_device_ = "CPU:0";
  std::vector<std::string> device_names_;
};

}  // namespace xla

#endif  // X10_XLA_CLIENT_FAKE_COMPUTATION_CLIENT_H_
