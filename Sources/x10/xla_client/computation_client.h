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

#ifndef X10_XLA_CLIENT_COMPUTATION_CLIENT_H_
#define X10_XLA_CLIENT_COMPUTATION_CLIENT_H_

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/xla_client/device.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/types.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class ComputationClient {
 public:
  class Computation;
  class CompileInstance;
  struct TensorSource;
  struct ExecuteChainedOp;
  struct ExecuteComputationOptions;
  class Data;
  using DataPtr = std::shared_ptr<Data>;
  using ComputationPtr = std::shared_ptr<Computation>;

  class TransferManager {
   public:
    virtual ~TransferManager() {}

    virtual std::vector<Literal> TransferFromServerImpl(
        absl::Span<const DataPtr> handles) = 0;
  };

  class Device {
   public:
    virtual ~Device() {}

    const std::string& name() const { return name_; }
    virtual int32_t mesh_id() const;
    const swift_xla::Device& device_id() const { return device_id_; }

    virtual TransferManager* GetTransferManager() const = 0;
    explicit Device(std::string name) : name_(name) {
      device_id_ = swift_xla::Device(name_);
    }

    virtual std::vector<ComputationPtr> Compile(
        const std::vector<std::string>& devices,
        std::vector<CompileInstance> instances) = 0;

    // Transfers local tensor values to the TPU servers and fetches the handles.
    virtual std::vector<DataPtr> TransferToServer(
        absl::Span<const TensorSource> tensors) = 0;

    // Copies a single tensor in the form of a xla::BorrowingLiteral async to
    // the TPU. The literal is copied to a temporary buffer and then copied
    // async as per the semantics of TransferLiteralToDeviceAsync. The next
    // computation that is scheduled will wait for this transfer to complete
    // before running.
    virtual DataPtr TransferToServer(xla::BorrowingLiteral literal,
                                     const xla::Shape& dest_shape);

    virtual std::vector<DataPtr> ExecuteChained(
        absl::Span<const ExecuteChainedOp> ops) = 0;

    virtual std::string ResourceDomain() const = 0;

    virtual DataPtr CreateDataPlaceholder(Shape shape) = 0;

    virtual std::vector<DataPtr> ExecuteComputation(
        const Computation& computation, absl::Span<const DataPtr> arguments,
        const ExecuteComputationOptions& options) = 0;

    virtual bool IsLocal() { return false; }

   private:
    std::string name_;
    swift_xla::Device device_id_;
  };
  class Data {
   public:
    struct Info {
      virtual ~Info() {}
    };

    using OpaqueHandle = int64_t;

    Data(Device* device, Shape shape)
        : device_(std::move(device)), shape_(std::move(shape)) {}

    virtual ~Data() {}

    Device* device() const { return device_; }

    const Shape& shape() const { return shape_; }

    Info* info() const { return info_.get(); }

    std::shared_ptr<Info> SetInfo(std::shared_ptr<Info> info) {
      std::swap(info, info_);
      return info;
    }

    virtual OpaqueHandle GetOpaqueHandle() = 0;

    virtual void Assign(const Data& data) = 0;

    virtual bool HasValue() const = 0;

   private:
    Device* device_;
    Shape shape_;
    std::shared_ptr<Info> info_;
  };

  class Computation {
   public:
    Computation(XlaComputation computation, ProgramShape program_shape,
                std::vector<std::string> devices)
        : computation_(std::move(computation)),
          program_shape_(std::move(program_shape)),
          devices_(std::move(devices)) {}

    virtual ~Computation() {}

    const XlaComputation& computation() const { return computation_; }

    const ProgramShape& program_shape() const { return program_shape_; }

    const std::vector<std::string>& devices() const { return devices_; }

   private:
    XlaComputation computation_;
    ProgramShape program_shape_;
    std::vector<std::string> devices_;
  };

  // The TensorSource provides a way for a client to populate a buffer allocated
  // by the core computation client code.
  struct TensorSource {
    // The PopulateFn accepts a dense buffer is standard array layout
    // (dim0-major) and deposits the source tensor data directly over the
    // provided buffer.
    using PopulateFn = std::function<void(const TensorSource&, void*, size_t)>;

    TensorSource() = default;
    TensorSource(Shape shape, PopulateFn populate_fn)
        : shape(std::move(shape)), populate_fn(std::move(populate_fn)) {}

    Shape shape;
    PopulateFn populate_fn;
  };

  struct CompileInstance {
    CompileInstance() = default;
    CompileInstance(XlaComputation computation, const Shape* output_shape)
        : computation(std::move(computation)), output_shape(output_shape) {}

    XlaComputation computation;
    const Shape* output_shape = nullptr;
  };

  struct ExecuteOptions {
    bool explode_tuple = true;
  };

  struct ExecuteComputationOptions : public ExecuteOptions {};

  struct ExecuteReplicatedOptions : public ExecuteOptions {};

  struct ExecuteParallelOptions : public ExecuteOptions {};

  // Describes an operation to be fed to the ExecuteChained() API.
  // If the device_data member is not nullptr, this operation is a device data
  // input. Otherwise computation must not be nullptr, and represents the
  // computation to be executed. The indices of the inputs to the computation,
  // are coming from the inputs member. Since the operations fed to
  // ExecuteChained() are a valid post-order, the op_index indices listed within
  // the inputs member must be lower of the index of the current
  // ExecuteChainedOp within the post-order. If the outputs member has values,
  // the result of this ExecuteChainedOp will become an output of the
  // ExecuteChained() API, with the output_index output of this ExecuteChainedOp
  // feeding the result_index result.
  struct ExecuteChainedOp {
    struct Input {
      size_t op_index;
      absl::optional<size_t> output_index;
    };
    struct Output {
      size_t result_index;
      absl::optional<size_t> output_index;
    };

    DataPtr device_data;
    ComputationPtr computation;
    std::vector<Output> outputs;
    std::vector<Input> inputs;
  };

  static std::unique_ptr<ComputationClient> Create();

  virtual ~ComputationClient() {}

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  static std::vector<Literal> TransferFromServer(
      absl::Span<const DataPtr> handles);

  virtual std::string GetDefaultDevice() const = 0;
  static Device* DefaultDevice();

  enum class DeviceKind { CPU, GPU, TPU };

  virtual swift_xla::Device GetDefaultDeviceStruct() const = 0;
  static swift_xla::Device DefaultDeviceStruct();

  std::vector<std::string> GetAllDevices() const;
  static std::vector<std::string> AllDevices();

  const std::vector<Device*>& GetAllDevicePointers() const { return devices_; }

  Device* GetDevice(const std::string& device_name) const;

  static void SetReplicationDevices(std::vector<std::string> devices);

  static const std::vector<std::string>& GetReplicationDevices();

  virtual void SetRngSeed(size_t seed) = 0;

  virtual std::map<std::string, Metric> GetMetrics() const = 0;
  static std::map<std::string, Metric> ReadMetrics();

  // Utility API around the vector based Compile() API to compile a single
  // computation.
  ComputationPtr Compile(XlaComputation computation,
                         std::string compilation_device,
                         std::vector<std::string> devices,
                         const Shape* output_shape);

  // Retrieves the set of devices to be passed to the computation client
  // Compile() API. If the devices array is empty, a vector with the single
  // device will be returned. Otherwise a vector with the devices content will
  // be returned.
  static std::vector<std::string> GetCompilationDevices(
      const std::string& device, absl::Span<const std::string> devices);

  // Retrieves the ordinal number out of a device string. This is the number
  // after the last ':' character of the device string.
  static int64_t GetDeviceOrdinal(const std::string& device);

  // Metrics common to all client intrfaces.
  static metrics::Metric* TransferToServerMetric();
  static metrics::Metric* TransferToServerTransformMetric();
  static metrics::Metric* TransferFromServerMetric();
  static metrics::Metric* CompileMetric();
  static metrics::Metric* ExecuteMetric();
  static metrics::Metric* ExecuteReplicatedMetric();
  static metrics::Metric* ExecuteParallelMetric();
  static metrics::Metric* ExecuteChainedMetric();
  static metrics::Metric* DeconstructTupleMetric();
  static metrics::Counter* CreateDataHandlesCounter();
  static metrics::Counter* ReleaseDataHandlesCounter();
  static metrics::Counter* DestroyDataHandlesCounter();
  static metrics::Metric* ReleaseDataHandlesTimeMetric();
  static metrics::Counter* CreateCompileHandlesCounter();
  static metrics::Counter* ReleaseCompileHandlesCounter();
  static metrics::Counter* DestroyCompileHandlesCounter();
  static metrics::Metric* ReleaseCompileHandlesTimeMetric();
  static metrics::Metric* InboundDataMetric();
  static metrics::Metric* OutboundDataMetric();

 protected:
  void AddDevice(std::unique_ptr<Device> device);

  // Returns the ComputationClient singleton.
  static ComputationClient* Get();

 private:
  std::vector<Device*> devices_;
  std::vector<std::unique_ptr<Device>> devices_owned_;
  absl::node_hash_map<std::string, Device*> devices_by_name_;

  friend ComputationClient::Device* GetX10Device(const std::string& device);
};

ComputationClient::Device* GetX10Device(const std::string& device);
ComputationClient::Device* GetX10Device(swift_xla::Device device_id);

}  // namespace xla

#endif  // X10_XLA_CLIENT_COMPUTATION_CLIENT_H_
