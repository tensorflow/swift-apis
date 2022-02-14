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

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/compiler/tf2xla/xla_tensor/computation.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/cross_replica_reduces.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ir_util.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/lowering_context.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/async_task.h"
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/device.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace swift_xla {

class XLATensor {
  class DeviceContextArena;
  struct Data;

 public:
  static XLATensor Create(const at::Tensor& tensor, const Device& device);
  static XLATensor Create(
      xla::ComputationClient::DataPtr xla_data,
      c10::optional<at::ScalarType> logical_element_type = absl::nullopt);
  static XLATensor Create(at::Scalar scalar, at::ScalarType type,
                          const Device& device);
  static XLATensor Create(
      ir::Value ir_value, const Device& device,
      c10::optional<at::ScalarType> logical_element_type = absl::nullopt);

  // Creates an empty/null tensor.
  XLATensor() = default;

  bool is_null() const { return data_ptr() == nullptr; }

  size_t generation() const { return data()->generation; }

  XLATensor alias() const { return XLATensor(data_ptr()); }

  int64_t size(int64_t dim) const;

  at::Tensor ToTensor(bool detached);

  void ShallowCopyTo(XLATensor* dest) const;

  at::ScalarType dtype() const;

  at::ScalarType physical_scalar_type() const;

  // Set logical_element_type which is visible to user code.
  void SetScalarType(c10::optional<at::ScalarType> logical_element_type);

  xla::util::MaybeRef<xla::Shape> shape() const;
  xla::Shape shape_with_layout() const;

  const Device& GetDevice() const;
  int64_t GetUniqueId() const;

  // Fetches the XLA data behind the tensor. If the tensor has a graph defining
  // its current value, executes the graph and fetches the XLA data result.
  xla::ComputationClient::DataPtr GetXlaData();

  // Fetches the current value of the XLA data, which can be missing (nullptr)
  // in case the tensor has a graph defining its current value,
  xla::ComputationClient::DataPtr CurrentXlaData() const;

  void SetXlaData(xla::ComputationClient::DataPtr xla_data);

  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  ir::Value CurrentIrValue() const;

  // Retrieves the IR Node representing this XLATensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state ofthe object.
  ir::Value GetIrValue() const;

  c10::optional<at::Tensor> CurrentTensorData() const;

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();

  static ir::Value GetIrValueForScalar(at::Scalar value,
                                       xla::PrimitiveType type,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(at::Scalar value, const Device& device);
  static ir::Value GetIrValueForScalar(at::Scalar value,
                                       xla::PrimitiveType type,
                                       absl::Span<const int64_t> dimensions,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(at::Scalar value,
                                       const xla::Shape& shape,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(
      at::Scalar value, const xla::Shape& shape,
      c10::optional<at::ScalarType> logical_element_type, const Device& device);

  static ir::Value GetRngSeed(const Device& device);

  static void SetRngSeed(const Device* device, xla::uint64 seed);

  static xla::uint64 GetRunningSeed(const Device& device);

  // Dispatches a comparison operator, setting the logical type of the result
  // appropriately.
  static XLATensor DispatchComparisonOp(c10::Symbol kind,
                                        const XLATensor& input,
                                        at::Scalar other);

  // Same as above, with the second input a tensor as well.
  static XLATensor DispatchComparisonOp(c10::Symbol kind,
                                        const XLATensor& input,
                                        const XLATensor& other);

  // Dumps the XLA HLO text of the computation accumulated in the graph which is
  // attached the tensors.
  static std::string DumpHloComputation(const std::vector<XLATensor>& tensors);

  // Retrieves the set of XLA tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned. Returned tensors are sorted by device as primary
  // key, and by unique ID as secondary key.
  static std::vector<XLATensor> GetLiveTensors(const Device* device);

  // Applies all the pending IR operations queued over the input tensors. All
  // the tensors must be on the same device. If wait is true, the sync operation
  // will be run synchronously. The devices argument, if not empty, tells the
  // devices which should be partecipating into the replicated computation.
  static void SyncTensorsGraph(std::vector<XLATensor>* tensors,
                               absl::Span<const std::string> devices, bool wait,
                               bool sync_xla_data);

  // Makes sure that any outstanding IR operation accumulated over live tensors,
  // gets turned into device data. If wait is true, the sync operation will be
  // run synchronously. The devices argument, if not empty, tells the devices
  // which should be partecipating into the replicated computation.
  static void SyncLiveTensorsGraph(const Device* device,
                                   absl::Span<const std::string> devices,
                                   bool wait);

  // Marks an execution step, which allows the tensor framework to understand
  // the computation boundaries.
  static void MarkStep(const Device* device);

  // Waits for all the outstanding operations on all the supplied devices.
  // If devices is empty, the wait will happen for all local devices.
  static void WaitDeviceOps(absl::Span<const std::string> devices);

  // Retrieves the CPU tensors behind the XLA tensors IR operations. All the
  // tensors must be on the same device.
  static std::vector<at::Tensor> GetTensors(std::vector<XLATensor>* tensors);

  // Operation which creates XLA tensors out of CPU tensors by batching the
  // requests to the computation servers.
  static std::vector<XLATensor> CreateTensors(
      const std::vector<at::Tensor>& tensors,
      const std::vector<std::string>& devices);

  //////////////////////////////////////////////////////////////////////////////
  // XLA dedicated operators follows here, listed in alphabetical order.
  //////////////////////////////////////////////////////////////////////////////
  static std::pair<XLATensor, ir::Value> all_reduce(
      const XLATensor& input, const ir::Value& token, AllReduceType reduce_type,
      double scale, std::vector<std::vector<int64_t>> groups);

  static std::pair<std::vector<XLATensor>, ir::Value> all_reduce(
      const std::vector<XLATensor>& inputs, const ir::Value& token,
      AllReduceType reduce_type, double scale,
      std::vector<std::vector<int64_t>> groups);

  static ir::Value all_reduce_(XLATensor& input, const ir::Value& token,
                               AllReduceType reduce_type, double scale,
                               std::vector<std::vector<int64_t>> groups);

  static ir::Value all_reduce_(std::vector<XLATensor>* inputs,
                               const ir::Value& token,
                               AllReduceType reduce_type, double scale,
                               std::vector<std::vector<int64_t>> groups);

  static std::pair<XLATensor, ir::Value> all_to_all(
      const XLATensor& input, const ir::Value& token,
      int64_t split_dimension, int64_t concat_dimension,
      int64_t split_count, std::vector<std::vector<int64_t>> groups);

  static std::pair<XLATensor, ir::Value> collective_permute(
      const XLATensor& input, const ir::Value& token,
      std::vector<std::pair<int64_t, int64_t>> source_target_pairs);

  static XLATensor get_dimensions_size(const XLATensor& input,
                                       std::vector<int64_t> dimensions);

  static std::vector<XLATensor> user_computation(
      const std::string& opname, absl::Span<const XLATensor> inputs,
      ComputationPtr computation);

  //////////////////////////////////////////////////////////////////////////////
  // ATEN operators follows here, listed in alphabetical order.
  //////////////////////////////////////////////////////////////////////////////
  static XLATensor annotate(const XLATensor& input, std::string annotation);

  static void arange_out(XLATensor& out, at::Scalar start, at::Scalar end,
                         at::Scalar step, at::ScalarType scalar_type);

  // Broadcasts the given tensors according to broadcasting semantics.
  static std::vector<XLATensor> broadcast_tensors(
      absl::Span<const XLATensor> tensors);

  static XLATensor tf_StatelessRandomNormal(absl::Span<const int64_t> size,
                                            const XLATensor& seeds,
                                            const Device& device,
                                            at::ScalarType scalar_type);

  static XLATensor to(XLATensor& input, c10::optional<Device> device,
                      c10::optional<at::ScalarType> scalar_type);

  static void linspace_out(XLATensor& out, at::Scalar start, at::Scalar stop,
                           int64_t num, at::ScalarType scalar_type);

  // XLA client operations exposed as tensor methods.

  static XLATensor xla_avg_pool(
      const XLATensor& input, absl::Span<const int64_t> kernel_size,
      absl::Span<const int64_t> stride,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      const xla::TensorFormat& data_format, const bool counts_include_padding);

  static XLATensor xla_avg_pool_grad(
      const XLATensor& out_backprop,
      absl::Span<const int64_t> gradients_size,
      absl::Span<const int64_t> kernel_size,
      absl::Span<const int64_t> stride,
      absl::Span<const std::pair<int64_t, int64_t>> spatial_padding,
      const xla::TensorFormat& data_format, const bool counts_include_padding);

  static XLATensor xla_max_pool(const XLATensor& input,
                                absl::Span<const int64_t> kernel_size,
                                absl::Span<const int64_t> stride,
                                xla::Padding padding,
                                const xla::TensorFormat& data_format);

  static XLATensor xla_max_pool_grad(const XLATensor& input,
                                     const XLATensor& out_backprop,
                                     absl::Span<const int64_t> kernel_size,
                                     absl::Span<const int64_t> stride,
                                     xla::Padding padding);

  static XLATensor xla_pad(const XLATensor& input, at::Scalar padding_value,
                           xla::PaddingConfig padding_config);

  static XLATensor xla_slice(const XLATensor& input,
                             absl::Span<const int64_t> start_indices,
                             absl::Span<const int64_t> limit_indices,
                             absl::Span<const int64_t> stride);

  static XLATensor xla_truncated_normal(const XLATensor& input);

  static XLATensor xla_replica_id(const Device& device);

 private:
  struct SyncTensorsConfig {
    // Whether we want to force XLA data on the target tensors (hence trimming
    // the IR graph above them).
    bool force_xla_data = true;
    // Whether when setting the XLA data, the other properties of the tensor
    // state should be reset.
    bool sync_xla_data = true;
  };

  struct SyncTensorCollection {
    SyncTensorCollection() : hash(0) {}

    SyncTensorsConfig config;
    std::vector<size_t> indices;
    xla::hash_t hash;
    std::vector<xla::util::ExceptionCleanup> unlocker;
    Device device;
  };

  struct PostOrderData {
    std::vector<const ir::Node*> post_order;
    ir::Util::EmissionMap emission_map;
    std::vector<xla::ComputationClient::DataPtr> parameters_data;
    std::vector<size_t> parameter_sequence;
  };

  struct CompilationResult {
    Device device;
    size_t emitted_nodes = 0;
    std::shared_ptr<xla::ComputationClient::Computation> computation;
    std::vector<xla::ComputationClient::DataPtr> parameters_data;
  };

  struct CachedComputation {
    CachedComputation(
        std::shared_ptr<xla::ComputationClient::Computation> computation)
        : computation(std::move(computation)) {}

    std::shared_ptr<xla::ComputationClient::Computation> computation;
  };

  using ComputationCache =
      xla::util::Cache<xla::hash_t, CachedComputation, xla::util::HashReducer>;

  struct Async {
    Async(SyncTensorCollection* coll,
          std::vector<xla::ComputationClient::DataPtr> parameters_data,
          std::vector<xla::ComputationClient::DataPtr> tensors_data,
          ComputationCache::TypePtr cached_computation);

    void Wait();

    xla::util::MultiWait mwait;
    std::vector<size_t> indices;
    std::vector<xla::util::ExceptionCleanup> unlocker;
    std::vector<xla::ComputationClient::DataPtr> parameters_data;
    std::string device;
    ComputationCache::TypePtr cached_computation;
    std::vector<xla::ComputationClient::DataPtr> tensors_data;
  };

  // This is the core XLA tensor data structure where all the tensor data is
  // held. The XLA tensor is nothing more than a shared pointer to a Data
  // object.
  struct Data {
    Data(xla::ComputationClient::DataPtr xla_data, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : xla_data(std::move(xla_data)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(ir::Value ir_value, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : ir_value(std::move(ir_value)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(at::Tensor tensor_data, const Device& device)
        : logical_element_type(tensor_data.scalar_type()),
          tensor_data(std::move(tensor_data)),
          device(device),
          unique_id(GetNextTensorId()) {}

    ~Data();

    xla::ComputationClient::DataPtr xla_data;
    ir::Value ir_value;
    c10::optional<at::ScalarType> logical_element_type;
    c10::optional<at::Tensor> tensor_data;
    const Device device;
    const int64_t unique_id = 0;
    size_t generation = 1;
  };

  XLATensor(const at::Tensor& tensor, const Device& device);
  XLATensor(xla::ComputationClient::DataPtr xla_data,
            c10::optional<at::ScalarType> logical_element_type = absl::nullopt);
  XLATensor(ir::Value ir_value, const Device& device,
            c10::optional<at::ScalarType> logical_element_type = absl::nullopt);
  XLATensor(std::shared_ptr<Data> data);

  Data* data() const;

  std::shared_ptr<Data> data_ptr() const { return data_; }

  void SetXlaData(xla::ComputationClient::DataPtr xla_data, bool sync);

  void SetIrValue(ir::Value ir_value);
  void SetInPlaceIrValue(ir::Value ir_value);

  void AssignIrValue(ir::Value ir_value) const;

  void SetTensorData(at::Tensor tensor_data);

  ir::Value CreateTensorNode(xla::ComputationClient::DataPtr data,
                             bool read_only) const;

  XLATensor CopyTensorToDevice(const Device& device);

 public:
  // Create a new XLA tensor with the same metadata of the input tensor (with
  // possible overrides), and the new IR value.
  XLATensor CreateFrom(ir::Value ir_value) const;
  XLATensor CreateFrom(ir::Value ir_value, const Device& device) const;
  XLATensor CreateFrom(ir::Value ir_value,
                       at::ScalarType logical_element_type) const;
  XLATensor CreateFrom(
      ir::Value ir_value,
      c10::optional<at::ScalarType> logical_element_type_opt) const;
  XLATensor CreateFrom(ir::Value ir_value, const Device& device,
                       at::ScalarType logical_element_type) const;
 private:

  // We build an XLA graph accumulating XLA operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  std::vector<XLATensor> MakeOutputTensors(ir::NodePtr node) const;

  ir::Value GetIrValueForTensor(const at::Tensor& tensor,
                                const Device& device) const;

  static ComputationCache* GetComputationCache();

  static SyncTensorCollection CollectSyncTensors(
      const std::vector<XLATensor>& tensors, const SyncTensorsConfig& config);

  // Implementation of the GetTensors() API using the op-by-op executor.
  static std::vector<at::Tensor> GetTensorsOpByOp(
      std::vector<XLATensor>* tensors);

  static std::vector<at::Tensor> GetTensorsFused(
      std::vector<XLATensor>* tensors);

  // Runs an asynchronous syn operation using the op-by-op executor.
  using OpByOpAsync = xla::util::AsyncTask<xla::Status>;
  static OpByOpAsync SyncTensorsGraphOpByOp(
      std::vector<XLATensor>* tensors, absl::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  // Gathers the XLA device data for all the input tensors, after an
  // asynchronous operation.
  static std::vector<xla::ComputationClient::DataPtr> GatherTensorsXlaData(
      const std::vector<XLATensor>& tensors, absl::Span<const size_t> indices,
      absl::Span<const xla::ComputationClient::DataPtr> tensors_data);

  static std::vector<ir::Value> CollectRoots(
      const std::vector<XLATensor>& tensors, absl::Span<const size_t> indices);

  static std::vector<xla::ComputationClient::DataPtr> FetchTensorData(
      std::vector<XLATensor>* tensors, const SyncTensorsConfig& config,
      absl::Span<const size_t> indices);

  // Schedules the execution of a sync tensors operation in background. The
  // asynchronous operation will hold the device locks by capturing the ones
  // present within the coll structure.
  static std::shared_ptr<XLATensor::Async> ScheduleSyncTensorsGraph(
      SyncTensorCollection* coll,
      std::vector<xla::ComputationClient::DataPtr> parameters_data,
      std::vector<xla::ComputationClient::DataPtr> tensors_data,
      ComputationCache::TypePtr cached_computation);

  static std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<XLATensor>* tensors, SyncTensorCollection* coll,
      std::vector<xla::ComputationClient::DataPtr> parameters_data,
      std::string device, ComputationCache::TypePtr cached_computation);

  static PostOrderData RunPostOrder(const std::vector<XLATensor>& tensors,
                                    absl::Span<const size_t> indices);

  static ComputationCache::TypePtr LookupCachedCompile(
      const std::vector<XLATensor>& tensors, const xla::hash_t& hash);

  static std::shared_ptr<Async> TryRunCachedSync(
      std::vector<XLATensor>* tensors, SyncTensorCollection* coll,
      PostOrderData* po_data);

  static void BuildInputOutputAliases(const std::vector<XLATensor>& tensors,
                                      absl::Span<const size_t> indices,
                                      ir::LoweringContext* lowering_ctx);

  static CompilationResult Compile(const std::vector<XLATensor>& tensors,
                                   absl::Span<const std::string> devices,
                                   const SyncTensorCollection& coll,
                                   PostOrderData* po_data);

  static std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<XLATensor>* tensors, absl::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  static int64_t GetNextTensorId();

  // Check if the current node is a cutpoint (by hash) and apply pending graph -
  // in other words, cut the trace - and return true iff that's the case.
  bool ApplyTraceletCutpoint();

  // Detect when new compilations are triggered after first few steps and
  // attempt to find common portions, after which the trace is cut. This is
  // meant to address variable upper bound loops, which lead to different
  // unrolled sequences of code despite the body being identical. The hope is to
  // reach a steady state in which no new tracelets are created after a
  // relatively small number of cuts.
  static void InsertTraceletCutpoint(const PostOrderData& po_data);

  std::shared_ptr<Data> data_;
};

}  // namespace swift_xla
