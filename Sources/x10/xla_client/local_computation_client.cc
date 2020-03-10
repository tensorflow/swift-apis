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

#include "tensorflow/compiler/xla/xla_client/local_computation_client.h"

#include <tuple>

#include "platforms/deepsea/executor/deepsea_platform.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace xla {
namespace {

// Dedup multiple computations going on at the same time. This is needed
// because multiple identical compilations running in parallel cost more
// than just one compilation.
// TODO(parkers): This deduping would happen best unified with the
// higher level cache.
struct ConcurrentCompileDedupping {
  struct Key {
    const xla::LocalClient* client;
    std::string result_layout;
    std::string computation;
    int64 num_replicas;
    size_t hash = ComputeHash();
    size_t ComputeHash() const {
      util::PartialHasher<std::string, 4096> hasher;
      return hasher(computation);
    }
    bool operator==(const Key& other) const {
      return computation == other.computation &&
             result_layout == other.result_layout &&
             num_replicas == other.num_replicas && client == other.client;
    }
  };
  struct Hasher {
    size_t operator()(const Key& key) const { return key.hash; }
  };
  struct Value {
    std::vector<std::shared_ptr<xla::LocalExecutable>*> result_listeners;
  };
  bool ShouldCompile(const Key& key,
                     std::shared_ptr<xla::LocalExecutable>* result_listener)
      EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    auto it = current_compiles.find(key);
    if (it != current_compiles.end()) {
      it->second.result_listeners.push_back(result_listener);
      return false;
    }
    current_compiles.emplace(key, Value{});
    return true;
  }

  void Publish(const Key& key, std::shared_ptr<xla::LocalExecutable> result)
      EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    auto it = current_compiles.find(key);
    for (auto* listener : it->second.result_listeners) {
      *listener = result;
    }
    current_compiles.erase(it);
  }

  absl::Mutex mutex;
  std::unordered_map<Key, Value, Hasher> current_compiles GUARDED_BY(mutex);
};

std::vector<xla::Shape> BuildArgumentLayouts(
    const XlaComputation& computation) {
  std::vector<xla::Shape> argument_layouts;

  argument_layouts.reserve(
      computation.proto().host_program_shape().parameters_size());
  for (const xla::ShapeProto& param :
       computation.proto().host_program_shape().parameters()) {
    argument_layouts.push_back(xla::Shape(param));
  }
  return argument_layouts;
}

std::vector<const xla::Shape*> ArgumentLayoutAsPointers(
    const std::vector<xla::Shape>& argument_layouts) {
  std::vector<const xla::Shape*> argument_layout_ptrs;
  argument_layout_ptrs.reserve(argument_layouts.size());
  for (const xla::Shape& shape : argument_layouts) {
    argument_layout_ptrs.push_back(&shape);
  }
  return argument_layout_ptrs;
}

}  // namespace

using DataPtr = ComputationClient::DataPtr;
using ComputationPtr = ComputationClient::ComputationPtr;
using Device = LocalComputationClient::Device;

class LocalComputationClient::Device {
 public:
  Device(xla::LocalClient* client, int device_ordinal, int32_t mesh_id,
         bool is_cpu)
      : client_(client),
        device_ordinal_(device_ordinal),
        mesh_id_(mesh_id),
        is_cpu_(is_cpu),
        stream_(std::make_unique<se::Stream>(
            client->backend().stream_executor(device_ordinal).ValueOrDie())) {
    stream_->Init();
  }

  xla::LocalClient* client() const { return client_; }
  int device_ordinal() const { return device_ordinal_; }
  int32_t mesh_id() const { return mesh_id_; }
  se::Stream* stream() const { return stream_.get(); }
  bool is_cpu() const { return is_cpu_; }

  void RunAsyncStart() {
    mutex_.Lock();
    XLA_CHECK(mutex_.AwaitWithTimeout(
        Condition(this, &Device::HasAvailableComputationSlots),
        absl::Hours(2)))
        << "TPU DEADLOCKED or very slow computation...";
    --available_computation_slots_;
    mutex_.Unlock();
  }
  void RunAsyncFinish() {
    mutex_.Lock();
    ++available_computation_slots_;
    mutex_.Unlock();
  }

  bool HasAvailableComputationSlots() EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return available_computation_slots_ > 0;
  }

 private:
  absl::Mutex mutex_;
  // This starts out as the number of allowable concurrent executions
  // on this particular device.
  int64 available_computation_slots_ GUARDED_BY(mutex_) = 8;
  xla::LocalClient* client_;
  int device_ordinal_;
  int32_t mesh_id_;
  bool is_cpu_;
  std::unique_ptr<se::Stream> stream_;
};

class LocalComputationClient::LocalData : public Data {
 public:
  LocalData(std::string device, Shape shape)
      : Data(std::move(device), std::move(shape)) {}
  LocalData(std::string device, ScopedShapedBuffer buffer)
      : Data(std::move(device), buffer.on_host_shape()),
        buffer_(std::make_shared<ScopedShapedBuffer>(std::move(buffer))) {}

  void Assign(const Data& data) override {
    const LocalData& xrt_data = dynamic_cast<const LocalData&>(data);
    if (&xrt_data != this) {
      buffer_ = xrt_data.buffer_;
    }
  }

  bool HasValue() const override { return buffer_ != nullptr; }

  OpaqueHandle GetOpaqueHandle() override {
    return reinterpret_cast<intptr_t>(buffer_.get());
  }

  const ShapedBuffer& buffer() const { return *buffer_; }

 private:
  // TODO(parkers): Remove Assign() and allow buffer_ to be by value.
  std::shared_ptr<ScopedShapedBuffer> buffer_;
};

struct LocalComputationClient::LocalComputation : public Computation {
  LocalComputation(XlaComputation computation, ProgramShape program_shape,
                   std::vector<std::string> devices,
                   std::shared_ptr<LocalExecutable> handle)
      : Computation(std::move(computation), std::move(program_shape),
                    std::move(devices)),
        handle(std::move(handle)) {}

  // This needs to be shared in order to hold onto computation until it has
  // finished async.
  std::shared_ptr<LocalExecutable> handle;
  std::shared_ptr<DeviceAssignment> assignment;
};

DataPtr LocalComputationClient::CreateDataPlaceholder(std::string device,
                                                      Shape shape) {
  return std::make_shared<LocalData>(device, shape);
}

DataPtr LocalComputationClient::TransferToServer(
    xla::BorrowingLiteral literal, const xla::Shape& dest_shape,
    const std::string& device_string) {
  tensorflow::profiler::TraceMe trace("TransferSingleTensorToServer");
  Device* device = GetDevice(device_string);

  int device_ordinal = device->device_ordinal();
  stream_executor::DeviceMemoryAllocator* allocator =
      device->client()->backend().memory_allocator();
  xla::TransferManager* transfer_manager =
      device->client()->backend().transfer_manager();

  se::Stream* stream = device->stream()->GetOrCreateSubStream();

  ScopedShapedBuffer buffer = [&] {
    tensorflow::profiler::TraceMe trace("Allocate");
    return transfer_manager
        ->AllocateScopedShapedBuffer(dest_shape, allocator, device_ordinal)
        .ValueOrDie();
  }();

  // TODO(parkers): Check if buffer is aliased and add dep on compute_stream.
  TF_CHECK_OK(
      transfer_manager->TransferLiteralToDeviceAsync(stream, literal, buffer));

  TF_CHECK_OK(stream->BlockHostUntilDone());

  device->stream()->ReturnSubStream(stream);

  return std::make_shared<LocalData>(device_string, std::move(buffer));
}

std::vector<DataPtr> LocalComputationClient::TransferToServer(
    absl::Span<const TensorSource> tensors) {
  tensorflow::profiler::TraceMe trace("TransferToServer");
  std::vector<std::unique_ptr<char[]>> buffers;
  buffers.resize(tensors.size());
  size_t total_size = 0;
  // TODO(parkers): This copy may not be strictly necessary when the layouts
  // are known to be properly sized from the start.
  util::MultiWait mwait(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    size_t size = xla::ShapeUtil::ByteSizeOf(tensors[i].shape);
    total_size += size;
    auto converter = [&, i, size]() {
      buffers[i] = std::make_unique<char[]>(size + 1);
      tensors[i].populate_fn(tensors[i], buffers[i].get(), size);
    };
    if (tensors.size() == 1) {
      mwait.Completer(std::move(converter))();
    } else {
      env::ScheduleClosure(mwait.Completer(std::move(converter)));
    }
  }
  mwait.Wait();

  OutboundDataMetric()->AddSample(total_size);

  struct ReturnSubStream {
    void operator()(se::Stream* substream) {
      if (substream) stream->ReturnSubStream(substream);
    }

    se::Stream* stream;
  };
  std::unordered_map<Device*, std::unique_ptr<se::Stream, ReturnSubStream>>
      streams;
  std::vector<DataPtr> out;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const TensorSource& tensor = tensors[i];
    Device* device = GetDevice(tensor.device);

    int device_ordinal = device->device_ordinal();
    stream_executor::DeviceMemoryAllocator* allocator =
        device->client()->backend().memory_allocator();
    xla::TransferManager* transfer_manager =
        device->client()->backend().transfer_manager();
    std::unique_ptr<se::Stream, ReturnSubStream>& stream = streams[device];

    if (!stream) {
      stream = std::unique_ptr<se::Stream, ReturnSubStream>(
          device->stream()->GetOrCreateSubStream(),
          ReturnSubStream{device->stream()});
    }

    ScopedShapedBuffer buffer = [&] {
      tensorflow::profiler::TraceMe trace("Allocate");
      return transfer_manager
          ->AllocateScopedShapedBuffer(tensor.shape, allocator, device_ordinal)
          .ValueOrDie();
    }();

    // TODO(parkers): Check if buffer is aliased and add dep on compute_stream.
    xla::BorrowingLiteral literal(buffers[i].get(), tensor.shape);

    TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
        stream.get(), literal, buffer));

    out.push_back(
        std::make_shared<LocalData>(tensor.device, std::move(buffer)));
  }

  for (auto& stream : streams) {
    TF_CHECK_OK(stream.second->BlockHostUntilDone());
  }
  return out;
}

std::vector<Literal> LocalComputationClient::TransferFromServer(
    absl::Span<const DataPtr> handles) {
  tensorflow::profiler::TraceMe trace("TransferFromServer");
  metrics::TimedSection timed(TransferFromServerMetric());
  std::unordered_set<Device*> devices;
  for (size_t i = 0; i < handles.size(); ++i) {
    const auto& local_data = dynamic_cast<const LocalData&>(*handles[i]);
    devices.insert(GetDevice(local_data.device()));
  }

  // Block until all compute is done before transfering from the server.
  for (Device* device : devices) {
    TF_CHECK_OK(device->stream()->BlockHostUntilDone());
  }

  std::vector<Literal> out;
  out.resize(handles.size());
  util::MultiWait mwait(handles.size());
  for (size_t i = 0; i < handles.size(); ++i) {
    const auto& local_data = dynamic_cast<const LocalData&>(*handles[i]);
    out[i] = xla::Literal(local_data.buffer().on_host_shape());
    Device* device = GetDevice(local_data.device());
    xla::TransferManager* transfer_manager =
        device->client()->backend().transfer_manager();
    transfer_manager->TransferLiteralFromDevice(device->stream(),
                                                local_data.buffer(), &out[i],
                                                [&mwait](Status status) {
                                                  TF_CHECK_OK(status);
                                                  mwait.Done();
                                                });
  }
  mwait.Wait();
  return out;
}

std::vector<ComputationPtr> LocalComputationClient::Compile(
    std::vector<CompileInstance> instances) {
  metrics::TimedSection timed(CompileMetric());
  // TODO(parkers): Probably have to do this in parallel if actually given
  // n computations.
  std::vector<ComputationPtr> out;
  for (CompileInstance& instance : instances) {
    Device* device = GetDevice(instance.compilation_device);

    std::unique_ptr<xla::DeviceAssignment> assignment;
    if (instance.devices.size() > 1) {
      assignment =
          std::make_unique<xla::DeviceAssignment>(instance.devices.size(), 1);
      for (size_t i = 0; i < instance.devices.size(); ++i) {
        if (absl::string_view(instance.devices[i]).substr(0, 10) ==
            "REMOTE_TPU") {
          auto it = remote_devices_.find(instance.devices[i]);
          if (it == remote_devices_.end()) {
            TF_LOG(FATAL) << "Could not find device: " << instance.devices[i];
          }
          (*assignment)(i, 0) = it->second;
        } else {
          (*assignment)(i, 0) = GetDevice(instance.devices[i])->mesh_id();
        }
      }
    }

    tensorflow::profiler::TraceMe trace([&] {
      return absl::StrCat("XLA Compile: ", instance.compilation_device);
    });

    const XlaComputation& computation = instance.computation;
    std::vector<xla::Shape> argument_layouts =
        BuildArgumentLayouts(computation);
    xla::ExecutableBuildOptions exec_build_options;

    if (instance.output_shape) {
      exec_build_options.set_result_layout(*instance.output_shape);
    }
    exec_build_options.set_device_ordinal(device->device_ordinal());
    exec_build_options.set_num_replicas(instance.devices.size());

    std::shared_ptr<xla::LocalExecutable> xla_computation;
    static auto* deduping = new ConcurrentCompileDedupping;

    deduping->mutex.Lock();
    ConcurrentCompileDedupping::Key key{
        device->client(),
        exec_build_options.result_layout()
            ? exec_build_options.result_layout()->ToProto().SerializeAsString()
            : "",
        computation.proto().SerializeAsString(),
        exec_build_options.num_replicas()};

    if (deduping->ShouldCompile(key, &xla_computation)) {
      deduping->mutex.Unlock();
      xla_computation = std::move(
          device->client()
              ->Compile(computation, ArgumentLayoutAsPointers(argument_layouts),
                        exec_build_options)
              .ValueOrDie()
              .front());
      deduping->mutex.Lock();
      deduping->Publish(key, xla_computation);
    }
    auto cond = [&]() { return xla_computation != nullptr; };
    deduping->mutex.Await(absl::Condition(&cond));
    deduping->mutex.Unlock();

    auto local_computation = std::make_shared<LocalComputation>(
        std::move(instance.computation),
        xla::ProgramShape(instance.computation.GetProgramShape().ValueOrDie()),
        instance.devices, std::move(xla_computation));
    local_computation->assignment = std::move(assignment);
    out.push_back(std::move(local_computation));
  }
  return out;
}

std::vector<DataPtr> LocalComputationClient::ExecuteComputation(
    const Computation& computation, absl::Span<const DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  auto& local_computation = dynamic_cast<const LocalComputation&>(computation);
  Device* device_ptr = GetDevice(device);
  std::vector<const xla::ShapedBuffer*> args;
  for (const DataPtr& opaque_arg : arguments) {
    args.push_back(&dynamic_cast<const LocalData&>(*opaque_arg).buffer());
  }

  std::unique_ptr<xla::DeviceAssignment> devices;

  xla::ExecutableRunOptions run_options;
  run_options.set_stream(device_ptr->stream());
  run_options.set_allocator(device_ptr->client()->backend().memory_allocator());
  run_options.set_intra_op_thread_pool(
      device_ptr->client()->backend().eigen_intra_op_thread_pool_device());

  run_options.set_device_assignment(local_computation.assignment.get());

  bool is_cpu = device_ptr->is_cpu();
  if (!is_cpu) {
    tensorflow::profiler::TraceMe trace("Acquire Async slot");
    device_ptr->RunAsyncStart();
  }
  xla::ScopedShapedBuffer tmp =
      local_computation.handle->RunAsync(args, run_options).ValueOrDie();
  size_t num_tuples = tmp.on_host_shape().tuple_shapes().size();
  std::vector<DataPtr> out;
  out.reserve(num_tuples);
  for (size_t i = 0; i < num_tuples; ++i) {
    out.push_back(std::make_shared<LocalData>(
        device, tmp.TakeSubTree(ShapeIndex({static_cast<xla::int64>(i)}))));
  }

  if (is_cpu) {
    TF_CHECK_OK(run_options.stream()->BlockHostUntilDone());
  } else {
    run_options.stream()->ThenDoHostCallback(
        [handle = local_computation.handle,
         assignment = local_computation.assignment,
         device_ptr]() { device_ptr->RunAsyncFinish(); });
  }

  return out;
}

std::vector<std::vector<DataPtr>> LocalComputationClient::ExecuteReplicated(
    const Computation& computation,
    const std::vector<std::vector<DataPtr>>& arguments,
    absl::Span<const std::string> devices,
    const ExecuteReplicatedOptions& options) {
  TF_LOG(FATAL) << "Implement";
}

std::vector<std::vector<DataPtr>> LocalComputationClient::ExecuteParallel(
    absl::Span<const Computation* const> computations,
    const std::vector<std::vector<DataPtr>>& arguments,
    absl::Span<const std::string> devices,
    const ExecuteParallelOptions& options) {
  TF_LOG(FATAL) << "Implement";
}

std::vector<DataPtr> LocalComputationClient::ExecuteChained(
    absl::Span<const ExecuteChainedOp> ops, const std::string& device) {
  TF_LOG(FATAL) << "Implement";
}

std::vector<std::vector<DataPtr>> LocalComputationClient::DeconstructTuple(
    absl::Span<const DataPtr> tuples) {
  TF_LOG(FATAL) << "Implement";
}

std::string LocalComputationClient::GetResourceDomain(
    const std::string& device) const {
  return GetDevice(device)->client()->platform()->Name();
}

std::string LocalComputationClient::GetDefaultDevice() const {
  return default_device_;
}

size_t LocalComputationClient::GetNumDevices() const { return devices_.size(); }

std::vector<std::string> LocalComputationClient::GetLocalDevices() const {
  TF_LOG(FATAL) << "Implement";
}

std::vector<std::string> LocalComputationClient::GetAllDevices() const {
  return device_names_;
}

static std::vector<std::string>* replicated_devices =
    new std::vector<std::string>;

void LocalComputationClient::SetReplicationDevices(
    std::vector<std::string> devices) {
  *replicated_devices = devices;
}

const std::vector<std::string>& LocalComputationClient::GetReplicationDevices()
    const {
  return *replicated_devices;
}

void LocalComputationClient::SetRngSeed(size_t seed) {
  TF_LOG(FATAL) << "Implement";
}

std::map<std::string, Metric>
LocalComputationClient::GetMetrics() const {
  TF_LOG(FATAL) << "Implement";
}

Device* LocalComputationClient::GetDevice(std::string device) const {
  auto it = devices_.find(device);
  if (it == devices_.end()) {
    TF_LOG(FATAL) << "Could not find device: " << device;
  }
  return it->second.get();
}

StatusOr<xla::LocalClient*> getTPULocalClient() {
  deepsea::executor::DeepseaPlatform* platform =
      deepsea::executor::DeepseaPlatform::GetRegisteredPlatform();
  if (!platform->Initialized()) {
    TF_RETURN_IF_ERROR(platform->Initialize({}));
  }
  if (platform->VisibleDeviceCount() <= 0) {
    return xla::InvalidArgument("No TPU devices found.");
  }
  xla::LocalClientOptions options;
  options.set_platform(platform);

  return xla::ClientLibrary::GetOrCreateLocalClient(options);
}

LocalComputationClient::LocalComputationClient() {
  xla::LocalClientOptions options;
  options.set_platform(xla::PlatformUtil::GetPlatform("cpu").ValueOrDie());
  xla::LocalClient* cpu_client =
      xla::ClientLibrary::GetOrCreateLocalClient(options).ValueOrDie();
  for (int i = 0; i < cpu_client->device_count(); ++i) {
    std::string key = absl::StrCat("CPU:", i);
    devices_[key] = std::make_unique<Device>(cpu_client, i, i, true);
    device_names_.push_back(key);
  }
  auto tpu_client_statusor = getTPULocalClient();
  if (tpu_client_statusor.ok()) {
    xla::LocalClient* tpu_client = tpu_client_statusor.ValueOrDie();
    deepsea::executor::DeepseaPlatform* platform =
        deepsea::executor::DeepseaPlatform::GetRegisteredPlatform();

    XLA_CHECK_EQ(tpu_client->device_count(),
                 platform->GetHostLocation()
                     .Cores(deepsea::executor::TpuCoreType::kTensorCore)
                     .size());

    size_t next_remote_tpu_id = 0;
    for (const tpu::TpuHostLocation& host : platform->topology().hosts()) {
      auto cores = host.Cores(deepsea::executor::TpuCoreType::kTensorCore);
      if (host == platform->GetHostLocation()) {
        for (size_t i = 0; i < tpu_client->device_count(); ++i) {
          std::string key = absl::StrCat("TPU:", i);
          devices_[key] =
              std::make_unique<Device>(tpu_client, i, cores[i].Id(), false);
          device_names_.push_back(key);
          if (i == 0) default_device_ = key;
          LOG(INFO) << "Found devices: TPU: " << i
                    << " core_id: " << cores[i].Id() << "\n";
        }
      } else {
        for (const auto& core : cores) {
          std::string key = absl::StrCat("REMOTE_TPU:", next_remote_tpu_id);
          device_names_.push_back(key);
          remote_devices_[key] = core.Id();
          LOG(INFO) << "Found devices: REMOTE_TPU:" << next_remote_tpu_id
                    << " = " << core << "\n";
          ++next_remote_tpu_id;
        }
      }
    }

  } else {
    LOG(INFO) << "Could not find any TPU Devices: "
              << tpu_client_statusor.status();
  }
  LOG(INFO) << "LocalComputationClient initialized";
}

std::unique_ptr<ComputationClient> ComputationClient::Create() {
  return std::make_unique<LocalComputationClient>();
}

bool ComputationClient::IsLocal() { return true; }

LocalComputationClient::~LocalComputationClient() {}

}  // namespace xla
