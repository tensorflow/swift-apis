#include "tensorflow/compiler/xla/xla_client/local_device.h"

#include <tuple>

#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace xla {
namespace {

using Data = ComputationClient::Data;
using Device = ComputationClient::Device;
using TransferManager = ComputationClient::TransferManager;
using DataPtr = ComputationClient::DataPtr;
using ComputationPtr = ComputationClient::ComputationPtr;
using TensorSource = ComputationClient::TensorSource;
using CompileInstance = ComputationClient::CompileInstance;
using Computation = ComputationClient::Computation;
using ExecuteComputationOptions = ComputationClient::ExecuteComputationOptions;

namespace {

std::unique_ptr<xla::DeviceAssignment> GetAssignment(
    const std::vector<std::string>& devices) {
  std::unique_ptr<xla::DeviceAssignment> assignment;
  if (devices.size() > 1) {
    assignment = std::make_unique<xla::DeviceAssignment>(devices.size(), 1);
    for (size_t i = 0; i < devices.size(); ++i) {
      (*assignment)(i, 0) = GetX10Device(devices[i])->mesh_id();
    }
  }
  return assignment;
}

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
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    auto it = current_compiles.find(key);
    if (it != current_compiles.end()) {
      it->second.result_listeners.push_back(result_listener);
      return false;
    }
    current_compiles.emplace(key, Value{});
    return true;
  }

  void Publish(const Key& key, std::shared_ptr<xla::LocalExecutable> result)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    auto it = current_compiles.find(key);
    for (auto* listener : it->second.result_listeners) {
      *listener = result;
    }
    current_compiles.erase(it);
  }

  absl::Mutex mutex;
  absl::node_hash_map<Key, Value, Hasher> current_compiles
      ABSL_GUARDED_BY(mutex);
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

class LocalTransferManager : public ComputationClient::TransferManager {
 public:
  std::vector<Literal> TransferFromServerImpl(
      absl::Span<const DataPtr> handles) override;
};

class LocalDevice : public ComputationClient::Device {
 public:
  LocalDevice(std::string name, xla::LocalClient* client, int device_ordinal,
              int32_t mesh_id, bool is_cpu)
      : Device(name),
        client_(client),
        device_ordinal_(device_ordinal),
        mesh_id_(mesh_id),
        is_cpu_(is_cpu),
        stream_(std::make_unique<se::Stream>(
            client->backend().stream_executor(device_ordinal).ValueOrDie())),
        transfer_from_device_stream_(std::make_unique<se::Stream>(
            client->backend().stream_executor(device_ordinal).ValueOrDie())) {
    stream_->Init();
    transfer_from_device_stream_->Init();
  }

  xla::LocalClient* client() const { return client_; }
  int device_ordinal() const { return device_ordinal_; }
  int32_t mesh_id() const final { return mesh_id_; }
  se::Stream* stream() const { return stream_.get(); }
  se::Stream* transfer_from_device_stream() const {
    return transfer_from_device_stream_.get();
  }
  bool is_cpu() const { return is_cpu_; }
  TransferManager* GetTransferManager() const override {
    static LocalTransferManager local_transfer;
    return &local_transfer;
  }
  virtual bool IsLocal() { return true; }

  int64 RunAsyncStart() {
    mutex_.Lock();
    XLA_CHECK(mutex_.AwaitWithTimeout(
        absl::Condition(this, &LocalDevice::HasAvailableComputationSlots),
        absl::Hours(2)))
        << "TPU DEADLOCKED or very slow computation...";
    --available_computation_slots_;
    int64 result = next_computation_id_;
    ++next_computation_id_;
    mutex_.Unlock();
    return result;
  }
  void RunAsyncFinish() {
    mutex_.Lock();
    ++available_computation_slots_;
    ++done_computation_id_;
    mutex_.Unlock();
  }

  void WaitUntilComputationFinished(int64 computation_id) {
    mutex_.Lock();
    auto cond = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
      return computation_id < done_computation_id_;
    };
    XLA_CHECK(mutex_.AwaitWithTimeout(absl::Condition(&cond), absl::Hours(2)))
        << "TPU DEADLOCKED or very slow computation...";
    mutex_.Unlock();
  }

  bool HasAvailableComputationSlots() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return available_computation_slots_ > 0;
  }

  DataPtr CreateDataPlaceholder(Shape shape);

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override;

  DataPtr TransferToServer(xla::BorrowingLiteral literal,
                           const xla::Shape& dest_shape) override;

  std::vector<ComputationClient::ComputationPtr> Compile(
      const std::vector<std::string>& devices,
      std::vector<CompileInstance> instances) override;

  std::string ResourceDomain() const override {
    return client_->platform()->Name();
  }

  std::vector<ComputationClient::DataPtr> ExecuteChained(
      absl::Span<const ComputationClient::ExecuteChainedOp> ops) override {
    TF_LOG(FATAL) << "Implement";
  }

  std::vector<ComputationClient::DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const ExecuteComputationOptions& options) override;

 private:
  absl::Mutex mutex_;
  // This starts out as the number of allowable concurrent executions
  // on this particular device.
  int64 available_computation_slots_ ABSL_GUARDED_BY(mutex_) = 64;
  // computation id assigned to the next computation executed on stream_.
  int64 next_computation_id_ ABSL_GUARDED_BY(mutex_) = 0;
  // Incremented when computations finish.
  // `computation_id < done_computation_id_` checks if a particular computation
  // is complete.
  int64 done_computation_id_ ABSL_GUARDED_BY(mutex_) = 0;
  xla::LocalClient* client_;
  int device_ordinal_;
  int32_t mesh_id_;
  bool is_cpu_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::Stream> transfer_from_device_stream_;
};

class LocalData : public Data {
 public:
  LocalData(Device* device, Shape shape) : Data(device, std::move(shape)) {}
  LocalData(Device* device, ScopedShapedBuffer buffer, int64 computation_id)
      : Data(device, buffer.on_host_shape()),
        buffer_(std::make_shared<ScopedShapedBuffer>(std::move(buffer))),
        computation_id_(computation_id) {}

  void Assign(const Data& data) override {
    const LocalData& xrt_data = dynamic_cast<const LocalData&>(data);
    if (&xrt_data != this) {
      buffer_ = xrt_data.buffer_;
      computation_id_ = xrt_data.computation_id_;
    }
  }

  bool HasValue() const override { return buffer_ != nullptr; }

  OpaqueHandle GetOpaqueHandle() override {
    return reinterpret_cast<intptr_t>(buffer_.get());
  }

  const ShapedBuffer& buffer() const { return *buffer_; }

  int64 computation_id() const { return computation_id_; }

 private:
  // TODO(parkers): Remove Assign() and allow buffer_ to be by value.
  std::shared_ptr<ScopedShapedBuffer> buffer_;
  int64 computation_id_;
};

struct LocalComputation : public Computation {
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

DataPtr LocalDevice::CreateDataPlaceholder(Shape shape) {
  return std::make_shared<LocalData>(this, shape);
}

DataPtr LocalDevice::TransferToServer(xla::BorrowingLiteral literal,
                                      const xla::Shape& dest_shape) {
  tensorflow::profiler::TraceMe trace("TransferSingleTensorToServer");

  stream_executor::DeviceMemoryAllocator* allocator =
      client()->backend().memory_allocator();
  xla::TransferManager* transfer_manager =
      client()->backend().transfer_manager();

  se::Stream* stream = stream_->GetOrCreateSubStream();

  ScopedShapedBuffer buffer = [&] {
    tensorflow::profiler::TraceMe trace("Allocate");
    return transfer_manager
        ->AllocateScopedShapedBuffer(dest_shape, allocator, device_ordinal_)
        .ValueOrDie();
  }();

  // TODO(parkers): Check if buffer is aliased and add dep on compute_stream.
  TF_CHECK_OK(
      transfer_manager->TransferLiteralToDeviceAsync(stream, literal, buffer));

  TF_CHECK_OK(stream->BlockHostUntilDone());

  stream_->ReturnSubStream(stream);

  return std::make_shared<LocalData>(this, std::move(buffer), -1);
}

std::vector<DataPtr> LocalDevice::TransferToServer(
    absl::Span<const TensorSource> tensors) {
  auto* device = this;
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

  ComputationClient::OutboundDataMetric()->AddSample(total_size);

  struct ReturnSubStream {
    void operator()(se::Stream* substream) {
      if (substream) stream->ReturnSubStream(substream);
    }

    se::Stream* stream;
  };
  std::unique_ptr<se::Stream, ReturnSubStream> stream;
  std::vector<DataPtr> out;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const TensorSource& tensor = tensors[i];

    stream_executor::DeviceMemoryAllocator* allocator =
        device->client()->backend().memory_allocator();
    xla::TransferManager* transfer_manager =
        device->client()->backend().transfer_manager();

    if (!stream) {
      stream = std::unique_ptr<se::Stream, ReturnSubStream>(
          device->stream()->GetOrCreateSubStream(),
          ReturnSubStream{device->stream()});
    }

    ScopedShapedBuffer buffer = [&] {
      tensorflow::profiler::TraceMe trace("Allocate");
      return transfer_manager
          ->AllocateScopedShapedBuffer(tensor.shape, allocator,
                                       device_ordinal())
          .ValueOrDie();
    }();

    // TODO(parkers): Check if buffer is aliased and add dep on compute_stream.
    xla::BorrowingLiteral literal(buffers[i].get(), tensor.shape);

    TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
        stream.get(), literal, buffer));

    out.push_back(std::make_shared<LocalData>(device, std::move(buffer), -1));
  }

  TF_CHECK_OK(stream->BlockHostUntilDone());
  return out;
}

std::vector<Literal> LocalTransferManager::TransferFromServerImpl(
    absl::Span<const DataPtr> handles) {
  tensorflow::profiler::TraceMe trace("TransferFromServer");
  metrics::TimedSection timed(ComputationClient::TransferFromServerMetric());
  absl::node_hash_set<LocalDevice*> devices;
  {
    tensorflow::profiler::TraceMe trace("Wait for transfer");
    for (size_t i = 0; i < handles.size(); ++i) {
      const auto& local_data = dynamic_cast<const LocalData&>(*handles[i]);
      // Block until all compute is done before transfering from the server.
      dynamic_cast<LocalDevice*>(local_data.device())
          ->WaitUntilComputationFinished(local_data.computation_id());
    }
  }

  std::vector<Literal> out;
  out.resize(handles.size());
  util::MultiWait mwait(handles.size());
  for (size_t i = 0; i < handles.size(); ++i) {
    const auto& local_data = dynamic_cast<const LocalData&>(*handles[i]);
    out[i] = xla::Literal(local_data.buffer().on_host_shape());
    LocalDevice* device = dynamic_cast<LocalDevice*>(local_data.device());
    xla::TransferManager* transfer_manager =
        device->client()->backend().transfer_manager();
    transfer_manager->TransferLiteralFromDevice(
        device->transfer_from_device_stream(), local_data.buffer(), &out[i],
        [&mwait](Status status) {
          TF_CHECK_OK(status);
          mwait.Done();
        });
  }
  mwait.Wait();
  return out;
}

std::vector<ComputationPtr> LocalDevice::Compile(
    const std::vector<std::string>& devices,
    std::vector<CompileInstance> instances) {
  metrics::TimedSection timed(ComputationClient::CompileMetric());
  // TODO(parkers): Probably have to do this in parallel if actually given
  // n computations.
  std::vector<ComputationPtr> out;
  for (CompileInstance& instance : instances) {
    std::unique_ptr<xla::DeviceAssignment> assignment = GetAssignment(devices);

    tensorflow::profiler::TraceMe trace(
        [&] { return absl::StrCat("XLA Compile: ", name()); });

    const XlaComputation& computation = instance.computation;
    std::vector<xla::Shape> argument_layouts =
        BuildArgumentLayouts(computation);
    xla::ExecutableBuildOptions exec_build_options;

    if (instance.output_shape) {
      exec_build_options.set_result_layout(*instance.output_shape);
    }
    exec_build_options.set_device_ordinal(device_ordinal());
    exec_build_options.set_num_replicas(devices.size());

    std::shared_ptr<xla::LocalExecutable> xla_computation;
    static auto* deduping = new ConcurrentCompileDedupping;

    deduping->mutex.Lock();
    ConcurrentCompileDedupping::Key key{
        client(),
        exec_build_options.result_layout()
            ? exec_build_options.result_layout()->ToProto().SerializeAsString()
            : "",
        computation.proto().SerializeAsString(),
        exec_build_options.num_replicas()};

    if (deduping->ShouldCompile(key, &xla_computation)) {
      deduping->mutex.Unlock();
      xla_computation = std::move(
          client()
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
        devices, std::move(xla_computation));
    local_computation->assignment = std::move(assignment);
    out.push_back(std::move(local_computation));
  }
  return out;
}

std::vector<DataPtr> LocalDevice::ExecuteComputation(
    const Computation& computation, absl::Span<const DataPtr> arguments,
    const ExecuteComputationOptions& options) {
  auto& local_computation = dynamic_cast<const LocalComputation&>(computation);
  std::vector<const xla::ShapedBuffer*> args;
  for (const DataPtr& opaque_arg : arguments) {
    args.push_back(&dynamic_cast<const LocalData&>(*opaque_arg).buffer());
  }

  std::unique_ptr<xla::DeviceAssignment> devices;

  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream_.get());
  run_options.set_allocator(client_->backend().memory_allocator());
  run_options.set_intra_op_thread_pool(
      client_->backend().eigen_intra_op_thread_pool_device());

  run_options.set_device_assignment(local_computation.assignment.get());

  bool is_cpu = this->is_cpu();
  int64 computation_id = -1;
  if (!is_cpu) {
    tensorflow::profiler::TraceMe trace("Acquire Async slot");
    computation_id = RunAsyncStart();
  }
  xla::ScopedShapedBuffer tmp =
      local_computation.handle->RunAsync(args, run_options).ValueOrDie();
  size_t num_tuples = tmp.on_host_shape().tuple_shapes().size();
  std::vector<DataPtr> out;
  out.reserve(num_tuples);
  for (size_t i = 0; i < num_tuples; ++i) {
    out.push_back(std::make_shared<LocalData>(
        this, tmp.TakeSubTree(ShapeIndex({static_cast<xla::int64>(i)})),
        computation_id));
  }

  if (is_cpu) {
    TF_CHECK_OK(run_options.stream()->BlockHostUntilDone());
  } else {
    run_options.stream()->ThenDoHostCallback(
        [handle = local_computation.handle,
         assignment = local_computation.assignment,
         this]() { RunAsyncFinish(); });
  }

  return out;
}

}  // namespace

std::unique_ptr<ComputationClient::Device> MakeLocalDeviceFromClient(
    std::string name, xla::LocalClient* client, int device_ordinal,
    int32_t mesh_id, bool is_cpu) {
  return std::make_unique<LocalDevice>(name, client, device_ordinal, mesh_id,
                                       is_cpu);
}

std::vector<std::unique_ptr<ComputationClient::Device>>
GetAllLocalDevicesForPlatform(const char* platform_name,
                              const char* device_prefix) {
  xla::LocalClientOptions options;
  options.set_platform(
      xla::PlatformUtil::GetPlatform(platform_name).ValueOrDie());
  xla::LocalClient* client =
      xla::ClientLibrary::GetOrCreateLocalClient(options).ValueOrDie();
  std::vector<std::unique_ptr<ComputationClient::Device>> devices;
  devices.reserve(client->device_count());
  for (int i = 0; i < client->device_count(); ++i) {
    devices.push_back(MakeLocalDeviceFromClient(
        absl::StrCat(device_prefix, ":", i), client, i, i, true));
  }
  return devices;
}

}  // namespace xla
