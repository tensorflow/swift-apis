#include "absl/strings/str_format.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/ops/token.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"

// clang-format off
// For TPU, run with:
// export XRT_TPU_CONFIG="tpu_worker;0;localhost:<tfrc port>"
// clang-format on

using swift_xla::Device;
using swift_xla::DeviceType;
using swift_xla::GetDefaultDevice;
using swift_xla::XLATensor;

namespace {

void WithAllDevices(
    DeviceType device_type,
    const std::function<void(const std::vector<Device>&,
                             const std::vector<Device>&)>& devfn) {
  std::vector<Device> devices;
  std::vector<Device> all_devices;
  for (const auto& device_str :
       xla::ComputationClient::Get()->GetLocalDevices()) {
    Device device(device_str);
    if (device.hw_type == device_type) {
      devices.push_back(device);
    }
  }
  for (const auto& device_str :
       xla::ComputationClient::Get()->GetAllDevices()) {
    Device device(device_str);
    if (device.hw_type == device_type) {
      all_devices.push_back(device);
    }
  }
  if (!devices.empty()) {
    devfn(devices, all_devices);
  }
}

void TestSingleReplication(const std::vector<Device>& all_devices) {
  // Simulates N threads executing the same computation, using separated XRT
  // executions, and issuing CRS operations.
  std::vector<xla::string> device_strings;
  device_strings.reserve(all_devices.size());
  for (auto& device : all_devices) {
    device_strings.push_back(device.ToString());
  }
  float content = 1;
  std::vector<XLATensor> inputs;
  for (const auto& device : all_devices) {
    at::Tensor c({content, content + 1}, {2});
    inputs.push_back(XLATensor::Create(c, device));
    content += 2;
  }
  auto token = swift_xla::ir::MakeNode<swift_xla::ir::ops::Token>();
  std::vector<XLATensor> results;
  results.reserve(device_strings.size());
  for (const auto& input : inputs) {
    auto reduced_and_token =
        XLATensor::all_reduce(std::vector<XLATensor>{input}, token,
                              swift_xla::AllReduceType::kSum, 1., {});
    results.push_back(reduced_and_token.first.back());
  }
  xla::util::MultiWait mwait(device_strings.size());
  for (size_t i = 0; i < results.size(); ++i) {
    auto executor = [&, i]() {
      XLATensor::SyncLiveTensorsGraph(/*device=*/&results[i].GetDevice(),
                                      /*devices=*/device_strings,
                                      /*wait=*/true);
    };
    xla::env::ScheduleIoClosure(mwait.Completer(std::move(executor)));
  }
  mwait.Wait();
  for (XLATensor& result : results) {
    at::Tensor cpu_result = result.ToTensor();
    auto data = cpu_result.data<float>();
    absl::PrintF("crs result: [%g, %g]\n", data[0], data[1]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  at::Tensor a({1, 2}, {2});
  at::Tensor b({7, 19}, {2});

  auto tena = XLATensor::Create(a, *GetDefaultDevice());
  auto tenb = XLATensor::Create(b, *GetDefaultDevice());

  auto tmp = XLATensor::add(tena, tenb, at::Scalar(1.0));
  auto result = tmp.ToTensor();

  auto data = result.data<float>();
  if (data.size() == 2) {
    // TODO(parkers): Better printing...
    absl::PrintF("result: [%g, %g]\n", data[0], data[1]);
  } else {
    absl::PrintF("result; ??x%d\n", static_cast<int>(data.size()));
  }
  WithAllDevices(DeviceType::TPU, [&](const std::vector<Device>& /*devices*/,
                                      const std::vector<Device>& all_devices) {
    TestSingleReplication(all_devices);
  });
  return 0;
}
