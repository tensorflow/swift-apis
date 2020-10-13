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

#ifndef X10_XLA_CLIENT_LOCAL_DEVICE_IMPL_H_
#define X10_XLA_CLIENT_LOCAL_DEVICE_IMPL_H_

#include <string>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/client/client_library.h"

namespace xla {

std::unique_ptr<ComputationClient::Device> MakeLocalDeviceFromClient(
    std::string name, xla::LocalClient* client, int device_ordinal,
    int32_t mesh_id, bool is_cpu);

std::vector<std::unique_ptr<ComputationClient::Device>>
GetAllLocalDevicesForPlatform(const char* platform_name,
                              const char* device_prefix);

}  // namespace xla

#endif  // X10_XLA_CLIENT_LOCAL_DEVICE_IMPL_H_
