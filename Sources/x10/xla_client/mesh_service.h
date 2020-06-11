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

#ifndef X10_XLA_CLIENT_MESH_SERVICE_H_
#define X10_XLA_CLIENT_MESH_SERVICE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/xla_client/mesh_service.pb.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace service {

class MeshService {
  struct Impl;

 public:
  MeshService(const std::string& address, grpc::Config config);

  ~MeshService();

 private:
  std::unique_ptr<Impl> impl_;
};

class MeshClient {
  struct Impl;

 public:
  static MeshClient* Get();

  const std::string& address() const;

  grpc::Config GetConfig() const;

  std::vector<std::string> Rendezvous(int ordinal, const std::string& tag,
                                      const std::string& payload,
                                      absl::Span<const int64> replicas) const;

  std::string GetNcclUniqueUid(absl::Span<const int64> replicas) const;

 private:
  explicit MeshClient(const std::string& address);

  ~MeshClient();

  std::unique_ptr<Impl> impl_;
};

}  // namespace service
}  // namespace xla

#endif  // X10_XLA_CLIENT_MESH_SERVICE_H_
