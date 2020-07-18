// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if USING_X10_BACKEND
  @_implementationOnly import x10_xla_tensor_wrapper
#endif

#if USING_X10_BACKEND
  extension _Raw {
    public static func commonBackend(
      _ a: Device.Backend, _ b: Device.Backend, file: StaticString = #file, line: UInt = #line
    ) -> Device.Backend {
      return .TF_EAGER
    }
  }
#endif
