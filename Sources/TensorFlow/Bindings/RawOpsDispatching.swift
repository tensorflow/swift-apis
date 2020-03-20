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

@available(
  *, deprecated, renamed: "_Raw",
  message:
    """
  'Raw' has been renamed to '_Raw' to indicate that it is not a guaranteed/stable API.
  """
)
public typealias Raw = _Raw

#if USING_X10_BACKEND
public typealias _Raw = _RawXLA
#else
public typealias _Raw = _RawTFEager
#endif
