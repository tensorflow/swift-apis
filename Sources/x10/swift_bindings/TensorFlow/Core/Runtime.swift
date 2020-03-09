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

public enum _RuntimeConfig {
  /// When true, prints various debug messages on the runtime state.
  ///
  /// If the value is true when running tensor computation for the first time in the process, INFO
  /// log from TensorFlow will also get printed.
  static public var printsDebugLog = false
}
