// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
//
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import PackageDescription

let package = Package(
  name: "TensorFlow",
  platforms: [
    .macOS(.v10_13)
  ],
  products: [
    .library(
      name: "TensorFlow",
      type: .dynamic,
      targets: ["TensorFlow"]),
    .library(
      name: "Tensor",
      type: .dynamic,
      targets: ["Tensor"]),
  ],
  dependencies: [],
  targets: [
    .target(
      name: "Tensor",
      dependencies: []),
    .target(
      name: "TensorFlow",
      dependencies: ["Tensor"]),
    .target(
      name: "Experimental",
      dependencies: [],
      path: "Sources/third_party/Experimental"),
    .testTarget(
      name: "ExperimentalTests",
      dependencies: ["Experimental"]),
    .testTarget(
      name: "TensorTests",
      dependencies: ["Tensor"]),
    .testTarget(
      name: "TensorFlowTests",
      dependencies: ["TensorFlow"]),
  ]
)
