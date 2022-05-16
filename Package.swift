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
      name: "x10_optimizers_optimizer",
      type: .dynamic,
      targets: ["x10_optimizers_optimizer"]),
    .library(
      name: "x10_optimizers_tensor_visitor_plan",
      type: .dynamic,
      targets: ["x10_optimizers_tensor_visitor_plan"]),
    // .library(
    //   name: "x10_training_loop",
    //   type: .dynamic,
    //   targets: ["x10_training_loop"]),
  ],
  dependencies: [
    .package(url: "https://github.com/apple/swift-numerics", .branch("main")),
    .package(url: "https://github.com/pvieito/PythonKit.git", .branch("master")),
  ],
  targets: [
    .target(
      name: "CTensorFlow",
      dependencies: []),
    .target(
      name: "CX10Modules",
      dependencies: []),
    .target(
      name: "TensorFlow",
      dependencies: [
        "PythonKit",
        "CTensorFlow",
        "CX10Modules",
        .product(name: "Numerics", package: "swift-numerics"),
      ],
      swiftSettings: [
        .define("DEFAULT_BACKEND_EAGER"),
      ]),
    .target(
      name: "x10_optimizers_tensor_visitor_plan",
      dependencies: ["TensorFlow"],
      path: "Sources/x10",
      sources: [
        "swift_bindings/optimizers/TensorVisitorPlan.swift",
      ]),
    .target(
      name: "x10_optimizers_optimizer",
      dependencies: [
        "x10_optimizers_tensor_visitor_plan",
        "TensorFlow",
      ],
      path: "Sources/x10",
      sources: [
        "swift_bindings/optimizers/Optimizer.swift",
        "swift_bindings/optimizers/Optimizers.swift",
      ]),
    // .target(
    //   name: "x10_training_loop",
    //   dependencies: ["TensorFlow"],
    //   path: "Sources/x10",
    //   sources: [
    //     "swift_bindings/training_loop.swift",
    //   ]),
    .target(
      name: "Experimental",
      dependencies: [],
      path: "Sources/third_party/Experimental"),
    .testTarget(
      name: "ExperimentalTests",
      dependencies: ["Experimental"]),
    .testTarget(
      name: "TensorFlowTests",
      dependencies: ["TensorFlow"]),
  ]
)
