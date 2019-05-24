// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
//
// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
    name: "DeepLearning",
    products: [
        .library(
            name: "DeepLearning",
            targets: ["DeepLearning"]),
        .library(
            name: "third_party",
            targets: ["third_party"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "DeepLearning",
            dependencies: []),
        .target(
            name: "third_party",
            dependencies: [],
            path: "Sources/third_party"),
        .testTarget(
            name: "DeepLearningTests",
            dependencies: ["DeepLearning", "third_party"]),
    ]
)
