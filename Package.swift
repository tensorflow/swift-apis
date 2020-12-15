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

import Dispatch
import Foundation
#if !os(macOS)
import FoundationNetworking
#endif

func extract(_ url: URL, _ cache: URL) throws {
  let process: Process = Process()
#if os(Windows)
  process.executableURL =
      URL(fileURLWithPath: "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe")

  let path = url.withUnsafeFileSystemRepresentation {
    String(cString: $0!)
  }
  process.arguments = [
    "-Command",
    "Expand-Archive -Force \"\(path)\" \"\(cache.path)\""
  ]
#else
  process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
  process.arguments = [
    "-qq",
    url.withUnsafeFileSystemRepresentation { String(cString: $0!) },
    "-d", cache.path
  ]
#endif
  try process.run()
  process.waitUntilExit()
}

let cache =
  try! FileManager.default.url(for: .cachesDirectory, in: .userDomainMask,
                               appropriateFor: nil, create: true)

var localURL = cache
#if os(Windows)
  localURL.appendPathComponent("tensorflow-windows-x64.zip")
  let resourceURL: URL = URL(string: "https://artprodeus21.artifacts.visualstudio.com/A8fd008a0-56bc-482c-ba46-67f9425510be/3133d6ab-80a8-4996-ac4f-03df25cd3224/_apis/artifact/cGlwZWxpbmVhcnRpZmFjdDovL2NvbXBuZXJkL3Byb2plY3RJZC8zMTMzZDZhYi04MGE4LTQ5OTYtYWM0Zi0wM2RmMjVjZDMyMjQvYnVpbGRJZC80Mzc2OC9hcnRpZmFjdE5hbWUvdGVuc29yZmxvdy13aW5kb3dzLXg2NA2/content?format=zip")!
#elseif os(macOS)
  localURL.appendPathComponent("tensorflow-darwin-x64.zip")
  let resourceURL: URL = URL(string: "https://artprodeus21.artifacts.visualstudio.com/A8fd008a0-56bc-482c-ba46-67f9425510be/3133d6ab-80a8-4996-ac4f-03df25cd3224/_apis/artifact/cGlwZWxpbmVhcnRpZmFjdDovL2NvbXBuZXJkL3Byb2plY3RJZC8zMTMzZDZhYi04MGE4LTQ5OTYtYWM0Zi0wM2RmMjVjZDMyMjQvYnVpbGRJZC80Mzc2OC9hcnRpZmFjdE5hbWUvdGVuc29yZmxvdy1kYXJ3aW4teDY00/content?format=zip")!
#else
print("Linux (and other OSes) are not supported")
#endif

private func download(_ url: URL, to location: URL) throws {
  let session: URLSession =
      URLSession(configuration: .default, delegate: nil, delegateQueue: nil)

  let semaphore: DispatchSemaphore = DispatchSemaphore(value: 0)
  let task: URLSessionTask = session.downloadTask(with: url) { url, response, error in
    defer { semaphore.signal() }

    guard let url = url else { return }
    try? FileManager.default.moveItem(at: url, to: location)
  }
  task.resume()
  semaphore.wait()
}

try! download(resourceURL, to: localURL)
try! extract(localURL, cache)

private func X10Flags(_ root: URL)
    -> (HeaderSearchPath: String, LinkerSearchPath: String) {
  return root.withUnsafeFileSystemRepresentation {
    let root: String = String(cString: $0!)
#if os(Windows)
    return (HeaderSearchPath: "\(root)\\tensorflow-windows-x64\\Library\\tensorflow-2.3.0\\usr\\include",
            LinkerSearchPath: "\(root)\\tensorflow-windows-x64\\Library\\tensorflow-2.3.0\\usr\\lib")
#else
    return (HeaderSearchPath: "\(root)/tensorflow-darwin-x64/Library/tensorflow-2.3.0/usr/include",
            LinkerSearchPath: "\(root)/tensorflow-darwin-x64/Library/tensorflow-2.3.0/usr/lib")
#endif
  }
}

let X10 = X10Flags(cache)
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
  dependencies: [
    .package(url: "https://github.com/apple/swift-numerics", .branch("main")),
    .package(url: "https://github.com/pvieito/PythonKit.git", .branch("master")),
  ],
  targets: [
    .target(
      name: "Tensor",
      dependencies: []),
    .target(
      name: "CTensorFlow",
      dependencies: [],
      cSettings: [
        .unsafeFlags([
          "-I\(X10.HeaderSearchPath)",
        ])
      ],
      linkerSettings: [
        .unsafeFlags([
          "-L\(X10.LinkerSearchPath)"
        ]),
      ]),
    .target(
      name: "TensorFlow",
      dependencies: [
        "Tensor",
        "PythonKit",
        "CTensorFlow",
        .product(name: "Numerics", package: "swift-numerics"),
      ]),
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
