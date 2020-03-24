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

import XCTest

@testable import Tensor

final class UtilitiesTests: XCTestCase {
  func testSHA1() {
    XCTAssertEqual(
      [UInt8](repeating: 0x61, count: 1000).sha1(),
      SIMD32<UInt8>(
        41, 30, 154, 108, 102, 153, 73, 73,
        181, 123, 165, 230, 80, 54, 30, 152,
        252, 54, 177, 186, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0))
  }

  func testSHA512() {
    XCTAssertEqual(
      [UInt8](repeating: 0x61, count: 1000).sha512(),
      SIMD64<UInt8>(
        103, 186, 85, 53, 164, 110, 63, 134, 219, 251, 237, 140, 187, 175, 1, 37,
        199, 110, 213, 73, 255, 139, 11, 158, 3, 224, 200, 140, 249, 15, 166, 52,
        250, 123, 18, 180, 125, 119, 182, 148, 222, 72, 138, 206, 141, 154, 101, 150,
        125, 201, 109, 245, 153, 114, 125, 50, 146, 168, 217, 212, 71, 112, 156, 151))
  }

  static var allTests = [
    ("testSHA1", testSHA1),
    ("testSHA512", testSHA512),
  ]
}
