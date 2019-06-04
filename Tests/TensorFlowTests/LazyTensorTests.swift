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
@testable import TensorFlow
import CTensorFlow

final class LazyTensorTests: XCTestCase {
    func testConstructions() {
        let zero = Tensor<Float>(0.0)
        let zeroTFEHandle = zero.handle.handle._tfeTensorHandle

        let concTensor = LazyTensor(zeroTFEHandle)
        XCTAssertEqual("\(concTensor)", "conc")

        let materializedConcTensor = LazyTensor(
            _materialized: zeroTFEHandle)
        XCTAssertEqual("\(materializedConcTensor)", "conc*")

        let op = LazyTensorOperation(_withID: "0", "IdentityN", 3)
        let symTensor0 = LazyTensor(_lazy: op, index: 0)
        XCTAssertEqual("\(symTensor0)", "IdentityN_0:0")

        let symTensor1 = LazyTensor(_lazy: op, index: 2)
        XCTAssertEqual("\(symTensor1)", "IdentityN_0:2")

        let liveSymTensor = LazyTensor(_lazyLive: op, index: 0)
        XCTAssertEqual("\(liveSymTensor)", "IdentityN_0:0*")
    }

    static var allTests = [
        ("testConstructions", testConstructions),
    ]

}
