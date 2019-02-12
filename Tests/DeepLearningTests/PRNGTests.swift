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
@testable import DeepLearning

final class ThreefryTests: XCTestCase {
    func testThreefry() {
        // Check the PRNG output given different seeds against reference values.
        // This guards against accidental changes to the generator which should
        // result in changes to its output.
        var generator = ThreefryRandomNumberGenerator(uint64seed: 123159812)
        XCTAssertEqual(generator.next(), 9411952874433594703)
        XCTAssertEqual(generator.next(), 6992100569504761807)
        XCTAssertEqual(generator.next(), 6249442510280393663)
        XCTAssertEqual(generator.next(), 13096801615464354606)
        XCTAssertEqual(generator.next(), 671569830597624217)
        XCTAssertEqual(generator.next(), 9499516613746162591)
        XCTAssertEqual(generator.next(), 14104268727839198528)
        XCTAssertEqual(generator.next(), 2729105059420396781)

        generator = ThreefryRandomNumberGenerator(uint64seed: 58172950819076)
        XCTAssertEqual(generator.next(), 8181320043134006362)
        XCTAssertEqual(generator.next(), 14375459274817572790)
        XCTAssertEqual(generator.next(), 1051151592956420496)
        XCTAssertEqual(generator.next(), 12482694246229339388)
        XCTAssertEqual(generator.next(), 2543901658316819773)
        XCTAssertEqual(generator.next(), 54584659268457468)
        XCTAssertEqual(generator.next(), 4068621515934625604)
        XCTAssertEqual(generator.next(), 10604176710283101491)
    }
}
