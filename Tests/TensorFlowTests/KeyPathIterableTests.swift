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

import TensorFlow
import XCTest

struct SimpleKPI: KeyPathIterable, Equatable {
  var w, b: Float
}

struct MixedKPI: KeyPathIterable, Equatable {
  // Mutable.
  var string: String
  var float: Float
  // Immutable.
  let int: Int
}

struct NestedKPI: KeyPathIterable, Equatable {
  // Immutable.
  let simple: SimpleKPI
  // Mutable.
  var mixed: MixedKPI
}

struct ComplexNestedKPI: KeyPathIterable, Equatable {
  var float: Float
  let simple: SimpleKPI
  let optional: SimpleKPI?
  let array: [SimpleKPI]
  var dictionary: [String: SimpleKPI]
}

final class KeyPathIterableTests: XCTestCase {
  func testSimple() {
    var x = SimpleKPI(w: 1, b: 2)
    XCTAssertEqual([\SimpleKPI.w, \SimpleKPI.b], x.allKeyPaths)
    XCTAssertEqual([\SimpleKPI.w, \SimpleKPI.b], x.allKeyPaths(to: Float.self))
    XCTAssertEqual([\SimpleKPI.w, \SimpleKPI.b], x.allWritableKeyPaths(to: Float.self))
    XCTAssertEqual([\SimpleKPI.w, \SimpleKPI.b], x.recursivelyAllKeyPaths)
    XCTAssertEqual([\SimpleKPI.w, \SimpleKPI.b], x.recursivelyAllKeyPaths(to: Float.self))
    XCTAssertEqual([\SimpleKPI.w, \SimpleKPI.b], x.recursivelyAllWritableKeyPaths(to: Float.self))
    XCTAssertEqual([], x.allKeyPaths(to: Int.self))
    XCTAssertEqual([], x.recursivelyAllKeyPaths(to: Double.self))

    // Mutate recursively all `Float` properties.
    for kp in x.allWritableKeyPaths(to: Float.self) {
      x[keyPath: kp] += x[keyPath: kp]
    }
    // Check that recursively all `Float` properties have been mutated.
    XCTAssertEqual(SimpleKPI(w: 2, b: 4), x)
  }

  func testMixed() {
    var x = MixedKPI(string: "hello", float: .pi, int: 0)
    XCTAssertEqual([\MixedKPI.string, \MixedKPI.float, \MixedKPI.int], x.allKeyPaths)
    XCTAssertEqual([\MixedKPI.string, \MixedKPI.float, \MixedKPI.int], x.recursivelyAllKeyPaths)

    XCTAssertEqual([\MixedKPI.string], x.allKeyPaths(to: String.self))
    XCTAssertEqual([\MixedKPI.string], x.allWritableKeyPaths(to: String.self))
    XCTAssertEqual([\MixedKPI.string], x.recursivelyAllKeyPaths(to: String.self))
    XCTAssertEqual([\MixedKPI.string], x.recursivelyAllWritableKeyPaths(to: String.self))

    XCTAssertEqual([\MixedKPI.float], x.allKeyPaths(to: Float.self))
    XCTAssertEqual([\MixedKPI.float], x.allWritableKeyPaths(to: Float.self))
    XCTAssertEqual([\MixedKPI.float], x.recursivelyAllKeyPaths(to: Float.self))
    XCTAssertEqual([\MixedKPI.float], x.recursivelyAllWritableKeyPaths(to: Float.self))

    XCTAssertEqual([\MixedKPI.int], x.allKeyPaths(to: Int.self))
    XCTAssertEqual([], x.allWritableKeyPaths(to: Int.self))
    XCTAssertEqual([\MixedKPI.int], x.recursivelyAllKeyPaths(to: Int.self))
    XCTAssertEqual([], x.recursivelyAllWritableKeyPaths(to: Int.self))

    // Mutate recursively all `String` properties.
    for kp in x.allWritableKeyPaths(to: String.self) {
      x[keyPath: kp] = x[keyPath: kp] + " world"
    }
    // Check that recursively all `String` properties have been mutated.
    XCTAssertEqual(MixedKPI(string: "hello world", float: .pi, int: 0), x)
  }

  func testSimpleNested() {
    var x = NestedKPI(
      simple: SimpleKPI(w: 1, b: 2),
      mixed: MixedKPI(string: "foo", float: 3, int: 0))

    XCTAssertEqual([\NestedKPI.simple, \NestedKPI.mixed], x.allKeyPaths)
    XCTAssertEqual(
      [
        \NestedKPI.simple, \NestedKPI.simple.w, \NestedKPI.simple.b,
        \NestedKPI.mixed, \NestedKPI.mixed.string,
        \NestedKPI.mixed.float, \NestedKPI.mixed.int,
      ],
      x.recursivelyAllKeyPaths)

    XCTAssertEqual([], x.allKeyPaths(to: Float.self))
    XCTAssertEqual([], x.allKeyPaths(to: Int.self))
    XCTAssertEqual([], x.allKeyPaths(to: String.self))

    XCTAssertEqual([], x.allWritableKeyPaths(to: Float.self))
    XCTAssertEqual([], x.allWritableKeyPaths(to: Int.self))
    XCTAssertEqual([], x.allWritableKeyPaths(to: String.self))

    XCTAssertEqual(
      [\NestedKPI.simple.w, \NestedKPI.simple.b, \NestedKPI.mixed.float],
      x.recursivelyAllKeyPaths(to: Float.self))
    XCTAssertEqual([\NestedKPI.mixed.int], x.recursivelyAllKeyPaths(to: Int.self))
    XCTAssertEqual([\NestedKPI.mixed.string], x.recursivelyAllKeyPaths(to: String.self))

    XCTAssertEqual([\NestedKPI.mixed.float], x.recursivelyAllWritableKeyPaths(to: Float.self))
    XCTAssertEqual([], x.recursivelyAllWritableKeyPaths(to: Int.self))
    XCTAssertEqual([\NestedKPI.mixed.string], x.recursivelyAllWritableKeyPaths(to: String.self))

    XCTAssertEqual([], x.recursivelyAllKeyPaths(to: Double.self))

    // Mutate recursively all `Float` properties.
    for kp in x.recursivelyAllWritableKeyPaths(to: Float.self) {
      x[keyPath: kp] *= 100
    }
    // Check that recursively all `Float` properties have been mutated.
    let expected = NestedKPI(
      simple: SimpleKPI(w: 1, b: 2),
      mixed: MixedKPI(string: "foo", float: 300, int: 0))
    XCTAssertEqual(expected, x)
  }

  func testComplexNested() {
    var x = ComplexNestedKPI(
      float: 1, simple: SimpleKPI(w: 3, b: 4),
      optional: SimpleKPI(w: 5, b: 6),
      array: [SimpleKPI(w: 5, b: 6), SimpleKPI(w: 7, b: 8)],
      dictionary: [
        "foo": SimpleKPI(w: 1, b: 2),
        "bar": SimpleKPI(w: 3, b: 4),
      ])
    XCTAssertEqual(
      [
        \ComplexNestedKPI.float, \ComplexNestedKPI.simple,
        \ComplexNestedKPI.optional, \ComplexNestedKPI.array,
        \ComplexNestedKPI.dictionary,
      ],
      x.allKeyPaths)
    let key1 = (x.dictionary.keys.map {$0})[0]
    let key2 = (x.dictionary.keys.map {$0})[1]
    XCTAssertEqual(
      [
        \ComplexNestedKPI.float,
        \ComplexNestedKPI.simple,
        \ComplexNestedKPI.simple.w,
        \ComplexNestedKPI.simple.b,
        \ComplexNestedKPI.optional,
        \ComplexNestedKPI.optional!,
        \ComplexNestedKPI.optional!.w,
        \ComplexNestedKPI.optional!.b,
        \ComplexNestedKPI.array,
        \ComplexNestedKPI.array[0],
        \ComplexNestedKPI.array[0].w,
        \ComplexNestedKPI.array[0].b,
        \ComplexNestedKPI.array[1],
        \ComplexNestedKPI.array[1].w,
        \ComplexNestedKPI.array[1].b,
        \ComplexNestedKPI.dictionary,
        \ComplexNestedKPI.dictionary[key1]!,
        \ComplexNestedKPI.dictionary[key1]!.w,
        \ComplexNestedKPI.dictionary[key1]!.b,
        \ComplexNestedKPI.dictionary[key2]!,
        \ComplexNestedKPI.dictionary[key2]!.w,
        \ComplexNestedKPI.dictionary[key2]!.b,
      ],
      x.recursivelyAllKeyPaths)
    XCTAssertEqual(
      [
        \ComplexNestedKPI.float,
        \ComplexNestedKPI.simple.w,
        \ComplexNestedKPI.simple.b,
        \ComplexNestedKPI.optional!.w,
        \ComplexNestedKPI.optional!.b,
        \ComplexNestedKPI.array[0].w,
        \ComplexNestedKPI.array[0].b,
        \ComplexNestedKPI.array[1].w,
        \ComplexNestedKPI.array[1].b,
        \ComplexNestedKPI.dictionary[key1]!.w,
        \ComplexNestedKPI.dictionary[key1]!.b,
        \ComplexNestedKPI.dictionary[key2]!.w,
        \ComplexNestedKPI.dictionary[key2]!.b,
      ],
      x.recursivelyAllKeyPaths(to: Float.self))
    XCTAssertEqual(
      [
        \ComplexNestedKPI.float,
        \ComplexNestedKPI.dictionary[key1]!.w,
        \ComplexNestedKPI.dictionary[key1]!.b,
        \ComplexNestedKPI.dictionary[key2]!.w,
        \ComplexNestedKPI.dictionary[key2]!.b,
      ],
      x.recursivelyAllWritableKeyPaths(to: Float.self))

    // Mutate recursively all `Float` properties.
    for kp in x.recursivelyAllWritableKeyPaths(to: Float.self) {
      x[keyPath: kp] += 1
    }
    // Check that recursively all `Float` properties have been mutated.
    let expected = ComplexNestedKPI(
      float: 2, simple: SimpleKPI(w: 3, b: 4),
      optional: SimpleKPI(w: 5, b: 6),
      array: [SimpleKPI(w: 5, b: 6), SimpleKPI(w: 7, b: 8)],
      dictionary: [
        "foo": SimpleKPI(w: 2, b: 3),
        "bar": SimpleKPI(w: 4, b: 5),
      ])
    XCTAssertEqual(expected, x)
  }

  static var allTests = [
    ("testSimple", testSimple),
    ("testMixed", testMixed),
    ("testSimpleNested", testSimpleNested),
    ("testComplexNested", testComplexNested),
  ]
}
