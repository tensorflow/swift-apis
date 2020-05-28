/// Test `KeyPathIterable` extensions.

import TensorFlow
import XCTest

extension KeyPathIterable {
  func recursivelyAllProperties<T>(withType type: T.Type) -> [T] {
    return recursivelyAllKeyPaths(to: type).map { self[keyPath: $0] }
  }

  /// Asserts that recursively all properties with the given type satisfy the given condition.
  func assertRecursivelyAllProperties<T>(
    withType type: T.Type, _ condition: (T) -> Bool, file: StaticString = #file, line: UInt = #line
  ) {
    for property in recursivelyAllProperties(withType: type) {
      XCTAssert(condition(property), file: file, line: line)
    }
  }
}

// Dummy `KeyPathIterable`-conforming type with nested properties/elements.
struct Wrapper<T>: KeyPathIterable {
  // Top-level property.
  var item: T

  // Nested properties.
  var array: [T] = []

  var dictionary: [String: T] = [:]
}

extension Wrapper where T == Tensor<Float> {
  static var fullPrecisionExample: Wrapper {
    let scalars = (0..<10).map(Float.init)
    let tensors = scalars.map { Tensor($0, on: Device.defaultXLA) }
    return Wrapper<Tensor<Float>>(
      item: Tensor(0, on: Device.defaultXLA),
      array: tensors,
      // FIXME: `Dictionary.allKeyPaths` does not return `WritableKeyPaths`.
      // Fixed in https://github.com/apple/swift/pull/29066.
      // dictionary: Dictionary(uniqueKeysWithValues: tensors.map { (String(describing: $0), $0) }))
      dictionary: [:])
  }

  static var reducedPrecisionExample: Wrapper {
    return fullPrecisionExample.toReducedPrecision
  }
}

final class KeyPathIterableTests: XCTestCase {
  func testConvertToReducedPrecision() throws {
    var example = Wrapper<Tensor<Float>>.fullPrecisionExample
    example.assertRecursivelyAllProperties(withType: Tensor<Float>.self, { !$0.isReducedPrecision })
    example.convertToReducedPrecision()
    example.assertRecursivelyAllProperties(withType: Tensor<Float>.self, { $0.isReducedPrecision })
  }

  func testConvertToFullPrecision() throws {
    var example = Wrapper<Tensor<Float>>.reducedPrecisionExample
    example.assertRecursivelyAllProperties(withType: Tensor<Float>.self, { $0.isReducedPrecision })
    example.convertToFullPrecision()
    example.assertRecursivelyAllProperties(withType: Tensor<Float>.self, { !$0.isReducedPrecision })
  }

  func testToReducedPrecision() throws {
    let example = Wrapper<Tensor<Float>>.fullPrecisionExample
    example.assertRecursivelyAllProperties(withType: Tensor<Float>.self, { !$0.isReducedPrecision })
    let result = example.toReducedPrecision
    result.assertRecursivelyAllProperties(withType: Tensor<Float>.self, { $0.isReducedPrecision })
  }

  func testToFullPrecision() throws {
    let example = Wrapper<Tensor<Float>>.reducedPrecisionExample
    example.assertRecursivelyAllProperties(withType: Tensor<Float>.self, { $0.isReducedPrecision })
    let result = example.toFullPrecision
    result.assertRecursivelyAllProperties(withType: Tensor<Float>.self, { !$0.isReducedPrecision })
  }

  func testMoveToDevice() throws {
    // Testing is possible only when there are multiple devices.
    // Skip test if only one device is available.
    guard let otherDevice = Device.allDevices.first(where: { $0 != Device.default }) else {
      return
    }

    var example = Wrapper<Tensor<Float>>.reducedPrecisionExample
    example.assertRecursivelyAllProperties(
      withType: Tensor<Float>.self, { $0.device == Device.default })
    example.move(to: otherDevice)
    example.assertRecursivelyAllProperties(
      withType: Tensor<Float>.self, { $0.device == otherDevice })
  }

  func testCopyingToDevice() throws {
    // Testing is possible only when there are multiple devices.
    // Skip test if only one device is available.
    guard let otherDevice = Device.allDevices.first(where: { $0 != Device.default }) else {
      return
    }

    let example = Wrapper<Tensor<Float>>.reducedPrecisionExample
    example.assertRecursivelyAllProperties(
      withType: Tensor<Float>.self, { $0.device == Device.default })
    let copiedExample = Wrapper<Tensor<Float>>(copying: example, to: otherDevice)
    copiedExample.assertRecursivelyAllProperties(
      withType: Tensor<Float>.self, { $0.device == otherDevice })
  }
}

extension KeyPathIterableTests {
  static var allTests = [
    ("testConvertToReducedPrecision", testConvertToReducedPrecision),
    ("testConvertToFullPrecision", testConvertToFullPrecision),
    ("testToReducedPrecision", testToReducedPrecision),
    ("testToFullPrecision", testToFullPrecision),
    ("testMoveToDevice", testMoveToDevice),
    ("testCopyingToDevice", testCopyingToDevice),
  ]
}

XCTMain([
  testCase(KeyPathIterableTests.allTests)
])
