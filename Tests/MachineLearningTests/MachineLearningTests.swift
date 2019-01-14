import XCTest
@testable import MachineLearning

final class MachineLearningTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(MachineLearning().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
