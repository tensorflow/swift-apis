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

import TensorFlow
import XCTest

class ImageTests: XCTestCase {
  func testResizeArea() {
    let image = Tensor<Float>(rangeFrom: 0, to: 16 * 16 * 3, stride: 1)
      .reshaped(to: [1, 16, 16, 3])
    let resized = resizeArea(images: image, size: (2, 2))

    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(1*16*16*3), [1, 16, 16, 3])
    // y = tf.image.resize(x, [2, 2], "area")
    // print(y)
    // ```

    XCTAssertEqual(
      resized,
      [
        [
          [[178.5, 179.5, 180.5], [202.5, 203.5, 204.5]],
          [[562.5, 563.5, 564.5], [586.5, 587.5, 588.5]],
        ]
      ])
  }

  func testResizeNearest() {
    let image = Tensor<Float>(rangeFrom: 0, to: 16 * 16 * 3, stride: 1)
      .reshaped(to: [1, 16, 16, 3])
    let resized = resize(images: image, size: (2, 2), method: .nearest)

    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(1*16*16*3), [1, 16, 16, 3])
    // y = tf.image.resize(x, [2, 2], "nearest")
    // print(y)
    // ```

    XCTAssertEqual(
      resized,
      [
        [
          [[204, 205, 206], [228, 229, 230]],
          [[588, 589, 590], [612, 613, 614]],
        ]
      ])
  }

  func testResizeBilinear() {
    let image = Tensor<Float>(rangeFrom: 0, to: 16 * 16 * 3, stride: 1)
      .reshaped(to: [1, 16, 16, 3])
    let resized = resize(images: image, size: (2, 2), method: .bilinear)

    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(1*16*16*3), [1, 16, 16, 3])
    // y = tf.image.resize(x, [2, 2], "bilinear")
    // print(y)
    // ```

    XCTAssertEqual(
      resized,
      [
        [
          [[178.5, 179.5, 180.5], [202.5, 203.5, 204.5]],
          [[562.5, 563.5, 564.5], [586.5, 587.5, 588.5]],
        ]
      ])
  }

  func testResizeBilinearAntialias() {
    let image = Tensor<Float>(rangeFrom: 0, to: 16 * 16 * 3, stride: 1)
      .reshaped(to: [1, 16, 16, 3])
    let resized = resize(images: image, size: (2, 2), method: .bilinear, antialias: true)

    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(1*16*16*3), [1, 16, 16, 3])
    // y = tf.image.resize(x, [2, 2], "bilinear", antialias=True)
    // print(y)
    // ```

    XCTAssertEqual(
      resized,
      [
        [
          [
            [217.66074, 218.66075, 219.66074],
            [237.0536, 238.05359, 239.0536],
          ],
          [
            [527.9465, 528.9465, 529.9465],
            [547.33936, 548.33936, 549.33936],
          ],
        ]
      ])
  }

  func testResizeBicubic() {
    let image = Tensor<Float>(rangeFrom: 0, to: 16 * 16 * 3, stride: 1)
      .reshaped(to: [1, 16, 16, 3])
    let resized = resize(images: image, size: (2, 2), method: .bicubic)

    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(1*16*16*3), [1, 16, 16, 3])
    // y = tf.image.resize(x, [2, 2], "bicubic")
    // print(y)
    // ```

    XCTAssertEqual(
      resized,
      [
        [
          [[178.5, 179.5, 180.5], [202.5, 203.5, 204.5]],
          [[562.5, 563.5, 564.5], [586.5, 587.5, 588.5]],
        ]
      ])
  }

  func testResizeBicubicAntialias() {
    let image = Tensor<Float>(rangeFrom: 0, to: 16 * 16 * 3, stride: 1)
      .reshaped(to: [1, 16, 16, 3])
    let resized = resize(images: image, size: (2, 2), method: .bicubic, antialias: true)

    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(1*16*16*3), [1, 16, 16, 3])
    // y = tf.image.resize(x, [2, 2], "bicubic", antialias=True)
    // print(y)
    // ```

    XCTAssertEqual(
      resized,
      [
        [
          [
            [197.1046, 198.10457, 199.10461],
            [218.9158, 219.9158, 220.91579],
          ],
          [
            [546.0843, 547.0842, 548.08417],
            [567.89545, 568.8954, 569.8954],
          ],
        ]
      ])
  }

  func testResizeLanczos3() {
    let image = Tensor<Float>(rangeFrom: 0, to: 16 * 16 * 3, stride: 1)
      .reshaped(to: [1, 16, 16, 3])
    let resized = resize(images: image, size: (2, 2), method: .lanczos3)

    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(1*16*16*3), [1, 16, 16, 3])
    // y = tf.image.resize(x, [2, 2], "lanczos3")
    // print(y)
    // ```

    XCTAssertEqual(
      resized,
      [
        [
          [
            [178.50002, 179.5, 180.50002],
            [202.5, 203.50002, 204.50002],
          ],
          [
            [562.50006, 563.49994, 564.5],
            [586.49994, 587.50006, 588.50006],
          ],
        ]
      ])
  }

  func testResizeLanczos5() {
    let image = Tensor<Float>(rangeFrom: 0, to: 16 * 16 * 3, stride: 1)
      .reshaped(to: [1, 16, 16, 3])
    let resized = resize(images: image, size: (2, 2), method: .lanczos5)

    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(1*16*16*3), [1, 16, 16, 3])
    // y = tf.image.resize(x, [2, 2], "lanczos5")
    // print(y)
    // ```

    XCTAssertEqual(
      resized,
      [
        [
          [
            [180.29031, 181.29033, 182.29033],
            [204.07971, 205.07973, 206.07973],
          ],
          [
            [560.9203, 561.92035, 562.9203],
            [584.7098, 585.7098, 586.7097],
          ],
        ]
      ])
  }

  func testResizeGaussian() {
    let image = Tensor<Float>(rangeFrom: 0, to: 16 * 16 * 3, stride: 1)
      .reshaped(to: [1, 16, 16, 3])
    let resized = resize(images: image, size: (2, 2), method: .gaussian)

    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(1*16*16*3), [1, 16, 16, 3])
    // y = tf.image.resize(x, [2, 2], "gaussian")
    // print(y)
    // ```

    XCTAssertEqual(
      resized,
      [
        [
          [
            [178.5, 179.5, 180.5],
            [202.5, 203.5, 204.5],
          ],
          [
            [562.5, 563.5, 564.5],
            [586.5, 587.5, 588.5],
          ],
        ]
      ])
  }

  func testResizeMitchellcubic() {
    let image = Tensor<Float>(rangeFrom: 0, to: 16 * 16 * 3, stride: 1)
      .reshaped(to: [1, 16, 16, 3])
    let resized = resize(images: image, size: (2, 2), method: .mitchellcubic)

    // The expected value was computed using the following Python code:
    // ```
    // import tensorflow as tf
    // x = tf.reshape(tf.range(1*16*16*3), [1, 16, 16, 3])
    // y = tf.image.resize(x, [2, 2], "mitchellcubic")
    // print(y)
    // ```

    XCTAssertEqual(
      resized,
      [
        [
          [
            [178.50002, 179.5, 180.5],
            [202.5, 203.5, 204.50002],
          ],
          [
            [562.5, 563.5, 564.5],
            [586.5, 587.50006, 588.5],
          ],
        ]
      ])
  }

  static let allTests = [
    ("testResizeArea", testResizeArea),
    ("testResizeNearest", testResizeNearest),
    ("testResizeBilinear", testResizeBilinear),
    ("testResizeBilinearAntialias", testResizeBilinearAntialias),
    ("testResizeBicubic", testResizeBicubic),
    ("testResizeBicubicAntialias", testResizeBicubicAntialias),
    ("testResizeLanczos3", testResizeLanczos3),
    ("testResizeLanczos5", testResizeLanczos5),
    ("testResizeGaussian", testResizeGaussian),
    ("testResizeMitchellcubic", testResizeMitchellcubic),
  ]
}
