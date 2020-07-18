// !!! THIS CODE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND !!!
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

import CTensorFlow

@inlinable @inline(__always)
func makeOp(_ name: String, _ nOutputs: Int) -> TFTensorOperation {
  _ExecutionContext.makeOp(name, nOutputs)
}

public struct _TFE_Op {
  @usableFromInline
  let outputCount: Int

  @usableFromInline
  init(_ name: String, _ outputCount: Int) {
    fatalError()
  }

  @inlinable @inline(__always)
  func execute<N: TensorFlowNumeric>() -> Tensor<N> {
    fatalError()
  }
}

public enum _RawTFEager {

  static let generatedTensorFlowVersion = "2.1.0"
  static let generatedTensorFlowGitVersion = "v2.1.0-rc2-17-ge5bf8de"

  // @_frozen // SR-9739
  public enum A {
    case apples
    case oranges

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum DataFormat {
    case nchw
    case nhwc

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum DataFormat1 {
    case ncdhw
    case ndhwc

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum DataFormat2 {
    case nchw
    case nchwVectC
    case nhwc

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum DensityUnit {
    case cm
    case in_

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum Direction {
    case bidirectional
    case unidirectional

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum Errors {
    case ignore
    case replace
    case strict

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum FinalOp {
    case div
    case id

    @inlinable
    var cName: String {
      fatalError()
    }
  }

  // @_frozen // SR-9739
  public enum Format {
    case empty
    case grayscale
    case rgb

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum InputMode {
    case autoSelect
    case linearInput
    case skipInput

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum InputQuantMode {
    case minFirst
    case scaled

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum LossType {
    case hingeLoss
    case logisticLoss
    case poissonLoss
    case smoothHingeLoss
    case squaredLoss

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum MergeOp {
    case add
    case max
    case min
    case mul

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum Method {
    case bilinear
    case nearest

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum Method1 {
    case bilinear

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum Mode {
    case minCombined
    case minFirst
    case scaled

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum Mode1 {
    case reflect
    case symmetric

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum OutputEncoding {
    case utf16Be
    case utf32Be
    case utf8

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum Padding {
    case same
    case valid

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum Padding1 {
    case explicit
    case same
    case valid

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum PrecisionMode {
    case fp16
    case fp32
    case int8

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum Reduction {
    case max
    case min
    case prod
    case sum

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        switch self {
        case .max: return "max"
        case .min: return "min"
        case .prod: return "prod"
        case .sum: return "sum"
        }
      }
    }
  }

  // @_frozen // SR-9739
  public enum ReductionType {
    case mean
    case sum

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum RnnMode {
    case gru
    case lstm
    case rnnRelu
    case rnnTanh

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        switch self {
        case .gru: return "gru"
        case .lstm: return "lstm"
        case .rnnRelu: return "rnn_relu"
        case .rnnTanh: return "rnn_tanh"
        }
      }
    }
  }

  // @_frozen // SR-9739
  public enum RoundMode {
    case halfToEven
    case halfUp

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum RoundMode1 {
    case halfAwayFromZero
    case halfToEven

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        switch self {
        case .halfAwayFromZero: return "HALF_AWAY_FROM_ZERO"
        case .halfToEven: return "HALF_TO_EVEN"
        }
      }
    }
  }

  // @_frozen // SR-9739
  public enum SplitType {
    case equality
    case inequality

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum SplitType1 {
    case inequality

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  // @_frozen // SR-9739
  public enum Unit {
    case byte
    case utf8Char

    @inlinable
    var cName: String {
      @inline(__always)
      get {
        fatalError()
      }
    }
  }

  @inlinable @inline(__always)
  public static func addV2<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    return _TFE_Op("AddV2", 1).execute()
  }
}
