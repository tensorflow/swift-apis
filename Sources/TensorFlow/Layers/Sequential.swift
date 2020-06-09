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

import _Differentiation

/// A layer that sequentially composes two or more other layers.
///
/// ### Examples: ###
///
/// - Build a simple 2-layer perceptron model for MNIST:
///
/// ````
/// let inputSize = 28 * 28
/// let hiddenSize = 300
/// var classifier = Sequential {
///      Dense<Float>(inputSize: inputSize, outputSize: hiddenSize, activation: relu)
///      Dense<Float>(inputSize: hiddenSize, outputSize: 3, activation: identity)
///  }
/// ````
///
/// - Build an autoencoder for MNIST:
///
/// ````
/// var autoencoder = Sequential {
///     // The encoder.
///     Dense<Float>(inputSize: 28 * 28, outputSize: 128, activation: relu)
///     Dense<Float>(inputSize: 128, outputSize: 64, activation: relu)
///     Dense<Float>(inputSize: 64, outputSize: 12, activation: relu)
///     Dense<Float>(inputSize: 12, outputSize: 3, activation: relu)
///     // The decoder.
///     Dense<Float>(inputSize: 3, outputSize: 12, activation: relu)
///     Dense<Float>(inputSize: 12, outputSize: 64, activation: relu)
///     Dense<Float>(inputSize: 64, outputSize: 128, activation: relu)
///     Dense<Float>(inputSize: 128, outputSize: imageHeight * imageWidth, activation: tanh)
/// }
/// ````
public struct Sequential<Layer1: Module, Layer2: Layer>: Module
where
  Layer1.Output == Layer2.Input,
  Layer1.TangentVector.VectorSpaceScalar == Layer2.TangentVector.VectorSpaceScalar
{
  public var layer1: Layer1
  public var layer2: Layer2

  public init(_ layer1: Layer1, _ layer2: Layer2) {
    self.layer1 = layer1
    self.layer2 = layer2
  }

  @differentiable(wrt: self)
  public func callAsFunction(_ input: Layer1.Input) -> Layer2.Output {
    layer2(layer1(input))
  }

  public init(@LayerBuilder layers: () -> Self) {
    self = layers()
  }
}

extension Sequential: Layer where Layer1: Layer {
  @differentiable
  public func callAsFunction(_ input: Layer1.Input) -> Layer2.Output {
    layer2(layer1(input))
  }
}

/// A layer that sequentially composes 3 layers.
public typealias Sequential3<L1: Module, L2: Layer, L3: Layer> = Sequential<L1, Sequential<L2, L3>>
where
  L1.Output == L2.Input, L2.Output == L3.Input,
  L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
  L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar

/// A layer that sequentially composes 4 layers.
public typealias Sequential4<L1: Module, L2: Layer, L3: Layer, L4: Layer> = Sequential<
  L1, Sequential<L2, Sequential<L3, L4>>
>
where
  L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input,
  L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
  L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
  L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar

/// A layer that sequentially composes 5 layers.
public typealias Sequential5<L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer> = Sequential<
  L1, Sequential<L2, Sequential<L3, Sequential<L4, L5>>>
>
where
  L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input,
  L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
  L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
  L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
  L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar

/// A layer that sequentially composes 6 layers.
public typealias Sequential6<L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer> =
  Sequential<L1, Sequential<L2, Sequential<L3, Sequential<L4, Sequential<L5, L6>>>>>
where
  L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input,
  L5.Output == L6.Input,
  L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
  L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
  L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
  L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar,
  L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar

/// A layer that sequentially composes 7 layers.
public typealias Sequential7<
  L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer, L7: Layer
> = Sequential<
  L1, Sequential<L2, Sequential<L3, Sequential<L4, Sequential<L5, Sequential<L6, L7>>>>>
>
where
  L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input,
  L5.Output == L6.Input, L6.Output == L7.Input,
  L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
  L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
  L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
  L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar,
  L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar,
  L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar

/// A layer that sequentially composes 8 layers.
public typealias Sequential8<
  L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer, L7: Layer, L8: Layer
> = Sequential<
  L1,
  Sequential<L2, Sequential<L3, Sequential<L4, Sequential<L5, Sequential<L6, Sequential<L7, L8>>>>>>
>
where
  L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input,
  L5.Output == L6.Input, L6.Output == L7.Input, L7.Output == L8.Input,
  L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
  L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
  L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
  L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar,
  L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar,
  L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar,
  L7.TangentVector.VectorSpaceScalar == L8.TangentVector.VectorSpaceScalar

/// A layer that sequentially composes 9 layers.
public typealias Sequential9<
  L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer, L7: Layer, L8: Layer, L9: Layer
> = Sequential<
  L1,
  Sequential<
    L2,
    Sequential<
      L3, Sequential<L4, Sequential<L5, Sequential<L6, Sequential<L7, Sequential<L8, L9>>>>>
    >
  >
>
where
  L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input,
  L5.Output == L6.Input, L6.Output == L7.Input, L7.Output == L8.Input, L8.Output == L9.Input,
  L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
  L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
  L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
  L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar,
  L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar,
  L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar,
  L7.TangentVector.VectorSpaceScalar == L8.TangentVector.VectorSpaceScalar,
  L8.TangentVector.VectorSpaceScalar == L9.TangentVector.VectorSpaceScalar

/// A layer that sequentially composes 10 layers.
public typealias Sequential10<
  L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer, L7: Layer, L8: Layer,
  L9: Layer, L10: Layer
> = Sequential<
  L1,
  Sequential<
    L2,
    Sequential<
      L3,
      Sequential<
        L4, Sequential<L5, Sequential<L6, Sequential<L7, Sequential<L8, Sequential<L9, L10>>>>>
      >
    >
  >
>
where
  L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input,
  L5.Output == L6.Input, L6.Output == L7.Input, L7.Output == L8.Input, L8.Output == L9.Input,
  L9.Output == L10.Input,
  L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
  L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
  L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
  L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar,
  L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar,
  L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar,
  L7.TangentVector.VectorSpaceScalar == L8.TangentVector.VectorSpaceScalar,
  L8.TangentVector.VectorSpaceScalar == L9.TangentVector.VectorSpaceScalar,
  L9.TangentVector.VectorSpaceScalar == L10.TangentVector.VectorSpaceScalar

@_functionBuilder
public struct LayerBuilder {
  public static func buildBlock<L1: Module, L2: Layer>(_ l1: L1, _ l2: L2) -> Sequential<L1, L2>
  where L1.Output == L2.Input {
    Sequential(l1, l2)
  }

  public static func buildBlock<
    L1: Module,
    L2: Layer,
    L3: Layer
  >(_ l1: L1, _ l2: L2, _ l3: L3)
    -> Sequential<L1, Sequential<L2, L3>>
  where
    L1.Output == L2.Input,
    L2.Output == L3.Input,
    L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
    L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar
  {
    Sequential(l1, Sequential(l2, l3))
  }

  public static func buildBlock<
    L1: Module,
    L2: Layer,
    L3: Layer,
    L4: Layer
  >(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4)
    -> Sequential<L1, Sequential<L2, Sequential<L3, L4>>>
  where
    L1.Output == L2.Input,
    L2.Output == L3.Input,
    L3.Output == L4.Input,
    L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
    L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
    L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar
  {
    Sequential(l1, Sequential(l2, Sequential(l3, l4)))
  }

  public static func buildBlock<
    L1: Module,
    L2: Layer,
    L3: Layer,
    L4: Layer,
    L5: Layer
  >(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5)
    -> Sequential<L1, Sequential<L2, Sequential<L3, Sequential<L4, L5>>>>
  where
    L1.Output == L2.Input,
    L2.Output == L3.Input,
    L3.Output == L4.Input,
    L4.Output == L5.Input,
    L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
    L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
    L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
    L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar
  {
    Sequential(l1, Sequential(l2, Sequential(l3, Sequential(l4, l5))))
  }

  public static func buildBlock<
    L1: Module,
    L2: Layer,
    L3: Layer,
    L4: Layer,
    L5: Layer,
    L6: Layer
  >(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6)
    -> Sequential<L1, Sequential<L2, Sequential<L3, Sequential<L4, Sequential<L5, L6>>>>>
  where
    L1.Output == L2.Input,
    L2.Output == L3.Input,
    L3.Output == L4.Input,
    L4.Output == L5.Input,
    L5.Output == L6.Input,
    L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
    L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
    L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
    L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar,
    L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar
  {
    Sequential(l1, Sequential(l2, Sequential(l3, Sequential(l4, Sequential(l5, l6)))))
  }

  public static func buildBlock<
    L1: Module,
    L2: Layer,
    L3: Layer,
    L4: Layer,
    L5: Layer,
    L6: Layer,
    L7: Layer
  >(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6, _ l7: L7)
    -> Sequential<
      L1, Sequential<L2, Sequential<L3, Sequential<L4, Sequential<L5, Sequential<L6, L7>>>>>
    >
  where
    L1.Output == L2.Input,
    L2.Output == L3.Input,
    L3.Output == L4.Input,
    L4.Output == L5.Input,
    L5.Output == L6.Input,
    L6.Output == L7.Input,
    L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
    L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
    L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
    L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar,
    L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar,
    L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar
  {
    Sequential(
      l1, Sequential(l2, Sequential(l3, Sequential(l4, Sequential(l5, Sequential(l6, l7))))))
  }

  public static func buildBlock<
    L1: Module,
    L2: Layer,
    L3: Layer,
    L4: Layer,
    L5: Layer,
    L6: Layer,
    L7: Layer,
    L8: Layer
  >(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6, _ l7: L7, _ l8: L8)
    -> Sequential<
      L1,
      Sequential<
        L2, Sequential<L3, Sequential<L4, Sequential<L5, Sequential<L6, Sequential<L7, L8>>>>>
      >
    >
  where
    L1.Output == L2.Input,
    L2.Output == L3.Input,
    L3.Output == L4.Input,
    L4.Output == L5.Input,
    L5.Output == L6.Input,
    L6.Output == L7.Input,
    L7.Output == L8.Input,
    L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
    L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
    L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
    L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar,
    L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar,
    L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar,
    L7.TangentVector.VectorSpaceScalar == L8.TangentVector.VectorSpaceScalar
  {
    Sequential(
      l1,
      Sequential(
        l2, Sequential(l3, Sequential(l4, Sequential(l5, Sequential(l6, Sequential(l7, l8)))))))
  }

  public static func buildBlock<
    L1: Module,
    L2: Layer,
    L3: Layer,
    L4: Layer,
    L5: Layer,
    L6: Layer,
    L7: Layer,
    L8: Layer,
    L9: Layer
  >(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6, _ l7: L7, _ l8: L8, _ l9: L9)
    -> Sequential<
      L1,
      Sequential<
        L2,
        Sequential<
          L3, Sequential<L4, Sequential<L5, Sequential<L6, Sequential<L7, Sequential<L8, L9>>>>>
        >
      >
    >
  where
    L1.Output == L2.Input,
    L2.Output == L3.Input,
    L3.Output == L4.Input,
    L4.Output == L5.Input,
    L5.Output == L6.Input,
    L6.Output == L7.Input,
    L7.Output == L8.Input,
    L8.Output == L9.Input,
    L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
    L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
    L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
    L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar,
    L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar,
    L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar,
    L7.TangentVector.VectorSpaceScalar == L8.TangentVector.VectorSpaceScalar,
    L8.TangentVector.VectorSpaceScalar == L9.TangentVector.VectorSpaceScalar
  {
    Sequential(
      l1,
      Sequential(
        l2,
        Sequential(
          l3, Sequential(l4, Sequential(l5, Sequential(l6, Sequential(l7, Sequential(l8, l9))))))))
  }

  public static func buildBlock<
    L1: Module,
    L2: Layer,
    L3: Layer,
    L4: Layer,
    L5: Layer,
    L6: Layer,
    L7: Layer,
    L8: Layer,
    L9: Layer,
    L10: Layer
  >(
    _ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6, _ l7: L7, _ l8: L8, _ l9: L9,
    _ l10: L10
  )
    -> Sequential<
      L1,
      Sequential<
        L2,
        Sequential<
          L3,
          Sequential<
            L4, Sequential<L5, Sequential<L6, Sequential<L7, Sequential<L8, Sequential<L9, L10>>>>>
          >
        >
      >
    >
  where
    L1.Output == L2.Input,
    L2.Output == L3.Input,
    L3.Output == L4.Input,
    L4.Output == L5.Input,
    L5.Output == L6.Input,
    L6.Output == L7.Input,
    L7.Output == L8.Input,
    L8.Output == L9.Input,
    L9.Output == L10.Input,
    L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar,
    L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar,
    L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar,
    L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar,
    L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar,
    L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar,
    L7.TangentVector.VectorSpaceScalar == L8.TangentVector.VectorSpaceScalar,
    L8.TangentVector.VectorSpaceScalar == L9.TangentVector.VectorSpaceScalar,
    L9.TangentVector.VectorSpaceScalar == L10.TangentVector.VectorSpaceScalar
  {
    Sequential(
      l1,
      Sequential(
        l2,
        Sequential(
          l3,
          Sequential(
            l4, Sequential(l5, Sequential(l6, Sequential(l7, Sequential(l8, Sequential(l9, l10))))))
        )))
  }

}
