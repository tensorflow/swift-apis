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

#if USING_X10_BACKEND
  import x10_xla_tensor_wrapper
#endif

// Augment the `_Raw` interface with ops that take Swift integers for the
// shape attributes rather than requiring that they be passed as `Int32` tensors.
// This is useful for implementations that do not require the parameters be
// passed as Tensors.
extension _RawTFEager {
  public static func argMax<
    T: TensorFlowNumeric,
    OutputType: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    dimension: Int64
  ) -> Tensor<OutputType> {
    argMax(input, dimension: Tensor<Int32>(Int32(dimension)))
  }

  public static func mean<
    T: TensorFlowNumeric
  >(
    _ input: Tensor<T>,
    reductionIndices: [Int64],
    keepDims: Bool = false
  ) -> Tensor<T> {
    mean(
      input, reductionIndices: Tensor<Int32>(reductionIndices.map { Int32($0) }),
      keepDims: keepDims)
  }

  public static func reshape<
    T: TensorFlowScalar
  >(
    _ tensor: Tensor<T>,
    shape: [Int64]
  ) -> Tensor<T> {
    reshape(tensor, shape: Tensor<Int32>(shape.map { Int32($0) }))
  }

  public static func sum<
    T: TensorFlowNumeric
  >(
    _ input: Tensor<T>,
    reductionIndices: [Int64],
    keepDims: Bool = false
  ) -> Tensor<T> {
    sum(
      input, reductionIndices: Tensor<Int32>(reductionIndices.map { Int32($0) }),
      keepDims: keepDims)
  }

  public static func broadcastTo<
    T: TensorFlowScalar
  >(
    _ input: Tensor<T>,
    shape: [Int64]
  ) -> Tensor<T> {
    broadcastTo(input, shape: Tensor<Int32>(shape.map { Int32($0) }))
  }

  public static func conv2DBackpropFilter<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filterSizes: [Int64],
    outBackprop: Tensor<T>,
    strides: [Int32],
    useCudnnOnGpu: Bool = true,
    padding: Padding1,
    explicitPaddings: [Int32],
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
  ) -> Tensor<T> {
    conv2DBackpropFilter(
      input, filterSizes: Tensor<Int32>(filterSizes.map { Int32($0) }),
      outBackprop: outBackprop, strides: strides, useCudnnOnGpu: useCudnnOnGpu,
      padding: padding, explicitPaddings: explicitPaddings, dataFormat: dataFormat,
      dilations: dilations)
  }

  public static func conv2DBackpropInput<T: FloatingPoint & TensorFlowScalar>(
    inputSizes: [Int64],
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    useCudnnOnGpu: Bool = true,
    padding: Padding1,
    explicitPaddings: [Int32],
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
  ) -> Tensor<T> {
    conv2DBackpropInput(
      inputSizes: Tensor<Int32>(inputSizes.map { Int32($0) }), filter: filter,
      outBackprop: outBackprop,
      strides: strides, useCudnnOnGpu: useCudnnOnGpu, padding: padding,
      explicitPaddings: explicitPaddings, dataFormat: dataFormat, dilations: dilations)
  }

  public static func maxPoolV2<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    ksize: [Int64],
    strides: [Int64],
    padding: Padding,
    dataFormat: DataFormat2 = .nhwc
  ) -> Tensor<T> {
    maxPoolV2(
      input, ksize: Tensor<Int32>(ksize.map { Int32($0) }),
      strides: Tensor<Int32>(strides.map { Int32($0) }), padding: padding, dataFormat: dataFormat)
  }

  public static func maxPoolGradV2<T: TensorFlowNumeric>(
    origInput: Tensor<T>,
    origOutput: Tensor<T>,
    grad: Tensor<T>,
    ksize: [Int64],
    strides: [Int64],
    padding: Padding,
    dataFormat: DataFormat = .nhwc
  ) -> Tensor<T> {
    maxPoolGradV2(
      origInput: origInput, origOutput: origOutput, grad: grad,
      ksize: Tensor<Int32>(ksize.map { Int32($0) }),
      strides: Tensor<Int32>(strides.map { Int32($0) }),
      padding: padding, dataFormat: dataFormat)
  }
}

#if USING_X10_BACKEND
  extension _Raw {
    public static func commonBackend(_ a: Device.Backend, _ b: Device.Backend) -> Device.Backend {
      if a != b { fatalError("Op must have the same backend type: \(a) vs \(b)") }
      return a
    }

    public static func commonBackend<T>(_ tensors: [Tensor<T>]) -> Device.Backend {
      var result = tensors.first!.handle.backend
      for tensor in tensors { result = commonBackend(result, tensor.handle.backend) }
      return result
    }

    public static func argMax<
      T: TensorFlowNumeric,
      OutputType: TensorFlowIndex
    >(
      _ input: Tensor<T>,
      dimension: Int64
    ) -> Tensor<OutputType> {
      argMax(input, dimension: Tensor<Int32>(Int32(dimension)))
    }

    public static func mean<
      T: TensorFlowNumeric
    >(
      _ input: Tensor<T>,
      reductionIndices: [Int64],
      keepDims: Bool = false
    ) -> Tensor<T> {
      mean(
        input, reductionIndices: Tensor<Int32>(reductionIndices.map { Int32($0) }),
        keepDims: keepDims)
    }

    public static func reshape<
      T: TensorFlowScalar
    >(
      _ tensor: Tensor<T>,
      shape: [Int64]
    ) -> Tensor<T> {
      reshape(tensor, shape: Tensor<Int32>(shape.map { Int32($0) }))
    }

    public static func sum<
      T: TensorFlowNumeric
    >(
      _ input: Tensor<T>,
      reductionIndices: [Int64],
      keepDims: Bool = false
    ) -> Tensor<T> {
      sum(
        input, reductionIndices: Tensor<Int32>(reductionIndices.map { Int32($0) }),
        keepDims: keepDims)
    }

    public static func broadcastTo<
      T: TensorFlowScalar
    >(
      _ input: Tensor<T>,
      shape: [Int64]
    ) -> Tensor<T> {
      broadcastTo(input, shape: Tensor<Int32>(shape.map { Int32($0) }))
    }

    public static func conv2DBackpropFilter<T: FloatingPoint & TensorFlowScalar>(
      _ input: Tensor<T>,
      filterSizes: [Int64],
      outBackprop: Tensor<T>,
      strides: [Int32],
      useCudnnOnGpu: Bool = true,
      padding: Padding1,
      explicitPaddings: [Int32],
      dataFormat: DataFormat = .nhwc,
      dilations: [Int32] = [1, 1, 1, 1]
    ) -> Tensor<T> {
      conv2DBackpropFilter(
        input, filterSizes: Tensor<Int32>(filterSizes.map { Int32($0) }),
        outBackprop: outBackprop, strides: strides, useCudnnOnGpu: useCudnnOnGpu,
        padding: padding, explicitPaddings: explicitPaddings, dataFormat: dataFormat,
        dilations: dilations)
    }

    public static func conv2DBackpropInput<T: FloatingPoint & TensorFlowScalar>(
      inputSizes: [Int64],
      filter: Tensor<T>,
      outBackprop: Tensor<T>,
      strides: [Int32],
      useCudnnOnGpu: Bool = true,
      padding: Padding1,
      explicitPaddings: [Int32],
      dataFormat: DataFormat = .nhwc,
      dilations: [Int32] = [1, 1, 1, 1]
    ) -> Tensor<T> {
      conv2DBackpropInput(
        inputSizes: Tensor<Int32>(inputSizes.map { Int32($0) }), filter: filter,
        outBackprop: outBackprop,
        strides: strides, useCudnnOnGpu: useCudnnOnGpu, padding: padding,
        explicitPaddings: explicitPaddings, dataFormat: dataFormat, dilations: dilations)
    }

    public static func maxPoolV2<T: TensorFlowNumeric>(
      _ input: Tensor<T>,
      ksize: [Int64],
      strides: [Int64],
      padding: Padding,
      dataFormat: DataFormat2 = .nhwc
    ) -> Tensor<T> {
      maxPoolV2(
        input, ksize: Tensor<Int32>(ksize.map { Int32($0) }),
        strides: Tensor<Int32>(strides.map { Int32($0) }), padding: padding, dataFormat: dataFormat)
    }

    public static func maxPoolGradV2<T: TensorFlowNumeric>(
      origInput: Tensor<T>,
      origOutput: Tensor<T>,
      grad: Tensor<T>,
      ksize: [Int64],
      strides: [Int64],
      padding: Padding,
      dataFormat: DataFormat = .nhwc
    ) -> Tensor<T> {
      maxPoolGradV2(
        origInput: origInput, origOutput: origOutput, grad: grad,
        ksize: Tensor<Int32>(ksize.map { Int32($0) }),
        strides: Tensor<Int32>(strides.map { Int32($0) }),
        padding: padding, dataFormat: dataFormat)
    }

    /// A simplified version of cross replica sum, with scaling.
    public static func crossReplicaSum<T: TensorFlowNumeric>(
      _ inputs: [Tensor<T>],
      _ scale: Double
    ) -> [Tensor<T>] {
      _RawXLA.crossReplicaSum(inputs, scale)
    }

    /// Transfer a tensor to a different device.
    public static func toDevice<T: TensorFlowScalar>(_ x: Tensor<T>, _ device: Device) -> Tensor<T>
    {
      _RawXLA.toDevice(x, device)
    }

    public static func physicalCast<T: TensorFlowScalar>(
      _ input: Tensor<T>, destType: XLATensorScalarType
    ) -> Tensor<T> {
      _RawXLA.physicalCast(input, destType: destType)
    }

    // Currently only used for deterministic testing.
    public static func rand(_ dims: [Int], _ seed: Int) -> Tensor<Float> {
      _RawXLA.rand(dims, seed)
    }

    public static func linSpace<
      T: FloatingPoint & TensorFlowScalar,
      Tidx: TensorFlowIndex
    >(
      start: Tensor<T>,
      stop: Tensor<T>,
      num: Tensor<Tidx>,
      device: Device
    ) -> Tensor<T> {
      _RawXLA.linSpace(start: start, stop: stop, num: num, device: device)
    }
  }
#endif
