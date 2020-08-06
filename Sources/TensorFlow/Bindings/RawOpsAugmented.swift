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
  @_implementationOnly import x10_xla_tensor_wrapper
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
    argMax(input, dimension: Tensor<Int32>(Int32(dimension), on: .defaultTFEager))
  }

  public static func mean<
    T: TensorFlowNumeric
  >(
    _ input: Tensor<T>,
    reductionIndices: [Int64],
    keepDims: Bool = false
  ) -> Tensor<T> {
    mean(
      input,
      reductionIndices: Tensor<Int32>(reductionIndices.map { Int32($0) }, on: .defaultTFEager),
      keepDims: keepDims)
  }

  public static func pad<
    T: TensorFlowScalar
  >(
    _ input: Tensor<T>,
    paddings: [Int]
  ) -> Tensor<T> {
    precondition(paddings.count % 2 == 0, "Length of linearized paddings must be even")
    let paddings2D = Tensor(
      shape: [paddings.count / 2, 2],
      scalars: paddings.map { Int32($0) },
      on: .defaultTFEager)
    return pad(input, paddings: paddings2D)
  }

  public static func reshape<
    T: TensorFlowScalar
  >(
    _ tensor: Tensor<T>,
    shape: [Int64]
  ) -> Tensor<T> {
    reshape(tensor, shape: Tensor<Int32>(shape.map { Int32($0) }, on: .defaultTFEager))
  }

  public static func slice<
    T: TensorFlowScalar
  >(
    _ input: Tensor<T>,
    begin: [Int],
    size: [Int]
  ) -> Tensor<T> {
    slice(
      input, begin: Tensor<Int32>(begin.map { Int32($0) }, on: .defaultTFEager),
      size: Tensor<Int32>(size.map { Int32($0) }, on: .defaultTFEager))
  }

  public static func split<T: TensorFlowScalar>(
    splitDim: Int,
    value: Tensor<T>,
    numSplit: Int64
  ) -> [Tensor<T>] {
    split(
      splitDim: Tensor<Int32>(Int32(splitDim), on: .defaultTFEager), value: value,
      numSplit: numSplit)
  }

  public static func splitV<
    T: TensorFlowScalar
  >(
    value: Tensor<T>,
    sizeSplits: [Int],
    splitDim: Int
  ) -> [Tensor<T>] {
    splitV(
      value: value,
      sizeSplits: Tensor<Int32>(sizeSplits.map { Int32($0) }, on: .defaultTFEager),
      splitDim: Tensor<Int32>(Int32(splitDim), on: .defaultTFEager),
      numSplit: Int64(sizeSplits.count))
  }

  public static func sum<
    T: TensorFlowNumeric
  >(
    _ input: Tensor<T>,
    reductionIndices: [Int64],
    keepDims: Bool = false
  ) -> Tensor<T> {
    sum(
      input,
      reductionIndices: Tensor<Int32>(reductionIndices.map { Int32($0) }, on: .defaultTFEager),
      keepDims: keepDims)
  }

  public static func tile<
    T: TensorFlowScalar
  >(
    _ input: Tensor<T>,
    multiples: [Int]
  ) -> Tensor<T> {
    tile(input, multiples: Tensor<Int32>(multiples.map { Int32($0) }, on: .defaultTFEager))
  }

  public static func transpose<
    T: TensorFlowScalar
  >(
    _ x: Tensor<T>,
    perm: [Int]
  ) -> Tensor<T> {
    transpose(x, perm: Tensor<Int32>(perm.map { Int32($0) }, on: .defaultTFEager))
  }

  public static func unsortedSegmentSum<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
  >(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>,
    numSegments: Int
  ) -> Tensor<T> {
    unsortedSegmentSum(
      data: data, segmentIds: segmentIds,
      numSegments: Tensor(Int32(numSegments), on: .defaultTFEager))
  }

  public static func broadcastTo<
    T: TensorFlowScalar
  >(
    _ input: Tensor<T>,
    shape: [Int64]
  ) -> Tensor<T> {
    broadcastTo(input, shape: Tensor<Int32>(shape.map { Int32($0) }, on: .defaultTFEager))
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
      input, filterSizes: Tensor<Int32>(filterSizes.map { Int32($0) }, on: .defaultTFEager),
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
      inputSizes: Tensor<Int32>(inputSizes.map { Int32($0) }, on: .defaultTFEager), filter: filter,
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
      input, ksize: Tensor<Int32>(ksize.map { Int32($0) }, on: .defaultTFEager),
      strides: Tensor<Int32>(strides.map { Int32($0) }, on: .defaultTFEager), padding: padding,
      dataFormat: dataFormat)
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
      ksize: Tensor<Int32>(ksize.map { Int32($0) }, on: .defaultTFEager),
      strides: Tensor<Int32>(strides.map { Int32($0) }, on: .defaultTFEager),
      padding: padding, dataFormat: dataFormat)
  }
}

#if USING_X10_BACKEND
  extension _Raw {
    public static func commonBackend(
      _ a: Device.Backend, _ b: Device.Backend, file: StaticString = #file, line: UInt = #line
    ) -> Device.Backend {
      if a != b {
        fatalError("Op must have the same backend type: \(a) vs \(b)", file: file, line: line)
      }
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
      switch input.handle.backend {
      case .XLA:
        return _RawXLA.argMax(input, dimension: dimension)
      case .TF_EAGER:
        return _RawTFEager.argMax(
          input, dimension: Tensor<Int32>(Int32(dimension), on: .defaultTFEager))
      }
    }

    public static func mean<
      T: TensorFlowNumeric
    >(
      _ input: Tensor<T>,
      reductionIndices: [Int64],
      keepDims: Bool = false
    ) -> Tensor<T> {
      switch input.handle.backend {
      case .XLA:
        return _RawXLA.mean(
          input, reductionIndices: reductionIndices,
          keepDims: keepDims)
      case .TF_EAGER:
        return _RawTFEager.mean(
          input,
          reductionIndices: Tensor<Int32>(reductionIndices.map { Int32($0) }, on: .defaultTFEager),
          keepDims: keepDims)
      }
    }

    public static func pad<
      T: TensorFlowScalar
    >(
      _ input: Tensor<T>,
      paddings: [Int]
    ) -> Tensor<T> {
      switch input.handle.backend {
      case .XLA:
        return _RawXLA.pad(input, paddings: paddings)
      case .TF_EAGER:
        return _RawTFEager.pad(input, paddings: paddings)
      }
    }

    public static func reshape<
      T: TensorFlowScalar
    >(
      _ tensor: Tensor<T>,
      shape: [Int64]
    ) -> Tensor<T> {
      switch tensor.handle.backend {
      case .XLA:
        return _RawXLA.reshape(tensor, shape: shape)
      case .TF_EAGER:
        return _RawTFEager.reshape(
          tensor, shape: Tensor<Int32>(shape.map { Int32($0) }, on: .defaultTFEager))
      }
    }

    public static func slice<
      T: TensorFlowScalar
    >(
      _ input: Tensor<T>,
      begin: [Int],
      size: [Int]
    ) -> Tensor<T> {
      switch input.handle.backend {
      case .XLA:
        return _RawXLA.slice(input, begin: begin, size: size)
      case .TF_EAGER:
        return _RawTFEager.slice(
          input, begin: Tensor<Int32>(begin.map { Int32($0) }, on: .defaultTFEager),
          size: Tensor<Int32>(size.map { Int32($0) }, on: .defaultTFEager))
      }
    }

    public static func split<T: TensorFlowScalar>(
      splitDim: Int,
      value: Tensor<T>,
      numSplit: Int64
    ) -> [Tensor<T>] {
      switch value.handle.backend {
      case .XLA:
        return _RawXLA.split(splitDim: splitDim, value: value, numSplit: numSplit)
      case .TF_EAGER:
        return _RawTFEager.split(
          splitDim: Tensor<Int32>(Int32(splitDim), on: .defaultTFEager), value: value,
          numSplit: numSplit)
      }
    }

    public static func splitV<
      T: TensorFlowScalar
    >(
      value: Tensor<T>,
      sizeSplits: [Int],
      splitDim: Int
    ) -> [Tensor<T>] {
      switch value.handle.backend {
      case .XLA:
        return _RawXLA.splitV(value: value, sizeSplits: sizeSplits, splitDim: splitDim)
      case .TF_EAGER:
        return _RawTFEager.splitV(
          value: value,
          sizeSplits: Tensor<Int32>(sizeSplits.map { Int32($0) }, on: .defaultTFEager),
          splitDim: Tensor<Int32>(Int32(splitDim), on: .defaultTFEager),
          numSplit: Int64(sizeSplits.count))
      }
    }

    public static func tile<
      T: TensorFlowScalar
    >(
      _ input: Tensor<T>,
      multiples: [Int]
    ) -> Tensor<T> {
      switch input.handle.backend {
      case .XLA:
        return _RawXLA.tile(input, multiples: multiples)
      case .TF_EAGER:
        return _RawTFEager.tile(input, multiples: multiples)
      }
    }

    public static func sum<
      T: TensorFlowNumeric
    >(
      _ input: Tensor<T>,
      reductionIndices: [Int64],
      keepDims: Bool = false
    ) -> Tensor<T> {
      switch input.handle.backend {
      case .XLA:
        return _RawXLA.sum(
          input, reductionIndices: reductionIndices,
          keepDims: keepDims)
      case .TF_EAGER:
        return _RawTFEager.sum(
          input,
          reductionIndices: Tensor<Int32>(reductionIndices.map { Int32($0) }, on: .defaultTFEager),
          keepDims: keepDims)
      }
    }

    public static func transpose<
      T: TensorFlowScalar
    >(
      _ x: Tensor<T>,
      perm: [Int]
    ) -> Tensor<T> {
      switch x.handle.backend {
      case .XLA:
        return _RawXLA.transpose(x, perm: perm)
      case .TF_EAGER:
        return _RawTFEager.transpose(
          x, perm: Tensor<Int32>(perm.map { Int32($0) }, on: .defaultTFEager))
      }
    }

    public static func unsortedSegmentSum<
      T: TensorFlowNumeric,
      Tindices: TensorFlowIndex
    >(
      data: Tensor<T>,
      segmentIds: Tensor<Tindices>,
      numSegments: Int
    ) -> Tensor<T> {
      switch data.handle.backend {
      case .XLA:
        return _RawXLA.unsortedSegmentSum(
          data: data, segmentIds: segmentIds, numSegments: numSegments)
      case .TF_EAGER:
        return _RawTFEager.unsortedSegmentSum(
          data: data, segmentIds: segmentIds,
          numSegments: Tensor(Int32(numSegments), on: .defaultTFEager))
      }
    }

    public static func broadcastTo<
      T: TensorFlowScalar
    >(
      _ input: Tensor<T>,
      shape: [Int64]
    ) -> Tensor<T> {
      switch input.handle.backend {
      case .XLA:
        return _RawXLA.broadcastTo(input, shape: shape)
      case .TF_EAGER:
        return _RawTFEager.broadcastTo(
          input, shape: Tensor<Int32>(shape.map { Int32($0) }, on: .defaultTFEager))
      }
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
      switch commonBackend(input.handle.backend, outBackprop.handle.backend) {
      case .XLA:
        return _RawXLA.conv2DBackpropFilter(
          input, filterSizes: filterSizes,
          outBackprop: outBackprop, strides: strides, useCudnnOnGpu: useCudnnOnGpu,
          padding: padding, explicitPaddings: explicitPaddings, dataFormat: dataFormat,
          dilations: dilations)
      case .TF_EAGER:
        return _RawTFEager.conv2DBackpropFilter(
          input, filterSizes: Tensor<Int32>(filterSizes.map { Int32($0) }, on: .defaultTFEager),
          outBackprop: outBackprop, strides: strides, useCudnnOnGpu: useCudnnOnGpu,
          padding: padding, explicitPaddings: explicitPaddings, dataFormat: dataFormat,
          dilations: dilations)
      }
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
      switch commonBackend(filter.handle.backend, outBackprop.handle.backend) {
      case .XLA:
        return _RawXLA.conv2DBackpropInput(
          inputSizes: inputSizes, filter: filter,
          outBackprop: outBackprop,
          strides: strides, useCudnnOnGpu: useCudnnOnGpu, padding: padding,
          explicitPaddings: explicitPaddings, dataFormat: dataFormat, dilations: dilations)
      case .TF_EAGER:
        return _RawTFEager.conv2DBackpropInput(
          inputSizes: Tensor<Int32>(inputSizes.map { Int32($0) }, on: .defaultTFEager),
          filter: filter,
          outBackprop: outBackprop,
          strides: strides, useCudnnOnGpu: useCudnnOnGpu, padding: padding,
          explicitPaddings: explicitPaddings, dataFormat: dataFormat, dilations: dilations)
      }
    }

    public static func maxPoolV2<T: TensorFlowNumeric>(
      _ input: Tensor<T>,
      ksize: [Int64],
      strides: [Int64],
      padding: Padding,
      dataFormat: DataFormat2 = .nhwc
    ) -> Tensor<T> {
      switch input.handle.backend {
      case .XLA:
        return _RawXLA.maxPoolV2(
          input, ksize: ksize,
          strides: strides, padding: padding, dataFormat: dataFormat)
      case .TF_EAGER:
        return _RawTFEager.maxPoolV2(
          input, ksize: Tensor<Int32>(ksize.map { Int32($0) }, on: .defaultTFEager),
          strides: Tensor<Int32>(strides.map { Int32($0) }, on: .defaultTFEager), padding: padding,
          dataFormat: dataFormat)
      }
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
      switch commonBackend(origInput.handle.backend, origOutput.handle.backend) {
      case .XLA:
        return _RawXLA.maxPoolGradV2(
          origInput: origInput, origOutput: origOutput, grad: grad,
          ksize: ksize,
          strides: strides,
          padding: padding, dataFormat: dataFormat)
      case .TF_EAGER:
        return _RawTFEager.maxPoolGradV2(
          origInput: origInput, origOutput: origOutput, grad: grad,
          ksize: Tensor<Int32>(ksize.map { Int32($0) }, on: .defaultTFEager),
          strides: Tensor<Int32>(strides.map { Int32($0) }, on: .defaultTFEager),
          padding: padding, dataFormat: dataFormat)
      }
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
      if x.handle.backend == device.backend && device.backend == .XLA {
        return _RawXLA.toDevice(x, device)
      }
      return Tensor(shape: x.shape, scalars: x.scalars, on: device)
    }

    public static func physicalCast<T: TensorFlowScalar, R: TensorFlowScalar>(
      _ input: Tensor<T>, destType: R.Type
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
