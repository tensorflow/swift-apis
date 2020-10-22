// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// These are just the ops that should have manual lowerings to XLA in order to
// support the current tensorflow API.

@_implementationOnly import x10_xla_tensor_tf_ops
@_implementationOnly import x10_xla_tensor_wrapper

public enum _RawXLA {
  public typealias DataFormat = _RawTFEager.DataFormat
  public typealias DataFormat1 = _RawTFEager.DataFormat1
  public typealias DataFormat4 = _RawTFEager.DataFormat2
  public typealias Padding = _RawTFEager.Padding
  public typealias Padding1 = _RawTFEager.Padding1
  public typealias Mode5 = _RawTFEager.Mode1
  typealias AnyScalar = XLAScalarType
  typealias ScalarType = XLATensorScalarType

  private static func canonicalDims(_ dims: [Int64], _ rank: Int64) -> [Int64] {
    dims.map { $0 < 0 ? $0 + rank : $0 }
  }

  static func checkSameDevice(
    _ deviceLhs: Device, _ deviceRhs: Device, file: StaticString = #file, line: UInt = #line
  ) {
    if deviceLhs != deviceRhs {
      fatalError("All tensors must be on the same device, got \(deviceLhs) != \(deviceRhs) instead")
    }
  }

  private static func checkSameDevice<T>(
    _ t1: Tensor<T>, _ t2: Tensor<T>, file: StaticString = #file, line: UInt = #line
  ) {
    checkSameDevice(t1.device, t2.device, file: file, line: line)
  }

  static func checkSameDevice<T>(
    _ tensors: [Tensor<T>], file: StaticString = #file, line: UInt = #line
  ) {
    guard let device = tensors.first?.device else {
      return
    }
    for tensor in tensors {
      checkSameDevice(tensor.device, device, file: file, line: line)
    }
  }

  private static func checkSameDevice<T>(
    _ t1: Tensor<T>, _ t2: Tensor<T>, _ t3: Tensor<T>, file: StaticString = #file,
    line: UInt = #line
  ) {
    let device = t1.device
    checkSameDevice(device, t2.device, file: file, line: line)
    checkSameDevice(device, t3.device, file: file, line: line)
  }

  private static func checkSamePrecision(
    _ isReducedPrecisionLhs: Bool, _ isReducedPrecisionRhs: Bool, file: StaticString = #file,
    line: UInt = #line
  ) {
    if isReducedPrecisionLhs != isReducedPrecisionRhs {
      fatalError(
        "All tensors must have the same precision (reduced or full)", file: file, line: line)
    }
  }

  static func checkSamePrecision<T>(
    _ t1: Tensor<T>, _ t2: Tensor<T>, file: StaticString = #file, line: UInt = #line
  ) {
    checkSamePrecision(t1.isReducedPrecision, t2.isReducedPrecision, file: file, line: line)
  }

  private static func checkSamePrecision<T>(
    _ t1: Tensor<T>, _ t2: Tensor<T>, _ t3: Tensor<T>, file: StaticString = #file,
    line: UInt = #line
  ) {
    let isReducedPrecision = t1.isReducedPrecision
    checkSamePrecision(isReducedPrecision, t2.isReducedPrecision, file: file, line: line)
    checkSamePrecision(isReducedPrecision, t3.isReducedPrecision, file: file, line: line)
  }

  static func checkSamePrecision<T>(
    _ tensors: [Tensor<T>], file: StaticString = #file, line: UInt = #line
  ) {
    guard let isReducedPrecision = tensors.first?.isReducedPrecision else {
      return
    }
    for tensor in tensors {
      checkSamePrecision(tensor.isReducedPrecision, isReducedPrecision, file: file, line: line)
    }
  }

  /// Computes the "logical and" of elements across dimensions of a tensor.
  ///
  /// Reduces `input` along the dimensions given in `axis`. Unless
  /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  /// `axis`. If `keep_dims` is true, the reduced dimensions are
  /// retained with length 1.
  ///
  /// - Parameters:
  ///     - input: The tensor to reduce.
  ///     - reduction_indices: The dimensions to reduce. Must be in the range
  ///         `[-rank(input), rank(input))`.
  ///
  /// - Attr keep_dims: If true, retain reduced dimensions with length 1.
  ///
  /// - Output output: The reduced tensor.
  public static func all<Tidx: TensorFlowIndex>(
    _ input: Tensor<Bool>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
  ) -> Tensor<Bool> {
    return all(
      input, dims: reductionIndices.scalars.map { Int64($0) }, keep_reduced_dimensions: keepDims)
  }

  /// Computes the "logical or" of elements across dimensions of a tensor.
  ///
  /// Reduces `input` along the dimensions given in `axis`. Unless
  /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  /// `axis`. If `keep_dims` is true, the reduced dimensions are
  /// retained with length 1.
  ///
  /// - Parameters:
  ///     - input: The tensor to reduce.
  ///     - reduction_indices: The dimensions to reduce. Must be in the range
  ///         `[-rank(input), rank(input))`.
  ///
  /// - Attr keep_dims: If true, retain reduced dimensions with length 1.
  ///
  /// - Output output: The reduced tensor.
  public static func any<Tidx: TensorFlowIndex>(
    _ input: Tensor<Bool>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
  ) -> Tensor<Bool> {
    return any(
      input, dims: reductionIndices.scalars.map { Int64($0) }, keep_reduced_dimensions: keepDims)
  }

  /// Returns the truth value of abs(x-y) < tolerance element-wise.
  public static func approximateEqual<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    tolerance: Double = 1e-05
  ) -> Tensor<Bool> {
    checkSameDevice(x, y)
    checkSamePrecision(x, y)
    let absDiff: Tensor<T> = abs(x - y)
    return less(absDiff, fullLike(tolerance, absDiff))
  }

  /// Returns the index with the largest value across dimensions of a tensor.
  ///
  /// Note that in case of ties the identity of the return value is not guaranteed.
  ///
  /// Usage:
  ///   ```python
  ///   import tensorflow as tf
  ///   a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  ///   b = tf.math.argmax(input = a)
  ///   c = tf.keras.backend.eval(b)
  ///   # c = 4
  ///   # here a[4] = 166.32 which is the largest element of a across axis 0
  ///   ```
  ///
  /// - Parameter dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`.
  ///     Describes which dimension of the input Tensor to reduce across. For vectors,
  ///     use dimension = 0.
  public static func argMax<
    T: TensorFlowNumeric,
    OutputType: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    dimension: Int64
  ) -> Tensor<OutputType> {
    return cast(argMax(input, dim: dimension, keepdim: false))
  }
  public static func argMax<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex,
    OutputType: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    dimension: Tensor<Tidx>
  ) -> Tensor<OutputType> {
    return argMax(input, dimension: Int64(dimension.scalarized()))
  }

  /// Returns the index with the smallest value across dimensions of a tensor.
  ///
  /// Note that in case of ties the identity of the return value is not guaranteed.
  ///
  /// Usage:
  ///   ```python
  ///   import tensorflow as tf
  ///   a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  ///   b = tf.math.argmin(input = a)
  ///   c = tf.keras.backend.eval(b)
  ///   # c = 0
  ///   # here a[0] = 1 which is the smallest element of a across axis 0
  ///   ```
  ///
  /// - Parameter dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`.
  ///     Describes which dimension of the input Tensor to reduce across. For vectors,
  ///     use dimension = 0.
  public static func argMin<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex,
    OutputType: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    dimension: Tensor<Tidx>
  ) -> Tensor<OutputType> {
    return cast(argMin(input, dim: Int64(dimension.scalarized()), keepdim: false))
  }

  private static func convertPadding(_ padding: Padding) -> TFPadding {
    switch padding {
    case .same: return TFPadding_SAME
    case .valid: return TFPadding_VALID
    }
  }

  private static func convertPadding1(_ padding: Padding1) -> TFPadding {
    switch padding {
    case .explicit: return TFPadding_EXPLICIT
    case .same: return TFPadding_SAME
    case .valid: return TFPadding_VALID
    }
  }

  private static func convertDataFormat(_ dataFormat: DataFormat) -> TFDataFormat {
    switch dataFormat {
    case .nchw: return TFDataFormat_NCHW
    case .nhwc: return TFDataFormat_NHWC
    }
  }

  private static func convertDataFormat1(_ dataFormat: DataFormat1) -> TFDataFormat {
    switch dataFormat {
    case .ncdhw: return TFDataFormat_NCHW
    case .ndhwc: return TFDataFormat_NHWC
    }
  }

  private static func convertDataFormat4(_ dataFormat: DataFormat4) -> TFDataFormat {
    switch dataFormat {
    case .nchw: return TFDataFormat_NCHW
    case .nchwVectC: return TFDataFormat_NCHW_VECT_C
    case .nhwc: return TFDataFormat_NHWC
    }
  }

  private static func convertMirrorPadMode(_ mode: Mode5) -> TFMirrorPadMode {
    switch mode {
    case .reflect: return TFMirrorPadMode_REFLECT
    case .symmetric: return TFMirrorPadMode_SYMMETRIC
    }
  }

  // Given low and high paddings for a list of dimensions, reverse the dimensions. This is required
  // because x10 and TensorFlow use opposite dimension ordering for pad specification.
  private static func reversedPaddings(_ paddings: [Int64]) -> [Int64] {
    var reversedPaddings: [Int64] = paddings.reversed()
    for i in stride(from: 0, to: reversedPaddings.count, by: 2) {
      reversedPaddings.swapAt(i, i + 1)
    }
    return reversedPaddings
  }

  /// Performs average pooling on the input.
  ///
  /// Each entry in `output` is the mean of the corresponding size `ksize`
  /// window in `value`.
  ///
  /// - Parameter value: 4-D with shape `[batch, height, width, channels]`.
  ///
  /// - Attrs:
  ///     - ksize: The size of the sliding window for each dimension of `value`.
  ///     - strides: The stride of the sliding window for each dimension of `value`.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: Specify the data format of the input and output data. With the
  ///         default format "NHWC", the data is stored in the order of:
  ///             [batch, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCHW", the data storage order of:
  ///             [batch, in_channels, in_height, in_width].
  ///
  /// - Output output: The average pooled output tensor.
  public static func avgPool<T: FloatingPoint & TensorFlowScalar>(
    value: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc
  ) -> Tensor<T> {
    Tensor(
      _xla: XLATensor.avgpool(
        value.xlaTensor, ksize.map { Int64($0) }, strides.map { Int64($0) },
        convertPadding(padding), convertDataFormat(dataFormat)))
  }

  /// Performs 3D average pooling on the input.
  ///
  /// - Parameter input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
  ///
  /// - Attrs:
  ///     - ksize: 1-D tensor of length 5. The size of the window for each dimension of
  ///         the input tensor. Must have `ksize[0] = ksize[4] = 1`.
  ///     - strides: 1-D tensor of length 5. The stride of the sliding window for each
  ///         dimension of `input`. Must have `strides[0] = strides[4] = 1`.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: The data format of the input and output data. With the
  ///         default format "NDHWC", the data is stored in the order of:
  ///             [batch, in_depth, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCDHW", the data storage order is:
  ///             [batch, in_channels, in_depth, in_height, in_width].
  ///
  /// - Output output: The average pooled output tensor.
  public static func avgPool3D<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc
  ) -> Tensor<T> {
    Tensor(
      _xla: XLATensor.avgpool(
        input.xlaTensor, ksize.map { Int64($0) }, strides.map { Int64($0) },
        convertPadding(padding), convertDataFormat1(dataFormat)))
  }

  /// Computes gradients of average pooling function.
  ///
  /// - Parameters:
  ///     - orig_input_shape: The original input dimensions.
  ///     - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
  ///
  /// - Attrs:
  ///     - ksize: 1-D tensor of length 5. The size of the window for each dimension of
  ///         the input tensor. Must have `ksize[0] = ksize[4] = 1`.
  ///     - strides: 1-D tensor of length 5. The stride of the sliding window for each
  ///         dimension of `input`. Must have `strides[0] = strides[4] = 1`.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: The data format of the input and output data. With the
  ///         default format "NDHWC", the data is stored in the order of:
  ///             [batch, in_depth, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCDHW", the data storage order is:
  ///             [batch, in_channels, in_depth, in_height, in_width].
  ///
  /// - Output output: The backprop for input.
  public static func avgPool3DGrad<T: FloatingPoint & TensorFlowScalar>(
    origInputShape: Tensor<Int32>,
    grad: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc
  ) -> Tensor<T> {
    Tensor(
      _xla: XLATensor.avgpool_grad(
        origInputShape.scalars.map { Int64($0) }, grad.xlaTensor, ksize.map { Int64($0) },
        strides.map { Int64($0) },
        convertPadding(padding), convertDataFormat1(dataFormat)))
  }

  /// Computes gradients of the average pooling function.
  ///
  /// - Parameters:
  ///     - orig_input_shape: 1-D.  Shape of the original input to `avg_pool`.
  ///     - grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
  ///         the output of `avg_pool`.
  ///
  /// - Attrs:
  ///     - ksize: The size of the sliding window for each dimension of the input.
  ///     - strides: The stride of the sliding window for each dimension of the input.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: Specify the data format of the input and output data. With the
  ///         default format "NHWC", the data is stored in the order of:
  ///             [batch, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCHW", the data storage order of:
  ///             [batch, in_channels, in_height, in_width].
  ///
  /// - Output output: 4-D.  Gradients w.r.t. the input of `avg_pool`.
  public static func avgPoolGrad<T: FloatingPoint & TensorFlowScalar>(
    origInputShape: Tensor<Int32>,
    grad: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc
  ) -> Tensor<T> {
    Tensor(
      _xla: XLATensor.avgpool_grad(
        origInputShape.scalars.map { Int64($0) }, grad.xlaTensor, ksize.map { Int64($0) },
        strides.map { Int64($0) },
        convertPadding(padding), convertDataFormat(dataFormat)))
  }

  /// Multiplies slices of two tensors in batches.
  ///
  /// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  /// viewed as an element of a batch), and arranges the individual results
  /// in a single output tensor of the same batch size. Each of the
  /// individual slices can optionally be adjointed (to adjoint a matrix
  /// means to transpose and conjugate it) before multiplication by setting
  /// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
  ///
  /// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
  /// and `[..., r_y, c_y]`.
  ///
  /// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
  ///
  ///     r_o = c_x if adj_x else r_x
  ///     c_o = r_y if adj_y else c_y
  ///
  /// It is computed as:
  ///
  ///     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
  ///
  /// *NOTE*: `BatchMatMulV2` supports broadcasting in the batch dimensions. More
  /// about broadcasting
  /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
  ///
  ///
  /// - Parameters:
  ///     - x: 2-D or higher with shape `[..., r_x, c_x]`.
  ///     - y: 2-D or higher with shape `[..., r_y, c_y]`.
  ///
  /// - Attrs:
  ///     - adj_x: If `True`, adjoint the slices of `x`. Defaults to `False`.
  ///     - adj_y: If `True`, adjoint the slices of `y`. Defaults to `False`.
  ///
  /// - Output output: 3-D or higher with shape `[..., r_o, c_o]`
  public static func batchMatMulV2<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    adjX: Bool = false,
    adjY: Bool = false
  ) -> Tensor<T> {
    // TODO(parkers): Conjugation is unhandled, but T is never complex.
    let xrank = Int64(x.rank)
    let yrank = Int64(y.rank)
    precondition(xrank >= 2 && yrank >= 2, "Ranks must be >= 2")
    return matmul(
      (adjX
        ? permute(x, dims: [Int64](0..<xrank - 2) + [xrank - 1, xrank - 2]) : x),
      (adjY
        ? permute(y, dims: [Int64](0..<yrank - 2) + [yrank - 1, yrank - 2]) : y))
  }

  /// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
  ///
  /// This is typically used by gradient computations for a broadcasting operation.
  public static func broadcastGradientArgs<T: TensorFlowIndex>(
    s0: Tensor<T>,
    s1: Tensor<T>
  ) -> (r0: Tensor<T>, r1: Tensor<T>) {
    checkSameDevice(s0, s1)
    checkSamePrecision(s0, s1)
    var shape0 = s0.scalars
    var shape1 = s1.scalars
    var reduceIdx0 = [T]()
    var reduceIdx1 = [T]()
    shape0.reverse()
    shape1.reverse()
    while shape0.count < shape1.count { shape0.append(1) }
    while shape1.count < shape0.count { shape1.append(1) }
    let n = shape1.count
    for i in 0..<n {
      let d0 = shape0[i]
      let d1 = shape1[i]
      if d0 == 1 { reduceIdx0.append(T(n - i - 1)) }
      if d1 == 1 { reduceIdx1.append(T(n - i - 1)) }
    }
    reduceIdx0.reverse()
    reduceIdx1.reverse()
    let device = s0.device
    return (r0: Tensor(reduceIdx0, on: device), r1: Tensor(reduceIdx1, on: device))
  }

  /// Broadcast an array for a compatible shape.
  ///
  /// Broadcasting is the process of making arrays to have compatible shapes
  /// for arithmetic operations. Two shapes are compatible if for each
  /// dimension pair they are either equal or one of them is one. When trying
  /// to broadcast a Tensor to a shape, it starts with the trailing dimensions,
  /// and works its way forward.
  ///
  /// For example,
  /// ```
  /// >>> x = tf.constant([1, 2, 3])
  /// >>> y = tf.broadcast_to(x, [3, 3])
  /// >>> sess.run(y)
  /// array([[1, 2, 3],
  ///        [1, 2, 3],
  ///        [1, 2, 3]], dtype=int32)
  /// ```
  /// In the above example, the input Tensor with the shape of `[1, 3]`
  /// is broadcasted to output Tensor with shape of `[3, 3]`.
  ///
  /// - Parameters:
  ///     - input: A Tensor to broadcast.
  ///     - shape: An 1-D `int` Tensor. The shape of the desired output.
  ///
  /// - Output output: A Tensor.
  public static func broadcastTo<
    T: TensorFlowScalar
  >(
    _ input: Tensor<T>,
    shape: [Int64]
  ) -> Tensor<T> {
    broadcastTo(input, dims: shape)
  }
  public static func broadcastTo<
    T: TensorFlowScalar,
    Tidx: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    shape: Tensor<Tidx>
  ) -> Tensor<T> {
    broadcastTo(input, shape: shape.scalars.map { Int64($0) })
  }

  /// Cast x of type SrcT to y of DstT.
  public static func cast<
    Srct: TensorFlowScalar,
    Dstt: TensorFlowScalar
  >(
    _ x: Tensor<Srct>,
    truncate: Bool = false
  ) -> Tensor<Dstt> {
    return logicalCast(x, destType: Dstt.xlaTensorScalarType)
  }

  /// Concatenates tensors along one dimension.
  ///
  /// - Parameters:
  ///     - values: List of `N` Tensors to concatenate. Their ranks and types must match,
  ///         and their sizes must match in all dimensions except `concat_dim`.
  ///     - axis: 0-D.  The dimension along which to concatenate.  Must be in the
  ///         range [-rank(values), rank(values)).
  ///
  /// - Output output: A `Tensor` with the concatenation of values stacked along the
  ///     `concat_dim` dimension.  This tensor's shape matches that of `values` except
  ///     in `concat_dim` where it has the sum of the sizes.
  public static func concatV2<
    T: TensorFlowScalar,
    Tidx: TensorFlowIndex
  >(
    _ values: [Tensor<T>],
    axis: Tensor<Tidx>
  ) -> Tensor<T> {
    return concat(values, dim: Int64(axis.scalarized()))
  }

  /// Computes a 2-D convolution given 4-D `input` and `filter` tensors.
  ///
  /// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  /// and a filter / kernel tensor of shape
  /// `[filter_height, filter_width, in_channels, out_channels]`, this op
  /// performs the following:
  ///
  /// 1. Flattens the filter to a 2-D matrix with shape
  ///    `[filter_height * filter_width * in_channels, output_channels]`.
  /// 2. Extracts image patches from the input tensor to form a *virtual*
  ///    tensor of shape `[batch, out_height, out_width,
  ///    filter_height * filter_width * in_channels]`.
  /// 3. For each patch, right-multiplies the filter matrix and the image patch
  ///    vector.
  ///
  /// In detail, with the default NHWC format,
  ///
  ///     output[b, i, j, k] =
  ///         sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
  ///                         filter[di, dj, q, k]
  ///
  /// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  /// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
  ///
  /// - Parameters:
  ///     - input: A 4-D tensor. The dimension order is interpreted according to the value
  ///         of `data_format`, see below for details.
  ///     - filter: A 4-D tensor of shape
  ///         `[filter_height, filter_width, in_channels, out_channels]`
  ///
  /// - Attrs:
  ///     - strides: 1-D tensor of length 4.  The stride of the sliding window for each
  ///         dimension of `input`. The dimension order is determined by the value of
  ///         `data_format`, see below for details.
  ///     - padding: The type of padding algorithm to use.
  ///     - explicit_paddings: If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
  ///         dimension, the amount of padding inserted before and after the dimension is
  ///         `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
  ///         `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
  ///     - data_format: Specify the data format of the input and output data. With the
  ///         default format "NHWC", the data is stored in the order of:
  ///             [batch, height, width, channels].
  ///         Alternatively, the format could be "NCHW", the data storage order of:
  ///             [batch, channels, height, width].
  ///     - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
  ///         `input`. If set to k > 1, there will be k-1 skipped cells between each
  ///         filter element on that dimension. The dimension order is determined by the
  ///         value of `data_format`, see above for details. Dilations in the batch and
  ///         depth dimensions must be 1.
  ///
  /// - Output output: A 4-D tensor. The dimension order is determined by the value of
  ///     `data_format`, see below for details.
  public static func conv2D<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    strides: [Int32],
    useCudnnOnGpu: Bool = true,
    padding: Padding1,
    explicitPaddings: [Int32],
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
  ) -> Tensor<T> {
    return tf_Conv(
      input, filter, false, strides.map { Int64($0) },
      convertPadding1(padding),
      explicitPaddings.map { Int64($0) }, convertDataFormat(dataFormat),
      dilations.map { Int64($0) })
  }

  /// Computes the gradients of convolution with respect to the filter.
  ///
  /// - Parameters:
  ///     - input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
  ///     - filter_sizes: An integer vector representing the tensor shape of `filter`,
  ///         where `filter` is a 4-D
  ///         `[filter_height, filter_width, in_channels, out_channels]` tensor.
  ///     - out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
  ///         Gradients w.r.t. the output of the convolution.
  ///
  /// - Attrs:
  ///     - strides: The stride of the sliding window for each dimension of the input
  ///         of the convolution. Must be in the same order as the dimension specified with
  ///         format.
  ///     - padding: The type of padding algorithm to use.
  ///     - explicit_paddings: If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
  ///         dimension, the amount of padding inserted before and after the dimension is
  ///         `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
  ///         `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
  ///     - data_format: Specify the data format of the input and output data. With the
  ///         default format "NHWC", the data is stored in the order of:
  ///             [batch, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCHW", the data storage order of:
  ///             [batch, in_channels, in_height, in_width].
  ///     - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
  ///         `input`. If set to k > 1, there will be k-1 skipped cells between each filter
  ///         element on that dimension. The dimension order is determined by the value of
  ///         `data_format`, see above for details. Dilations in the batch and depth
  ///         dimensions must be 1.
  ///
  /// - Output output: 4-D with shape
  ///     `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
  ///     the `filter` input of the convolution.
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
    return tf_ConvBackpropFilter(
      input, filterSizes,
      outBackprop, false, strides.map { Int64($0) },
      convertPadding1(padding), explicitPaddings.map { Int64($0) },
      convertDataFormat(dataFormat), dilations.map { Int64($0) })
  }
  public static func conv2DBackpropFilter<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filterSizes: Tensor<Int32>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    useCudnnOnGpu: Bool = true,
    padding: Padding1,
    explicitPaddings: [Int32],
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
  ) -> Tensor<T> {
    return tf_ConvBackpropFilter(
      input, filterSizes.scalars.map { Int64($0) },
      outBackprop, false, strides.map { Int64($0) },
      convertPadding1(padding), explicitPaddings.map { Int64($0) },
      convertDataFormat(dataFormat), dilations.map { Int64($0) })
  }

  /// Computes the gradients of convolution with respect to the input.
  ///
  /// - Parameters:
  ///     - input_sizes: An integer vector representing the shape of `input`,
  ///         where `input` is a 4-D `[batch, height, width, channels]` tensor.
  ///     - filter: 4-D with shape
  ///         `[filter_height, filter_width, in_channels, out_channels]`.
  ///     - out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
  ///         Gradients w.r.t. the output of the convolution.
  ///
  /// - Attrs:
  ///     - strides: The stride of the sliding window for each dimension of the input
  ///         of the convolution. Must be in the same order as the dimension specified with
  ///         format.
  ///     - padding: The type of padding algorithm to use.
  ///     - explicit_paddings: If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
  ///         dimension, the amount of padding inserted before and after the dimension is
  ///         `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
  ///         `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
  ///     - data_format: Specify the data format of the input and output data. With the
  ///         default format "NHWC", the data is stored in the order of:
  ///             [batch, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCHW", the data storage order of:
  ///             [batch, in_channels, in_height, in_width].
  ///     - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
  ///         `input`. If set to k > 1, there will be k-1 skipped cells between each filter
  ///         element on that dimension. The dimension order is determined by the value of
  ///         `data_format`, see above for details. Dilations in the batch and depth
  ///         dimensions must be 1.
  ///
  /// - Output output: 4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
  ///     w.r.t. the input of the convolution.
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
    return tf_ConvBackpropInput(
      inputSizes, filter,
      outBackprop, false, strides.map { Int64($0) },
      convertPadding1(padding), explicitPaddings.map { Int64($0) },
      convertDataFormat(dataFormat), dilations.map { Int64($0) })
  }
  public static func conv2DBackpropInput<T: TensorFlowNumeric>(
    inputSizes: Tensor<Int32>,
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    useCudnnOnGpu: Bool = true,
    padding: Padding1,
    explicitPaddings: [Int32],
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
  ) -> Tensor<T> {
    return tf_ConvBackpropInput(
      inputSizes.scalars.map { Int64($0) }, filter,
      outBackprop, false, strides.map { Int64($0) },
      convertPadding1(padding), explicitPaddings.map { Int64($0) },
      convertDataFormat(dataFormat), dilations.map { Int64($0) })
  }

  /// Computes a 3-D convolution given 5-D `input` and `filter` tensors.
  ///
  /// In signal processing, cross-correlation is a measure of similarity of
  /// two waveforms as a function of a time-lag applied to one of them. This
  /// is also known as a sliding dot product or sliding inner-product.
  ///
  /// Our Conv3D implements a form of cross-correlation.
  ///
  /// - Parameters:
  ///     - input: Shape `[batch, in_depth, in_height, in_width, in_channels]`.
  ///     - filter: Shape `[filter_depth, filter_height, filter_width, in_channels,
  ///         out_channels]`. `in_channels` must match between `input` and `filter`.
  ///
  /// - Attrs:
  ///     - strides: 1-D tensor of length 5. The stride of the sliding window for each
  ///         dimension of `input`. Must have `strides[0] = strides[4] = 1`.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: The data format of the input and output data. With the
  ///         default format "NDHWC", the data is stored in the order of:
  ///             [batch, in_depth, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCDHW", the data storage order is:
  ///             [batch, in_channels, in_depth, in_height, in_width].
  ///     - dilations: 1-D tensor of length 5.  The dilation factor for each dimension of
  ///         `input`. If set to k > 1, there will be k-1 skipped cells between each
  ///         filter element on that dimension. The dimension order is determined by the
  ///         value of `data_format`, see above for details. Dilations in the batch and
  ///         depth dimensions must be 1.
  public static func conv3D<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc,
    dilations: [Int32] = [1, 1, 1, 1, 1]
  ) -> Tensor<T> {
    return tf_Conv(
      input, filter, false, strides.map { Int64($0) },
      convertPadding(padding),
      [], convertDataFormat1(dataFormat), dilations.map { Int64($0) })
  }

  /// Computes the gradients of 3-D convolution with respect to the filter.
  ///
  /// - Parameters:
  ///     - input: Shape `[batch, depth, rows, cols, in_channels]`.
  ///     - filter_sizes: An integer vector representing the tensor shape of `filter`,
  ///         where `filter` is a 5-D
  ///         `[filter_depth, filter_height, filter_width, in_channels, out_channels]`
  ///         tensor.
  ///     - out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
  ///         out_channels]`.
  ///
  /// - Attrs:
  ///     - strides: 1-D tensor of length 5. The stride of the sliding window for each
  ///         dimension of `input`. Must have `strides[0] = strides[4] = 1`.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: The data format of the input and output data. With the
  ///         default format "NDHWC", the data is stored in the order of:
  ///             [batch, in_depth, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCDHW", the data storage order is:
  ///             [batch, in_channels, in_depth, in_height, in_width].
  ///     - dilations: 1-D tensor of length 5.  The dilation factor for each dimension of
  ///         `input`. If set to k > 1, there will be k-1 skipped cells between each
  ///         filter element on that dimension. The dimension order is determined by the
  ///         value of `data_format`, see above for details. Dilations in the batch and
  ///         depth dimensions must be 1.
  public static func conv3DBackpropFilterV2<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filterSizes: Tensor<Int32>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc,
    dilations: [Int32] = [1, 1, 1, 1, 1]
  ) -> Tensor<T> {
    return tf_ConvBackpropFilter(
      input, filterSizes.scalars.map { Int64($0) },
      outBackprop, false, strides.map { Int64($0) },
      convertPadding(padding), [], convertDataFormat1(dataFormat),
      dilations.map { Int64($0) })
  }

  /// Computes the gradients of 3-D convolution with respect to the input.
  ///
  /// - Parameters:
  ///     - input_sizes: An integer vector representing the tensor shape of `input`,
  ///         where `input` is a 5-D
  ///         `[batch, depth, rows, cols, in_channels]` tensor.
  ///     - filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
  ///         `in_channels` must match between `input` and `filter`.
  ///     - out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
  ///         out_channels]`.
  ///
  /// - Attrs:
  ///     - strides: 1-D tensor of length 5. The stride of the sliding window for each
  ///         dimension of `input`. Must have `strides[0] = strides[4] = 1`.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: The data format of the input and output data. With the
  ///         default format "NDHWC", the data is stored in the order of:
  ///             [batch, in_depth, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCDHW", the data storage order is:
  ///             [batch, in_channels, in_depth, in_height, in_width].
  ///     - dilations: 1-D tensor of length 5.  The dilation factor for each dimension of
  ///         `input`. If set to k > 1, there will be k-1 skipped cells between each
  ///         filter element on that dimension. The dimension order is determined by the
  ///         value of `data_format`, see above for details. Dilations in the batch and
  ///         depth dimensions must be 1.
  public static func conv3DBackpropInputV2<
    T: FloatingPoint & TensorFlowScalar,
    Tshape: TensorFlowIndex
  >(
    inputSizes: Tensor<Tshape>,
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc,
    dilations: [Int32] = [1, 1, 1, 1, 1]
  ) -> Tensor<T> {
    return tf_ConvBackpropInput(
      inputSizes.scalars.map { Int64($0) }, filter,
      outBackprop, false, strides.map { Int64($0) },
      convertPadding(padding), [], convertDataFormat1(dataFormat),
      dilations.map { Int64($0) })
  }

  /// A simplified version of cross replica sum, with scaling.
  public static func crossReplicaSum<T: TensorFlowNumeric>(
    _ inputs: [Tensor<T>],
    _ scale: Double
  ) -> [Tensor<T>] {
    XLATensor.crossReplicaSum(inputs.map { $0.xlaTensor }, scale).map {
      Tensor(_xla: $0)
    }
  }

  /// Compute the cumulative product of the tensor `x` along `axis`.
  ///
  /// By default, this op performs an inclusive cumprod, which means that the first
  /// element of the input is identical to the first element of the output:
  ///
  /// ```python
  /// tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
  /// ```
  ///
  /// By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
  /// performed instead:
  ///
  /// ```python
  /// tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
  /// ```
  ///
  /// By setting the `reverse` kwarg to `True`, the cumprod is performed in the
  /// opposite direction:
  ///
  /// ```python
  /// tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
  /// ```
  ///
  /// This is more efficient than using separate `tf.reverse` ops.
  ///
  /// The `reverse` and `exclusive` kwargs can also be combined:
  ///
  /// ```python
  /// tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
  /// ```
  ///
  /// - Parameters:
  ///     - x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
  ///         `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
  ///         `complex128`, `qint8`, `quint8`, `qint32`, `half`.
  ///     - axis: A `Tensor` of type `int32` (default: 0). Must be in the range
  ///         `[-rank(x), rank(x))`.
  ///
  /// - Attrs:
  ///     - exclusive: If `True`, perform exclusive cumprod.
  ///     - reverse: A `bool` (default: False).
  public static func cumprod<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
  >(
    _ x: Tensor<T>,
    axis: Tensor<Tidx>,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> Tensor<T> {
    cumprod(x, dim: Int64(axis.scalarized()), exclusive: exclusive, reverse: reverse)
  }

  /// Compute the cumulative sum of the tensor `x` along `axis`.
  ///
  /// By default, this op performs an inclusive cumsum, which means that the first
  /// element of the input is identical to the first element of the output:
  ///
  /// ```python
  /// tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
  /// ```
  ///
  /// By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
  /// performed instead:
  ///
  /// ```python
  /// tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
  /// ```
  ///
  /// By setting the `reverse` kwarg to `True`, the cumsum is performed in the
  /// opposite direction:
  ///
  /// ```python
  /// tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
  /// ```
  ///
  /// This is more efficient than using separate `tf.reverse` ops.
  ///
  /// The `reverse` and `exclusive` kwargs can also be combined:
  ///
  /// ```python
  /// tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
  /// ```
  ///
  /// - Parameters:
  ///     - x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
  ///         `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
  ///         `complex128`, `qint8`, `quint8`, `qint32`, `half`.
  ///     - axis: A `Tensor` of type `int32` (default: 0). Must be in the range
  ///         `[-rank(x), rank(x))`.
  ///
  /// - Attrs:
  ///     - exclusive: If `True`, perform exclusive cumsum.
  ///     - reverse: A `bool` (default: False).
  public static func cumsum<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
  >(
    _ x: Tensor<T>,
    axis: Tensor<Tidx>,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> Tensor<T> {
    cumsum(x, dim: Int64(axis.scalarized()), exclusive: exclusive, reverse: reverse)
  }

  /// Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.
  ///
  /// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  /// and a filter / kernel tensor of shape
  /// `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
  /// `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
  /// a different filter to each input channel (expanding from 1 channel to
  /// `channel_multiplier` channels for each), then concatenates the results
  /// together. Thus, the output has `in_channels * channel_multiplier` channels.
  ///
  /// ```
  /// for k in 0..in_channels-1
  ///   for q in 0..channel_multiplier-1
  ///     output[b, i, j, k * channel_multiplier + q] =
  ///       sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
  ///                         filter[di, dj, k, q]
  /// ```
  ///
  /// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  /// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
  ///
  /// - Attrs:
  ///     - strides: 1-D of length 4.  The stride of the sliding window for each dimension
  ///         of `input`.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: Specify the data format of the input and output data. With the
  ///         default format "NHWC", the data is stored in the order of:
  ///             [batch, height, width, channels].
  ///         Alternatively, the format could be "NCHW", the data storage order of:
  ///             [batch, channels, height, width].
  ///     - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
  ///         `input`. If set to k > 1, there will be k-1 skipped cells between each filter
  ///         element on that dimension. The dimension order is determined by the value of
  ///         `data_format`, see above for details. Dilations in the batch and depth
  ///         dimensions must be 1.
  public static func depthwiseConv2dNative<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filter: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
  ) -> Tensor<T> {
    return tf_Conv(
      input, filter, true, strides.map { Int64($0) }, convertPadding(padding),
      [], convertDataFormat(dataFormat),
      dilations.map { Int64($0) })
  }

  /// Computes the gradients of depthwise convolution with respect to the filter.
  ///
  /// - Parameters:
  ///     - input: 4-D with shape based on `data_format`.  For example, if
  ///         `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
  ///         in_width, in_channels]` tensor.
  ///     - filter_sizes: An integer vector representing the tensor shape of `filter`,
  ///         where `filter` is a 4-D
  ///         `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
  ///     - out_backprop: 4-D with shape  based on `data_format`.
  ///         For example, if `data_format` is 'NHWC' then
  ///         out_backprop shape is `[batch, out_height, out_width, out_channels]`.
  ///         Gradients w.r.t. the output of the convolution.
  ///
  /// - Attrs:
  ///     - strides: The stride of the sliding window for each dimension of the input
  ///         of the convolution.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: Specify the data format of the input and output data. With the
  ///         default format "NHWC", the data is stored in the order of:
  ///             [batch, height, width, channels].
  ///         Alternatively, the format could be "NCHW", the data storage order of:
  ///             [batch, channels, height, width].
  ///     - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
  ///         `input`. If set to k > 1, there will be k-1 skipped cells between each filter
  ///         element on that dimension. The dimension order is determined by the value of
  ///         `data_format`, see above for details. Dilations in the batch and depth
  ///         dimensions must be 1.
  ///
  /// - Output output: 4-D with shape
  ///     `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
  ///     the `filter` input of the convolution.
  public static func depthwiseConv2dNativeBackpropFilter<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    filterSizes: Tensor<Int32>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
  ) -> Tensor<T> {
    return tf_ConvBackpropFilter(
      input, filterSizes.scalars.map { Int64($0) },
      outBackprop, true, strides.map { Int64($0) },
      convertPadding(padding), [],
      convertDataFormat(dataFormat), dilations.map { Int64($0) })
  }

  /// Computes the gradients of depthwise convolution with respect to the input.
  ///
  /// - Parameters:
  ///     - input_sizes: An integer vector representing the shape of `input`, based
  ///         on `data_format`.  For example, if `data_format` is 'NHWC' then
  ///          `input` is a 4-D `[batch, height, width, channels]` tensor.
  ///     - filter: 4-D with shape
  ///         `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
  ///     - out_backprop: 4-D with shape  based on `data_format`.
  ///         For example, if `data_format` is 'NHWC' then
  ///         out_backprop shape is `[batch, out_height, out_width, out_channels]`.
  ///         Gradients w.r.t. the output of the convolution.
  ///
  /// - Attrs:
  ///     - strides: The stride of the sliding window for each dimension of the input
  ///         of the convolution.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: Specify the data format of the input and output data. With the
  ///         default format "NHWC", the data is stored in the order of:
  ///             [batch, height, width, channels].
  ///         Alternatively, the format could be "NCHW", the data storage order of:
  ///             [batch, channels, height, width].
  ///     - dilations: 1-D tensor of length 4.  The dilation factor for each dimension of
  ///         `input`. If set to k > 1, there will be k-1 skipped cells between each filter
  ///         element on that dimension. The dimension order is determined by the value of
  ///         `data_format`, see above for details. Dilations in the batch and depth
  ///         dimensions must be 1.
  ///
  /// - Output output: 4-D with shape according to `data_format`.  For example, if
  ///     `data_format` is 'NHWC', output shape is `[batch, in_height,
  ///     in_width, in_channels]`.  Gradient w.r.t. the input of the
  ///     convolution.
  public static func depthwiseConv2dNativeBackpropInput<T: FloatingPoint & TensorFlowScalar>(
    inputSizes: Tensor<Int32>,
    filter: Tensor<T>,
    outBackprop: Tensor<T>,
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat = .nhwc,
    dilations: [Int32] = [1, 1, 1, 1]
  ) -> Tensor<T> {
    return tf_ConvBackpropInput(
      inputSizes.scalars.map { Int64($0) }, filter,
      outBackprop, true, strides.map { Int64($0) },
      convertPadding(padding), [],
      convertDataFormat(dataFormat), dilations.map { Int64($0) })
  }

  /// Returns the diagonal part of the tensor.
  ///
  /// This operation returns a tensor with the `diagonal` part
  /// of the `input`. The `diagonal` part is computed as follows:
  ///
  /// Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
  /// tensor of rank `k` with dimensions `[D1,..., Dk]` where:
  ///
  /// `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.
  ///
  /// For example:
  ///
  /// ```
  /// # 'input' is [[1, 0, 0, 0]
  ///               [0, 2, 0, 0]
  ///               [0, 0, 3, 0]
  ///               [0, 0, 0, 4]]
  ///
  /// tf.diag_part(input) ==> [1, 2, 3, 4]
  /// ```
  ///
  /// - Parameter input: Rank k tensor where k is even and not zero.
  ///
  /// - Output diagonal: The extracted diagonal.
  public static func diagPart<T: TensorFlowNumeric>(
    _ input: Tensor<T>
  ) -> Tensor<T> {
    let shape = input.shape.dimensions.map { Int64($0) }
    if shape.count != 2 {
      precondition(shape.count != 0, "Cannot take diagonal of a scalar")
      let nOutDims = shape.count / 2
      precondition(nOutDims * 2 == shape.count, "Must be even rank")
      let outputDims = shape[0..<nOutDims]
      precondition(outputDims == shape[nOutDims..<shape.count], "Must be even rank")
      let flatDims = outputDims.reduce(1, *)
      return resize_value(
        _RawXLA.diagPart(
          resize_value(input, dims: [flatDims, flatDims])
        ),
        dims: [Int64](outputDims))
    }
    return diagonal_value(input, offset: 0, dim1: 0, dim2: 1)
  }

  /// Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.
  ///
  /// See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
  /// ](http://arxiv.org/abs/1511.07289)
  public static func elu<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>
  ) -> Tensor<T> {
    _RawXLA.select(
      condition: _RawXLA.greater(features, _RawXLA.zerosLike(features)),
      t: features,
      e: _RawXLA.expm1(features))
  }

  /// Computes gradients for the exponential linear (Elu) operation.
  ///
  /// - Parameters:
  ///     - gradients: The backpropagated gradients to the corresponding Elu operation.
  ///     - outputs: The outputs of the corresponding Elu operation.
  ///
  /// - Output backprops: The gradients: `gradients * (outputs + 1)` if outputs < 0,
  ///     `gradients` otherwise.
  public static func eluGrad<T: FloatingPoint & TensorFlowScalar>(
    gradients: Tensor<T>,
    outputs: Tensor<T>
  ) -> Tensor<T> {
    checkSameDevice(gradients, outputs)
    checkSamePrecision(gradients, outputs)
    return _RawXLA.select(
      condition: _RawXLA.greater(outputs, _RawXLA.zerosLike(outputs)),
      t: gradients,
      e: _RawXLA.mul(gradients, _RawXLA.addV2(outputs, _RawXLA.onesLike(outputs))))
  }

  /// Returns the truth value of (x == y) element-wise.
  ///
  /// *NOTE*: `Equal` supports broadcasting. More about broadcasting
  /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  public static func equal<T: TensorFlowScalar>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    incompatibleShapeError: Bool = true
  ) -> Tensor<Bool> {
    precondition(incompatibleShapeError)
    return eq(x, y)
  }

  /// Inserts a dimension of 1 into a tensor's shape.
  ///
  /// Given a tensor `input`, this operation inserts a dimension of 1 at the
  /// dimension index `axis` of `input`'s shape. The dimension index `axis` starts at
  /// zero; if you specify a negative number for `axis` it is counted backward from
  /// the end.
  ///
  /// This operation is useful if you want to add a batch dimension to a single
  /// element. For example, if you have a single image of shape `[height, width,
  /// channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
  /// which will make the shape `[1, height, width, channels]`.
  ///
  /// Other examples:
  ///
  /// ```
  /// # 't' is a tensor of shape [2]
  /// shape(expand_dims(t, 0)) ==> [1, 2]
  /// shape(expand_dims(t, 1)) ==> [2, 1]
  /// shape(expand_dims(t, -1)) ==> [2, 1]
  ///
  /// # 't2' is a tensor of shape [2, 3, 5]
  /// shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
  /// shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
  /// shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
  /// ```
  ///
  /// This operation requires that:
  ///
  /// `-1-input.dims() <= dim <= input.dims()`
  ///
  /// This operation is related to `squeeze()`, which removes dimensions of
  /// size 1.
  ///
  /// - Parameter dim: 0-D (scalar). Specifies the dimension index at which to
  ///     expand the shape of `input`. Must be in the range
  ///     `[-rank(input) - 1, rank(input)]`.
  ///
  /// - Output output: Contains the same data as `input`, but its shape has an additional
  ///     dimension of size 1 added.
  public static func expandDims<
    T: TensorFlowScalar,
    Tdim: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    dim: Tensor<Tdim>
  ) -> Tensor<T> {
    var dims = input.shape.dimensions.map(Int64.init)
    var dim = Int(dim.scalarized())
    if dim < 0 { dim += dims.count + 1 }
    dims.insert(1, at: dim)
    return resize_value(input, dims: dims)
  }

  /// Creates a tensor filled with a scalar value.
  ///
  /// This operation creates a tensor of shape `dims` and fills it with `value`.
  ///
  /// For example:
  ///
  /// ```
  /// # Output tensor has shape [2, 3].
  /// fill([2, 3], 9) ==> [[9, 9, 9]
  ///                      [9, 9, 9]]
  /// ```
  ///
  /// `tf.fill` differs from `tf.constant` in a few ways:
  ///
  /// *   `tf.fill` only supports scalar contents, whereas `tf.constant` supports
  ///     Tensor values.
  /// *   `tf.fill` creates an Op in the computation graph that constructs the actual
  ///     Tensor value at runtime. This is in contrast to `tf.constant` which embeds
  ///     the entire Tensor into the graph with a `Const` node.
  /// *   Because `tf.fill` evaluates at graph runtime, it supports dynamic shapes
  ///     based on other runtime Tensors, unlike `tf.constant`.
  ///
  /// - Parameters:
  ///     - dims: 1-D. Represents the shape of the output tensor.
  ///     - value: 0-D (scalar). Value to fill the returned tensor.
  ///
  ///         @compatibility(numpy)
  ///         Equivalent to np.full
  ///         @end_compatibility
  public static func fill<
    T: TensorFlowScalar,
    IndexType: TensorFlowIndex
  >(
    dims: Tensor<IndexType>,
    value: Tensor<T>
  ) -> Tensor<T> {
    return broadcastTo(value, dims: dims.scalars.map { Int64($0) })
  }
  static func fullLike<T: TensorFlowScalar>(_ value: XLAScalarType, _ x: Tensor<T>)
    -> Tensor<T>
  {
    return broadcastTo(scalarLike(value, x), dims: x.shape.dimensions.map(Int64.init))
  }
  static func scalarLike<T: TensorFlowScalar>(_ value: XLAScalarType, _ x: Tensor<T>)
    -> Tensor<T>
  {
    let result = Tensor<T>(
      _xlaHandle: XLATensor_makeScalar(value.xlaScalar, T.xlaTensorScalarType, x.device.cdevice))
    return x.isReducedPrecision ? result.toReducedPrecision : result
  }

  /// Gather slices from `params` according to `indices`.
  ///
  /// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
  /// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
  ///
  /// ```python
  ///     # Scalar indices
  ///     output[:, ..., :] = params[indices, :, ... :]
  ///
  ///     # Vector indices
  ///     output[i, :, ..., :] = params[indices[i], :, ... :]
  ///
  ///     # Higher rank indices
  ///     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
  /// ```
  ///
  /// If `indices` is a permutation and `len(indices) == params.shape[0]` then
  /// this operation will permute `params` accordingly.
  ///
  /// `validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
  /// `indices` are always validated to be within range. If assigned to GPU,
  /// out-of-bound indices result in safe but unspecified behavior, which may include
  /// raising an error.
  ///
  /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  /// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
  /// </div>
  public static func gather<
    Tparams: TensorFlowScalar,
    Tindices: TensorFlowIndex
  >(
    params: Tensor<Tparams>,
    indices: Tensor<Tindices>,
    validateIndices: Bool = true
  ) -> Tensor<Tparams> {
    // TODO(b/145948524): Honor validateIndices.
    gatherV2(params: params, indices: indices, axis: Tensor<Int32>(0))
  }

  /// Gather slices from `params` axis `axis` according to `indices`.
  ///
  /// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
  /// Produces an output tensor with shape `params.shape[:axis] + indices.shape +
  /// params.shape[axis + 1:]` where:
  ///
  /// ```python
  ///     # Scalar indices (output is rank(params) - 1).
  ///     output[a_0, ..., a_n, b_0, ..., b_n] =
  ///       params[a_0, ..., a_n, indices, b_0, ..., b_n]
  ///
  ///     # Vector indices (output is rank(params)).
  ///     output[a_0, ..., a_n, i, b_0, ..., b_n] =
  ///       params[a_0, ..., a_n, indices[i], b_0, ..., b_n]
  ///
  ///     # Higher rank indices (output is rank(params) + rank(indices) - 1).
  ///     output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
  ///       params[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
  /// ```
  ///
  /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  /// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
  /// </div>
  ///
  /// Note that on CPU, if an out of bound index is found, an error is returned.
  /// On GPU, if an out of bound index is found, a 0 is stored in the
  /// corresponding output value.
  ///
  /// See also `tf.batch_gather` and `tf.gather_nd`.
  ///
  /// - Parameters:
  ///     - params: The tensor from which to gather values. Must be at least rank
  ///         `axis + 1`.
  ///     - indices: Index tensor. Must be in range `[0, params.shape[axis])`.
  ///     - axis: The axis in `params` to gather `indices` from. Defaults to the first
  ///         dimension. Supports negative indexes.
  ///
  /// - Output output: Values from `params` gathered from indices given by `indices`, with
  ///     shape `params.shape[:axis] + indices.shape + params.shape[axis + 1:]`.
  public static func gatherV2<
    Tparams: TensorFlowScalar,
    Tindices: TensorFlowIndex,
    Taxis: TensorFlowIndex
  >(
    params: Tensor<Tparams>,
    indices: Tensor<Tindices>,
    axis: Tensor<Taxis>,
    batchDims: Int64 = 0
  ) -> Tensor<Tparams> {
    precondition(batchDims == 0)
    let canonicalAxis = canonicalDims(axis.scalars.map { Int64($0) }, Int64(params.rank)).first!
    return gather(
      params, indices: Tensor<Tindices>(stacking: [indices], alongAxis: indices.rank),
      start_dim: canonicalAxis)
  }

  /// Computes the inverse permutation of a tensor.
  ///
  /// This operation computes the inverse of an index permutation. It takes a 1-D
  /// integer tensor `x`, which represents the indices of a zero-based array, and
  /// swaps each value with its index position. In other words, for an output tensor
  /// `y` and an input tensor `x`, this operation computes the following:
  ///
  /// `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`
  ///
  /// The values must include 0. There can be no duplicate values or negative values.
  ///
  /// For example:
  ///
  /// ```
  /// # tensor `x` is [3, 4, 0, 2, 1]
  /// invert_permutation(x) ==> [2, 4, 3, 0, 1]
  /// ```
  ///
  /// - Parameter x: 1-D.
  ///
  /// - Output y: 1-D.
  @inlinable @inline(__always)
  public static func invertPermutation<T: TensorFlowIndex>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    if x.rank != 1 {
      fatalError("Input should be rank 1, got \(x.rank)")
    }
    let scalars = invertPermutationArray(x.scalars)
    return Tensor<T>(shape: [scalars.count], scalars: scalars, on: x.device)
  }

  /// Computes rectified linear: `max(features, features * alpha)`.
  public static func leakyRelu<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>,
    alpha: Double = 0.2
  ) -> Tensor<T> {
    return _RawXLA.select(
      condition: _RawXLA.greater(features, _RawXLA.zerosLike(features)),
      t: features,
      e: _RawXLA.mul(features, scalarLike(alpha, features)))
  }

  /// Computes rectified linear gradients for a LeakyRelu operation.
  ///
  /// - Parameters:
  ///     - gradients: The backpropagated gradients to the corresponding LeakyRelu operation.
  ///     - features: The features passed as input to the corresponding LeakyRelu operation,
  ///         OR the outputs of that operation (both work equivalently).
  ///
  /// - Output backprops: `gradients * (features > 0) + alpha * gradients * (featurs <= 0)`.
  public static func leakyReluGrad<T: FloatingPoint & TensorFlowScalar>(
    gradients: Tensor<T>,
    features: Tensor<T>,
    alpha: Double = 0.2
  ) -> Tensor<T> {
    checkSameDevice(gradients, features)
    checkSamePrecision(gradients, features)
    return _RawXLA.select(
      condition: _RawXLA.greater(features, _RawXLA.zerosLike(features)),
      t: gradients,
      e: _RawXLA.mul(
        gradients,
        fullLike(alpha, gradients)))
  }

  /// Generates values in an interval.
  ///
  /// A sequence of `num` evenly-spaced values are generated beginning at `start`.
  /// If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
  /// so that the last one is exactly `stop`.
  ///
  /// For example:
  ///
  /// ```
  /// tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
  /// ```
  ///
  /// - Parameters:
  ///     - start: 0-D tensor. First entry in the range.
  ///     - stop: 0-D tensor. Last entry in the range.
  ///     - num: 0-D tensor. Number of values to generate.
  ///
  /// - Output output: 1-D. The generated values.
  public static func linSpace<
    T: FloatingPoint & TensorFlowScalar,
    Tidx: TensorFlowIndex
  >(
    start: Tensor<T>,
    stop: Tensor<T>,
    num: Tensor<Tidx>,
    device: Device
  ) -> Tensor<T> {
    checkSameDevice(start, stop)
    checkSameDevice(num.device, start.device)
    checkSamePrecision(start, stop)
    let numScalarized = num.scalarized()
    precondition(numScalarized > 0, "num should be > 0 for linSpace.")
    if numScalarized == 1 { return start }
    let startScalar: T = start.scalarized()
    let stopScalar: T = stop.scalarized()
    var linspace = Tensor<T>(
      _xla: XLATensor.linspace(
        startScalar, stopScalar, Int64(numScalarized), T.xlaTensorScalarType, device))
    if start.isReducedPrecision {
      linspace = linspace.toReducedPrecision
    }
    return linspace
  }

  public static func linSpace<
    T: FloatingPoint & TensorFlowScalar,
    Tidx: TensorFlowIndex
  >(
    start: Tensor<T>,
    stop: Tensor<T>,
    num: Tensor<Tidx>
  ) -> Tensor<T> {
    let device = start.device
    return linSpace(start: start, stop: stop, num: num, device: device)
  }

  /// Computes log softmax activations.
  ///
  /// For each batch `i` and class `j` we have
  ///
  ///     logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
  ///
  /// - Parameter logits: 2-D with shape `[batch_size, num_classes]`.
  ///
  /// - Output logsoftmax: Same shape as `logits`.
  public static func logSoftmax<T: FloatingPoint & TensorFlowScalar>(
    logits: Tensor<T>
  ) -> Tensor<T> {
    return logSoftmax(logits, dim: -1)
  }

  /// Multiply the matrix "a" by the matrix "b".
  ///
  /// The inputs must be two-dimensional matrices and the inner dimension of
  /// "a" (after being transposed if transpose_a is true) must match the
  /// outer dimension of "b" (after being transposed if transposed_b is
  /// true).
  ///
  /// *Note*: The default kernel implementation for MatMul on GPUs uses
  /// cublas.
  ///
  /// - Attrs:
  ///     - transpose_a: If true, "a" is transposed before multiplication.
  ///     - transpose_b: If true, "b" is transposed before multiplication.
  public static func matMul<T: TensorFlowNumeric>(
    _ a: Tensor<T>,
    _ b: Tensor<T>,
    transposeA: Bool = false,
    transposeB: Bool = false
  ) -> Tensor<T> {
    return matMul(
      (transposeA ? permute(a, dims: [1, 0]) : a),
      (transposeB ? permute(b, dims: [1, 0]) : b))
  }

  /// Computes the maximum of elements across dimensions of a tensor.
  ///
  /// Reduces `input` along the dimensions given in `axis`. Unless
  /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  /// `axis`. If `keep_dims` is true, the reduced dimensions are
  /// retained with length 1.
  ///
  /// - Parameters:
  ///     - input: The tensor to reduce.
  ///     - reduction_indices: The dimensions to reduce. Must be in the range
  ///         `[-rank(input), rank(input))`.
  ///
  /// - Attr keep_dims: If true, retain reduced dimensions with length 1.
  ///
  /// - Output output: The reduced tensor.
  public static func max<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
  ) -> Tensor<T> {
    var out = input
    var dims = canonicalDims(reductionIndices.scalars.map { Int64($0) }, Int64(input.rank))
    // Go through dims in reverse order.
    dims.sort(by: >)
    for dim in dims {
      out = max(out, dim: dim, keepDim: keepDims)
    }
    return out
  }

  /// Performs 3D max pooling on the input.
  ///
  /// - Parameter input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
  ///
  /// - Attrs:
  ///     - ksize: 1-D tensor of length 5. The size of the window for each dimension of
  ///         the input tensor. Must have `ksize[0] = ksize[4] = 1`.
  ///     - strides: 1-D tensor of length 5. The stride of the sliding window for each
  ///         dimension of `input`. Must have `strides[0] = strides[4] = 1`.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: The data format of the input and output data. With the
  ///         default format "NDHWC", the data is stored in the order of:
  ///             [batch, in_depth, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCDHW", the data storage order is:
  ///             [batch, in_channels, in_depth, in_height, in_width].
  ///
  /// - Output output: The max pooled output tensor.
  public static func maxPool3D<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc
  ) -> Tensor<T> {
    Tensor(
      _xla: XLATensor.maxpool(
        input.xlaTensor, ksize.map { Int64($0) }, strides.map { Int64($0) },
        convertPadding(padding), convertDataFormat1(dataFormat)))
  }

  /// Computes gradients of max pooling function.
  ///
  /// - Parameters:
  ///     - orig_input: The original input tensor.
  ///     - orig_output: The original output tensor.
  ///     - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
  ///
  /// - Attrs:
  ///     - ksize: 1-D tensor of length 5. The size of the window for each dimension of
  ///         the input tensor. Must have `ksize[0] = ksize[4] = 1`.
  ///     - strides: 1-D tensor of length 5. The stride of the sliding window for each
  ///         dimension of `input`. Must have `strides[0] = strides[4] = 1`.
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: The data format of the input and output data. With the
  ///         default format "NDHWC", the data is stored in the order of:
  ///             [batch, in_depth, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCDHW", the data storage order is:
  ///             [batch, in_channels, in_depth, in_height, in_width].
  public static func maxPool3DGrad<
    T: FloatingPoint & TensorFlowScalar,
    Tinput: FloatingPoint & TensorFlowScalar
  >(
    origInput: Tensor<Tinput>,
    origOutput: Tensor<Tinput>,
    grad: Tensor<T>,
    ksize: [Int32],
    strides: [Int32],
    padding: Padding,
    dataFormat: DataFormat1 = .ndhwc
  ) -> Tensor<T> {
    checkSameDevice(origInput, origOutput)
    checkSameDevice(grad.device, origInput.device)
    checkSamePrecision(origInput, origOutput)
    checkSamePrecision(grad.isReducedPrecision, origInput.isReducedPrecision)
    return Tensor(
      _xla: XLATensor.maxpool_grad(
        origInput.xlaTensor, grad.xlaTensor, ksize.map { Int64($0) },
        strides.map { Int64($0) }, convertPadding(padding)))
  }

  /// Computes gradients of the maxpooling function.
  ///
  /// - Parameters:
  ///     - orig_input: The original input tensor.
  ///     - orig_output: The original output tensor.
  ///     - grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
  ///     - ksize: The size of the window for each dimension of the input tensor.
  ///     - strides: The stride of the sliding window for each dimension of the
  ///         input tensor.
  ///
  /// - Attrs:
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: Specify the data format of the input and output data. With the
  ///         default format "NHWC", the data is stored in the order of:
  ///             [batch, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCHW", the data storage order of:
  ///             [batch, in_channels, in_height, in_width].
  ///
  /// - Output output: Gradients w.r.t. the input to `max_pool`.
  public static func maxPoolGradV2<T: TensorFlowNumeric>(
    origInput: Tensor<T>,
    origOutput: Tensor<T>,
    grad: Tensor<T>,
    ksize: [Int64],
    strides: [Int64],
    padding: Padding,
    dataFormat: DataFormat = .nhwc
  ) -> Tensor<T> {
    checkSameDevice(origInput, origOutput, grad)
    checkSamePrecision(origInput, origOutput, grad)
    return Tensor(
      _xla: XLATensor.maxpool_grad(
        origInput.xlaTensor, grad.xlaTensor, ksize,
        strides, convertPadding(padding)))
  }
  public static func maxPoolGradV2<T: TensorFlowNumeric>(
    origInput: Tensor<T>,
    origOutput: Tensor<T>,
    grad: Tensor<T>,
    ksize: Tensor<Int32>,
    strides: Tensor<Int32>,
    padding: Padding,
    dataFormat: DataFormat = .nhwc
  ) -> Tensor<T> {
    checkSameDevice(origInput, origOutput, grad)
    checkSamePrecision(origInput, origOutput, grad)
    return Tensor(
      _xla: XLATensor.maxpool_grad(
        origInput.xlaTensor, grad.xlaTensor, ksize.scalars.map { Int64($0) },
        strides.scalars.map { Int64($0) }, convertPadding(padding)))
  }

  /// Performs max pooling on the input.
  ///
  /// - Parameters:
  ///     - input: 4-D input to pool over.
  ///     - ksize: The size of the window for each dimension of the input tensor.
  ///     - strides: The stride of the sliding window for each dimension of the
  ///         input tensor.
  ///
  /// - Attrs:
  ///     - padding: The type of padding algorithm to use.
  ///     - data_format: Specify the data format of the input and output data. With the
  ///         default format "NHWC", the data is stored in the order of:
  ///             [batch, in_height, in_width, in_channels].
  ///         Alternatively, the format could be "NCHW", the data storage order of:
  ///             [batch, in_channels, in_height, in_width].
  ///
  /// - Output output: The max pooled output tensor.
  public static func maxPoolV2<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    ksize: [Int64],
    strides: [Int64],
    padding: Padding,
    dataFormat: DataFormat4 = .nhwc
  ) -> Tensor<T> {
    Tensor(
      _xla: XLATensor.maxpool(
        input.xlaTensor, ksize, strides,
        convertPadding(padding), convertDataFormat4(dataFormat)))
  }
  public static func maxPoolV2<T: TensorFlowNumeric>(
    _ input: Tensor<T>,
    ksize: Tensor<Int32>,
    strides: Tensor<Int32>,
    padding: Padding,
    dataFormat: DataFormat4 = .nhwc
  ) -> Tensor<T> {
    Tensor(
      _xla: XLATensor.maxpool(
        input.xlaTensor, ksize.scalars.map { Int64($0) }, strides.scalars.map { Int64($0) },
        convertPadding(padding), convertDataFormat4(dataFormat)))
  }

  /// Computes the mean of elements across dimensions of a tensor.
  ///
  /// Reduces `input` along the dimensions given in `axis`. Unless
  /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  /// `axis`. If `keep_dims` is true, the reduced dimensions are
  /// retained with length 1.
  ///
  /// - Parameters:
  ///     - input: The tensor to reduce.
  ///     - reduction_indices: The dimensions to reduce. Must be in the range
  ///         `[-rank(input), rank(input))`.
  ///
  /// - Attr keep_dims: If true, retain reduced dimensions with length 1.
  ///
  /// - Output output: The reduced tensor.
  public static func mean<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
  ) -> Tensor<T> {
    mean(input, reductionIndices: reductionIndices.scalars.map { Int64($0) }, keepDims: keepDims)
  }

  /// Computes the minimum of elements across dimensions of a tensor.
  ///
  /// Reduces `input` along the dimensions given in `axis`. Unless
  /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  /// `axis`. If `keep_dims` is true, the reduced dimensions are
  /// retained with length 1.
  ///
  /// - Parameters:
  ///     - input: The tensor to reduce.
  ///     - reduction_indices: The dimensions to reduce. Must be in the range
  ///         `[-rank(input), rank(input))`.
  ///
  /// - Attr keep_dims: If true, retain reduced dimensions with length 1.
  ///
  /// - Output output: The reduced tensor.
  public static func min<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
  ) -> Tensor<T> {
    var out = input
    let rank = Int64(input.rank)
    var dims = reductionIndices.scalars.map { (originalDim: Tidx) -> Int64 in
      var dim = Int64(originalDim)
      if dim < 0 { dim += rank }
      return dim
    }
    // Go through dims in reverse order.
    dims.sort(by: >)
    for dim in dims {
      out = min(out, dim: dim, keepDim: keepDims)
    }
    return out
  }

  /// Pads a tensor with mirrored values.
  ///
  /// This operation pads a `input` with mirrored values according to the `paddings`
  /// you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
  /// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  /// how many values to add before the contents of `input` in that dimension, and
  /// `paddings[D, 1]` indicates how many values to add after the contents of `input`
  /// in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
  /// than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
  /// (if false, respectively).
  ///
  /// The padded size of each dimension D of the output is:
  ///
  /// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
  ///
  /// For example:
  ///
  /// ```
  /// # 't' is [[1, 2, 3], [4, 5, 6]].
  /// # 'paddings' is [[1, 1]], [2, 2]].
  /// # 'mode' is SYMMETRIC.
  /// # rank of 't' is 2.
  /// pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
  ///                       [2, 1, 1, 2, 3, 3, 2]
  ///                       [5, 4, 4, 5, 6, 6, 5]
  ///                       [5, 4, 4, 5, 6, 6, 5]]
  /// ```
  ///
  /// - Parameters:
  ///     - input: The input tensor to be padded.
  ///     - paddings: A two-column matrix specifying the padding sizes. The number of
  ///         rows must be the same as the rank of `input`.
  ///
  /// - Attr mode: Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
  ///     do not include the borders, while in symmetric mode the padded regions
  ///     do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
  ///     is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
  ///     it is `[1, 2, 3, 3, 2]` in symmetric mode.
  ///
  /// - Output output: The padded tensor.
  public static func mirrorPad<
    T: TensorFlowScalar,
    Tpaddings: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    paddings: Tensor<Tpaddings>,
    mode: Mode5
  ) -> Tensor<T> {
    let linearizedPaddings = paddings.scalars.map { Int64($0) }
    return tf_MirrorPad(input, reversedPaddings(linearizedPaddings), convertMirrorPadMode(mode))
  }

  /// Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.
  ///
  /// This operation folds the padded areas of `input` by `MirrorPad` according to the
  /// `paddings` you specify. `paddings` must be the same as `paddings` argument
  /// given to the corresponding `MirrorPad` op.
  ///
  /// The folded size of each dimension D of the output is:
  ///
  /// `input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`
  ///
  /// For example:
  ///
  /// ```
  /// # 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
  /// # 'paddings' is [[0, 1]], [0, 1]].
  /// # 'mode' is SYMMETRIC.
  /// # rank of 't' is 2.
  /// pad(t, paddings) ==> [[ 1,  5]
  ///                       [11, 28]]
  /// ```
  ///
  /// - Parameters:
  ///     - input: The input tensor to be folded.
  ///     - paddings: A two-column matrix specifying the padding sizes. The number of
  ///         rows must be the same as the rank of `input`.
  ///
  /// - Attr mode: The mode used in the `MirrorPad` op.
  ///
  /// - Output output: The folded tensor.
  public static func mirrorPadGrad<
    T: TensorFlowScalar,
    Tpaddings: TensorFlowIndex
  >(
    _ grad: Tensor<T>,
    paddings: Tensor<Tpaddings>,
    mode: Mode5
  ) -> Tensor<T> {
    let linearizedPaddings = paddings.scalars.map { Int64($0) }
    let inputDimensions = grad.shape.dimensions.indices.map { (dim: Int) -> Int64 in
      let totalPadding = linearizedPaddings[2 * dim] + linearizedPaddings[2 * dim + 1]
      return Int64(grad.shape.dimensions[dim]) - totalPadding
    }
    return tf_MirrorPadGrad(
      grad, inputDimensions, reversedPaddings(linearizedPaddings),
      convertMirrorPadMode(mode))
  }

  /// Returns the truth value of (x != y) element-wise.
  ///
  /// *NOTE*: `NotEqual` supports broadcasting. More about broadcasting
  /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  public static func notEqual<T: TensorFlowScalar>(
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    incompatibleShapeError: Bool
  ) -> Tensor<Bool> {
    precondition(incompatibleShapeError)
    return notEqual(x, y)
  }

  /// Returns a one-hot tensor.
  ///
  /// The locations represented by indices in `indices` take value `on_value`,
  /// while all other locations take value `off_value`.
  ///
  /// If the input `indices` is rank `N`, the output will have rank `N+1`,
  /// The new axis is created at dimension `axis` (default: the new axis is
  /// appended at the end).
  ///
  /// If `indices` is a scalar the output shape will be a vector of length `depth`.
  ///
  /// If `indices` is a vector of length `features`, the output shape will be:
  /// ```
  ///   features x depth if axis == -1
  ///   depth x features if axis == 0
  /// ```
  ///
  /// If `indices` is a matrix (batch) with shape `[batch, features]`,
  /// the output shape will be:
  /// ```
  ///   batch x features x depth if axis == -1
  ///   batch x depth x features if axis == 1
  ///   depth x batch x features if axis == 0
  /// ```
  ///
  ///
  /// Examples
  /// =========
  ///
  /// Suppose that
  /// ```
  ///   indices = [0, 2, -1, 1]
  ///   depth = 3
  ///   on_value = 5.0
  ///   off_value = 0.0
  ///   axis = -1
  /// ```
  ///
  /// Then output is `[4 x 3]`:
  /// ```
  /// output =
  ///   [5.0 0.0 0.0]  // one_hot(0)
  ///   [0.0 0.0 5.0]  // one_hot(2)
  ///   [0.0 0.0 0.0]  // one_hot(-1)
  ///   [0.0 5.0 0.0]  // one_hot(1)
  /// ```
  ///
  /// Suppose that
  /// ```
  ///   indices = [0, 2, -1, 1]
  ///   depth = 3
  ///   on_value = 0.0
  ///   off_value = 3.0
  ///   axis = 0
  /// ```
  ///
  /// Then output is `[3 x 4]`:
  /// ```
  /// output =
  ///   [0.0 3.0 3.0 3.0]
  ///   [3.0 3.0 3.0 0.0]
  ///   [3.0 3.0 3.0 3.0]
  ///   [3.0 0.0 3.0 3.0]
  /// //  ^                one_hot(0)
  /// //      ^            one_hot(2)
  /// //          ^        one_hot(-1)
  /// //              ^    one_hot(1)
  /// ```
  ///
  /// Suppose that
  /// ```
  ///   indices = [[0, 2], [1, -1]]
  ///   depth = 3
  ///   on_value = 1.0
  ///   off_value = 0.0
  ///   axis = -1
  /// ```
  ///
  /// Then output is `[2 x 2 x 3]`:
  /// ```
  /// output =
  ///   [
  ///     [1.0, 0.0, 0.0]  // one_hot(0)
  ///     [0.0, 0.0, 1.0]  // one_hot(2)
  ///   ][
  ///     [0.0, 1.0, 0.0]  // one_hot(1)
  ///     [0.0, 0.0, 0.0]  // one_hot(-1)
  ///   ]
  /// ```
  ///
  /// - Parameters:
  ///     - indices: A tensor of indices.
  ///     - depth: A scalar defining the depth of the one hot dimension.
  ///     - on_xla: A scalar defining the value to fill in output when `indices[j] = i`.
  ///     - off_xla: A scalar defining the value to fill in output when `indices[j] != i`.
  ///
  /// - Attr axis: The axis to fill (default: -1, a new inner-most axis).
  ///
  /// - Output output: The one-hot tensor.
  public static func oneHot<
    T: TensorFlowScalar,
    Ti: TensorFlowInteger
  >(
    indices: Tensor<Ti>,
    depth: Int64,
    onValue: Tensor<T>,
    offValue: Tensor<T>,
    axis: Int64 = -1
  ) -> Tensor<T> {
    return tf_OneHot(indices, onValue, offValue, depth, axis)
  }
  public static func oneHot<
    T: TensorFlowScalar,
    Ti: TensorFlowInteger
  >(
    indices: Tensor<Ti>,
    depth: Tensor<Int32>,
    onValue: Tensor<T>,
    offValue: Tensor<T>,
    axis: Int64 = -1
  ) -> Tensor<T> {
    return tf_OneHot(indices, onValue, offValue, Int64(depth.scalarized()), axis)
  }

  /// Returns a tensor of ones with the same shape and type as x.
  ///
  /// - Parameter x: a tensor of type T.
  ///
  /// - Output y: a tensor of the same shape and type as x but filled with ones.
  public static func onesLike<T: TensorFlowScalar>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    return fullLike(1, x)
  }

  /// Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.
  ///
  /// Packs the `N` tensors in `values` into a tensor with rank one higher than each
  /// tensor in `values`, by packing them along the `axis` dimension.
  /// Given a list of tensors of shape `(A, B, C)`;
  ///
  /// if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
  /// if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
  /// Etc.
  ///
  /// For example:
  ///
  /// ```
  /// # 'x' is [1, 4]
  /// # 'y' is [2, 5]
  /// # 'z' is [3, 6]
  /// pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
  /// pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
  /// ```
  ///
  /// This is the opposite of `unpack`.
  ///
  /// - Parameter values: Must be of same shape and type.
  ///
  /// - Attr axis: Dimension along which to pack.  Negative values wrap around, so the
  ///     valid range is `[-(R+1), R+1)`.
  ///
  /// - Output output: The packed tensor.
  public static func pack<T: TensorFlowScalar>(
    _ values: [Tensor<T>],
    axis: Int64 = 0
  ) -> Tensor<T> {
    checkSameDevice(values)
    checkSamePrecision(values)
    return stack(values, dim: axis)
  }

  /// Pads a tensor with zeros.
  ///
  /// This operation pads a `input` with zeros according to the `paddings` you
  /// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
  /// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  /// how many zeros to add before the contents of `input` in that dimension, and
  /// `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
  /// in that dimension.
  ///
  /// The padded size of each dimension D of the output is:
  ///
  /// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
  ///
  /// For example:
  ///
  /// ```
  /// # 't' is [[1, 1], [2, 2]]
  /// # 'paddings' is [[1, 1], [2, 2]]
  /// # rank of 't' is 2
  /// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
  ///                       [0, 0, 1, 1, 0, 0]
  ///                       [0, 0, 2, 2, 0, 0]
  ///                       [0, 0, 0, 0, 0, 0]]
  /// ```
  ///
  public static func pad<
    T: TensorFlowScalar,
    Tpaddings: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    paddings: Tensor<Tpaddings>
  ) -> Tensor<T> {
    pad(input, paddings: paddings.scalars.map { Int($0) })
  }

  public static func pad<
    T: TensorFlowScalar
  >(
    _ input: Tensor<T>,
    paddings: [Int]
  ) -> Tensor<T> {
    constant_pad_nd(input, reversedPaddings(paddings.map { Int64($0) }), 0)
  }

  /// Pads a tensor.
  ///
  /// This operation pads `input` according to the `paddings` and `constant_values`
  /// you specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
  /// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  /// how many padding values to add before the contents of `input` in that dimension,
  /// and `paddings[D, 1]` indicates how many padding values to add after the contents
  /// of `input` in that dimension. `constant_values` is a scalar tensor of the same
  /// type as `input` that indicates the value to use for padding `input`.
  ///
  /// The padded size of each dimension D of the output is:
  ///
  /// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
  ///
  /// For example:
  ///
  /// ```
  /// # 't' is [[1, 1], [2, 2]]
  /// # 'paddings' is [[1, 1], [2, 2]]
  /// # 'constant_values' is 0
  /// # rank of 't' is 2
  /// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
  ///                       [0, 0, 1, 1, 0, 0]
  ///                       [0, 0, 2, 2, 0, 0]
  ///                       [0, 0, 0, 0, 0, 0]]
  /// ```
  public static func padV2<
    T: TensorFlowScalar,
    Tpaddings: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    paddings: Tensor<Tpaddings>,
    constantValues: Tensor<T>
  ) -> Tensor<T> {
    let linearizedPaddings = paddings.scalars.map { Int64($0) }
    return constant_pad_nd(
      input, reversedPaddings(linearizedPaddings), constantValues.scalarized())
  }

  public static func physicalCast<T: TensorFlowScalar, R: TensorFlowScalar>(
    _ input: Tensor<T>, destType: R.Type
  ) -> Tensor<T> {
    physicalCast(input, destType: destType.xlaTensorScalarType)
  }

  /// Computes the product of elements across dimensions of a tensor.
  ///
  /// Reduces `input` along the dimensions given in `axis`. Unless
  /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  /// `axis`. If `keep_dims` is true, the reduced dimensions are
  /// retained with length 1.
  ///
  /// - Parameters:
  ///     - input: The tensor to reduce.
  ///     - reduction_indices: The dimensions to reduce. Must be in the range
  ///         `[-rank(input), rank(input))`.
  ///
  /// - Attr keep_dims: If true, retain reduced dimensions with length 1.
  ///
  /// - Output output: The reduced tensor.
  public static func prod<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
  ) -> Tensor<T> {
    return prod(
      input, reductionIndices: reductionIndices.scalars.map { Int64($0) }, keepDims: keepDims)
  }

  /// Creates a sequence of numbers.
  ///
  /// This operation creates a sequence of numbers that begins at `start` and
  /// extends by increments of `delta` up to but not including `limit`.
  ///
  /// For example:
  ///
  /// ```
  /// # 'start' is 3
  /// # 'limit' is 18
  /// # 'delta' is 3
  /// tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
  /// ```
  ///
  /// - Parameters:
  ///     - start: 0-D (scalar). First entry in the sequence.
  ///     - limit: 0-D (scalar). Upper limit of sequence, exclusive.
  ///     - delta: 0-D (scalar). Optional. Default is 1. Number that increments `start`.
  ///
  /// - Output output: 1-D.
  public static func range<Tidx: TensorFlowNumeric>(
    start: Tensor<Tidx>,
    limit: Tensor<Tidx>,
    delta: Tensor<Tidx>
  ) -> Tensor<Tidx> {
    checkSameDevice(start, limit)
    checkSameDevice(start, delta)
    checkSamePrecision(start, limit, delta)
    return Tensor<Tidx>(
      _xla: XLATensor.arange(
        start.scalarized(), limit.scalarized(), delta.scalarized(), Tidx.xlaTensorScalarType,
        start.device))
  }

  /// Returns the rank of a tensor.
  ///
  /// This operation returns an integer representing the rank of `input`.
  ///
  /// For example:
  ///
  /// ```
  /// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  /// # shape of tensor 't' is [2, 2, 3]
  /// rank(t) ==> 3
  /// ```
  ///
  /// **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
  /// of a tensor is the number of indices required to uniquely select each element
  /// of the tensor. Rank is also known as "order", "degree", or "ndims."
  public static func rank<T: TensorFlowScalar>(
    _ input: Tensor<T>
  ) -> Tensor<Int32> {
    return Tensor<Int32>(Int32(input.shape.rank), on: input.device)
  }

  /// Computes rectified linear 6: `min(max(features, 0), 6)`.
  public static func relu6<T: TensorFlowNumeric>(
    features: Tensor<T>
  ) -> Tensor<T> {
    return _RawXLA.minimum(
      _RawXLA.maximum(features, Tensor<T>(0, deviceAndPrecisionLike: features)),
      Tensor<T>(6, deviceAndPrecisionLike: features))
  }

  /// Computes rectified linear 6 gradients for a Relu6 operation.
  ///
  /// - Parameters:
  ///     - gradients: The backpropagated gradients to the corresponding Relu6 operation.
  ///     - features: The features passed as input to the corresponding Relu6 operation, or
  ///         its output; using either one produces the same result.
  ///
  /// - Output backprops: The gradients:
  ///     `gradients * (features > 0) * (features < 6)`.
  public static func relu6Grad<T: TensorFlowNumeric>(
    gradients: Tensor<T>,
    features: Tensor<T>
  ) -> Tensor<T> {
    checkSameDevice(gradients, features)
    checkSamePrecision(gradients, features)
    return _RawXLA.select(
      condition: _RawXLA.logicalAnd(
        _RawXLA.greater(features, Tensor<T>(0, deviceAndPrecisionLike: features)),
        _RawXLA.less(features, Tensor<T>(6, deviceAndPrecisionLike: features))),
      t: gradients,
      e: zerosLike(features))
  }

  /// Computes rectified linear gradients for a Relu operation.
  ///
  /// - Parameters:
  ///     - gradients: The backpropagated gradients to the corresponding Relu operation.
  ///     - features: The features passed as input to the corresponding Relu operation, OR
  ///         the outputs of that operation (both work equivalently).
  ///
  /// - Output backprops: `gradients * (features > 0)`.
  public static func reluGrad<T: TensorFlowNumeric>(
    gradients: Tensor<T>,
    features: Tensor<T>
  ) -> Tensor<T> {
    return threshold(features, output: gradients, threshold: 0, value: 0)
  }

  public static func replicaId(_ device: Device) -> Tensor<Int32> {
    return Tensor(_xla: XLATensor.replica_id(device))
  }

  /// Reshapes a tensor.
  ///
  /// Given `tensor`, this operation returns a tensor that has the same values
  /// as `tensor` with shape `shape`.
  ///
  /// If one component of `shape` is the special value -1, the size of that dimension
  /// is computed so that the total size remains constant.  In particular, a `shape`
  /// of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
  ///
  /// If `shape` is 1-D or higher, then the operation returns a tensor with shape
  /// `shape` filled with the values of `tensor`. In this case, the number of elements
  /// implied by `shape` must be the same as the number of elements in `tensor`.
  ///
  /// For example:
  ///
  /// ```
  /// # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
  /// # tensor 't' has shape [9]
  /// reshape(t, [3, 3]) ==> [[1, 2, 3],
  ///                         [4, 5, 6],
  ///                         [7, 8, 9]]
  ///
  /// # tensor 't' is [[[1, 1], [2, 2]],
  /// #                [[3, 3], [4, 4]]]
  /// # tensor 't' has shape [2, 2, 2]
  /// reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
  ///                         [3, 3, 4, 4]]
  ///
  /// # tensor 't' is [[[1, 1, 1],
  /// #                 [2, 2, 2]],
  /// #                [[3, 3, 3],
  /// #                 [4, 4, 4]],
  /// #                [[5, 5, 5],
  /// #                 [6, 6, 6]]]
  /// # tensor 't' has shape [3, 2, 3]
  /// # pass '[-1]' to flatten 't'
  /// reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
  ///
  /// # -1 can also be used to infer the shape
  ///
  /// # -1 is inferred to be 9:
  /// reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
  ///                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  /// # -1 is inferred to be 2:
  /// reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
  ///                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  /// # -1 is inferred to be 3:
  /// reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
  ///                               [2, 2, 2],
  ///                               [3, 3, 3]],
  ///                              [[4, 4, 4],
  ///                               [5, 5, 5],
  ///                               [6, 6, 6]]]
  ///
  /// # tensor 't' is [7]
  /// # shape `[]` reshapes to a scalar
  /// reshape(t, []) ==> 7
  /// ```
  ///
  /// - Parameter shape: Defines the shape of the output tensor.
  public static func reshape<
    T: TensorFlowScalar
  >(
    _ tensor: Tensor<T>,
    shape: [Int64]
  ) -> Tensor<T> {
    var dims = shape
    let dimsShape = dims.reduce(1, *)
    if let dynamicDim = dims.firstIndex(of: -1) {
      if dimsShape < 0 {
        dims[dynamicDim] = Int64(tensor.shape.contiguousSize) / -dimsShape
      }
    }
    return resize_value(tensor, dims: dims)
  }
  public static func reshape<
    T: TensorFlowScalar,
    Tshape: TensorFlowIndex
  >(
    _ tensor: Tensor<T>,
    shape: Tensor<Tshape>
  ) -> Tensor<T> {
    return reshape(tensor, shape: shape.scalars.map(Int64.init))
  }

  /// Reverses specific dimensions of a tensor.
  ///
  /// NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
  /// `tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.
  ///
  /// Given a `tensor`, and a `int32` tensor `axis` representing the set of
  /// dimensions of `tensor` to reverse. This operation reverses each dimension
  /// `i` for which there exists `j` s.t. `axis[j] == i`.
  ///
  /// `tensor` can have up to 8 dimensions. The number of dimensions specified
  /// in `axis` may be 0 or more entries. If an index is specified more than
  /// once, a InvalidArgument error is raised.
  ///
  /// For example:
  ///
  /// ```
  /// # tensor 't' is [[[[ 0,  1,  2,  3],
  /// #                  [ 4,  5,  6,  7],
  /// #                  [ 8,  9, 10, 11]],
  /// #                 [[12, 13, 14, 15],
  /// #                  [16, 17, 18, 19],
  /// #                  [20, 21, 22, 23]]]]
  /// # tensor 't' shape is [1, 2, 3, 4]
  ///
  /// # 'dims' is [3] or 'dims' is [-1]
  /// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
  ///                         [ 7,  6,  5,  4],
  ///                         [ 11, 10, 9, 8]],
  ///                        [[15, 14, 13, 12],
  ///                         [19, 18, 17, 16],
  ///                         [23, 22, 21, 20]]]]
  ///
  /// # 'dims' is '[1]' (or 'dims' is '[-3]')
  /// reverse(t, dims) ==> [[[[12, 13, 14, 15],
  ///                         [16, 17, 18, 19],
  ///                         [20, 21, 22, 23]
  ///                        [[ 0,  1,  2,  3],
  ///                         [ 4,  5,  6,  7],
  ///                         [ 8,  9, 10, 11]]]]
  ///
  /// # 'dims' is '[2]' (or 'dims' is '[-2]')
  /// reverse(t, dims) ==> [[[[8, 9, 10, 11],
  ///                         [4, 5, 6, 7],
  ///                         [0, 1, 2, 3]]
  ///                        [[20, 21, 22, 23],
  ///                         [16, 17, 18, 19],
  ///                         [12, 13, 14, 15]]]]
  /// ```
  ///
  /// - Parameters:
  ///     - tensor: Up to 8-D.
  ///     - axis: 1-D. The indices of the dimensions to reverse. Must be in the range
  ///         `[-rank(tensor), rank(tensor))`.
  ///
  /// - Output output: The same shape as `tensor`.
  @inlinable @inline(__always)
  public static func reverseV2<
    Tidx: TensorFlowIndex,
    T: TensorFlowScalar
  >(
    _ tensor: Tensor<T>,
    axis: Tensor<Tidx>
  ) -> Tensor<T> {
    fatalError("Implement reverseV2")
  }

  /// Computes the gradient for the rsqrt of `x` wrt its input.
  ///
  /// Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
  /// is the corresponding input gradient.
  public static func rsqrtGrad<T: FloatingPoint & TensorFlowScalar>(
    _ y: Tensor<T>,
    dy: Tensor<T>
  ) -> Tensor<T> {
    checkSameDevice(y, dy)
    checkSamePrecision(y, dy)
    return _RawXLA.mul(
      _RawXLA.mul(_RawXLA.mul(y, y), y), _RawXLA.div(dy, Tensor<T>(-2, deviceAndPrecisionLike: y)))
  }

  /// Selects elements from `x` or `y`, depending on `condition`.
  ///
  /// The `x`, and `y` tensors must all have the same shape, and the
  /// output will also have that shape.
  ///
  /// The `condition` tensor must be a scalar if `x` and `y` are scalars.
  /// If `x` and `y` are vectors or higher rank, then `condition` must be either a
  /// scalar, a vector with size matching the first dimension of `x`, or must have
  /// the same shape as `x`.
  ///
  /// The `condition` tensor acts as a mask that chooses, based on the value at each
  /// element, whether the corresponding element / row in the output should be
  /// taken from `x` (if true) or `y` (if false).
  ///
  /// If `condition` is a vector and `x` and `y` are higher rank matrices, then
  /// it chooses which row (outer dimension) to copy from `x` and `y`.
  /// If `condition` has the same shape as `x` and `y`, then it chooses which
  /// element to copy from `x` and `y`.
  ///
  /// For example:
  ///
  /// ```python
  /// # 'condition' tensor is [[True,  False]
  /// #                        [False, True]]
  /// # 't' is [[1, 2],
  /// #         [3, 4]]
  /// # 'e' is [[5, 6],
  /// #         [7, 8]]
  /// select(condition, t, e)  # => [[1, 6], [7, 4]]
  ///
  ///
  /// # 'condition' tensor is [True, False]
  /// # 't' is [[1, 2],
  /// #         [3, 4]]
  /// # 'e' is [[5, 6],
  /// #         [7, 8]]
  /// select(condition, t, e) ==> [[1, 2],
  ///                              [7, 8]]
  ///
  /// ```
  ///
  /// - Parameters:
  ///     - t: = A `Tensor` which may have the same shape as `condition`.
  ///         If `condition` is rank 1, `x` may have higher rank,
  ///         but its first dimension must match the size of `condition`.
  ///     - e: = A `Tensor` with the same type and shape as `x`.
  ///
  /// - Output output: = A `Tensor` with the same type and shape as `x` and `y`.
  public static func select<T: TensorFlowScalar>(
    condition: Tensor<Bool>,
    t: Tensor<T>,
    e: Tensor<T>
  ) -> Tensor<T> {
    var dims = condition.shape.dimensions.map(Int64.init)
    let tdims = t.shape.dimensions.map { Int64($0) }
    while dims.count < t.rank { dims.append(1) }
    let broadcastedCondition = broadcastTo(resize_value(condition, dims: dims), dims: tdims)
    return where_(condition: broadcastedCondition, input: t, other: e)
  }

  private static let seluGamma = 1.0507009873554804934193349852946
  private static let seluAlphaGamma = 1.7580993408473768599402175208123

  /// Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`
  ///
  /// if < 0, `scale * features` otherwise.
  ///
  /// To be used together with
  /// `initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN')`.
  /// For correct dropout, use `tf.contrib.nn.alpha_dropout`.
  ///
  /// See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
  public static func selu<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>
  ) -> Tensor<T> {
    return _RawXLA.select(
      condition: _RawXLA.greater(features, _RawXLA.zerosLike(features)),
      t: _RawXLA.mul(fullLike(seluGamma, features), features),
      e: _RawXLA.mul(fullLike(seluAlphaGamma, features), _RawXLA.expm1(features)))
  }

  /// Computes gradients for the scaled exponential linear (Selu) operation.
  ///
  /// - Parameters:
  ///     - gradients: The backpropagated gradients to the corresponding Selu operation.
  ///     - outputs: The outputs of the corresponding Selu operation.
  ///
  /// - Output backprops: The gradients: `gradients * (outputs + scale * alpha)`
  ///     if outputs < 0, `scale * gradients` otherwise.
  public static func seluGrad<T: FloatingPoint & TensorFlowScalar>(
    gradients: Tensor<T>,
    outputs: Tensor<T>
  ) -> Tensor<T> {
    checkSameDevice(gradients, outputs)
    checkSamePrecision(gradients, outputs)
    return _RawXLA.select(
      condition: _RawXLA.greater(outputs, _RawXLA.zerosLike(outputs)),
      t: _RawXLA.mul(fullLike(seluGamma, outputs), gradients),
      e: _RawXLA.mul(gradients, _RawXLA.addV2(outputs, fullLike(seluAlphaGamma, outputs))))
  }

  /// Returns the shape of a tensor.
  ///
  /// This operation returns a 1-D integer tensor representing the shape of `input`.
  ///
  /// For example:
  ///
  /// ```
  /// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  /// shape(t) ==> [2, 2, 3]
  /// ```
  public static func shape<
    T: TensorFlowScalar,
    OutType: TensorFlowIndex
  >(
    _ input: Tensor<T>
  ) -> Tensor<OutType> {
    let shape = input.xlaTensor.shape
    return Tensor(shape.map { OutType($0) }, on: input.device)
  }

  /// Computes the gradient of the sigmoid of `x` wrt its input.
  ///
  /// Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
  /// `dy` is the corresponding input gradient.
  public static func sigmoidGrad<T: FloatingPoint & TensorFlowScalar>(
    _ y: Tensor<T>,
    dy: Tensor<T>
  ) -> Tensor<T> {
    checkSameDevice(y, dy)
    checkSamePrecision(y, dy)
    return _RawXLA.mul(_RawXLA.mul(dy, y), _RawXLA.sub(_RawXLA.onesLike(y), y))
  }

  /// Returns the size of a tensor.
  ///
  /// This operation returns an integer representing the number of elements in
  /// `input`.
  ///
  /// For example:
  ///
  /// ```
  /// # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
  /// size(t) ==> 12
  /// ```
  public static func size<
    T: TensorFlowScalar,
    OutType: TensorFlowIndex
  >(
    _ input: Tensor<T>
  ) -> Tensor<OutType> {
    return Tensor<OutType>(OutType(input.shape.contiguousSize), on: input.device)
  }

  /// Return a slice from 'input'.
  ///
  /// The output tensor is a tensor with dimensions described by 'size'
  /// whose values are extracted from 'input' starting at the offsets in
  /// 'begin'.
  ///
  /// *Requirements*:
  ///   0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
  ///
  /// - Parameters:
  ///     - begin: begin[i] specifies the offset into the 'i'th dimension of
  ///         'input' to slice from.
  ///     - size: size[i] specifies the number of elements of the 'i'th dimension
  ///         of 'input' to slice. If size[i] is -1, all remaining elements in dimension
  ///         i are included in the slice (i.e. this is equivalent to setting
  ///         size[i] = input.dim_size(i) - begin[i]).
  public static func slice<
    T: TensorFlowScalar,
    Index: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    begin: Tensor<Index>,
    size: Tensor<Index>
  ) -> Tensor<T> {
    let inputRank = input.rank
    precondition(begin.shape[0] == inputRank && size.shape[0] == inputRank)
    return slice(input, begin: begin.scalars.map { Int($0) }, size: size.scalars.map { Int($0) })
  }

  public static func slice<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    begin: [Int],
    size: [Int]
  ) -> Tensor<T> {
    let inputRank = input.rank
    precondition(begin.count == inputRank && size.count == inputRank)
    var output = input
    for (axis, (begin, size)) in zip(begin, size).enumerated() {
      let dimensionSize = input.shape.dimensions[axis]
      if size != -1 {
        if size < 0 || size > dimensionSize {
          fatalError("Expected size[\(axis)] in [0, \(dimensionSize)], but got \(size)")
        }
        output = slice(
          output, dim: Int64(axis), start: Int64(begin), end: Int64(begin + size), stride: 1)
      } else {
        output = slice(
          output, dim: Int64(axis), start: Int64(begin), end: Int64(dimensionSize), stride: 1)
      }
    }
    return output
  }

  /// Computes softmax activations.
  ///
  /// For each batch `i` and class `j` we have
  ///
  ///     $$softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))$$
  ///
  /// - Parameter logits: 2-D with shape `[batch_size, num_classes]`.
  ///
  /// - Output softmax: Same shape as `logits`.
  public static func softmax<T: FloatingPoint & TensorFlowScalar>(
    logits: Tensor<T>
  ) -> Tensor<T> {
    return softmax(logits, dim: -1)
  }

  /// Computes softmax cross entropy cost and gradients to backpropagate.
  ///
  /// Inputs are the logits, not probabilities.
  ///
  /// - Parameters:
  ///     - features: batch_size x num_classes matrix
  ///     - labels: batch_size x num_classes matrix
  ///         The caller must ensure that each batch of labels represents a valid
  ///         probability distribution.
  ///
  /// - Outputs:
  ///     - loss: Per example loss (batch_size vector).
  ///     - backprop: backpropagated gradients (batch_size x num_classes matrix).
  public static func softmaxCrossEntropyWithLogits<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>,
    labels: Tensor<T>
  ) -> (loss: Tensor<T>, backprop: Tensor<T>) {
    checkSameDevice(features, labels)
    checkSamePrecision(features, labels)
    let logits = features
    let logits_max = max(logits, dim: 1, keepDim: true)
    let shifted_logits = _RawXLA.sub(logits, logits_max)
    let exp = _RawXLA.exp(shifted_logits)
    let sum_exp = _RawXLA.sum(exp, reductionIndices: Tensor<Int64>([1]), keepDims: true)
    let log_sum_exp = _RawXLA.log(sum_exp)
    return (
      loss: _RawXLA.sum(
        _RawXLA.mul(-labels, shifted_logits - log_sum_exp),
        reductionIndices: Tensor<Int64>([1])),
      backprop: _RawXLA.sub(_RawXLA.div(exp, sum_exp), labels)
    )
  }

  /// Computes softplus: `log(exp(features) + 1)`.
  public static func softplus<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>
  ) -> Tensor<T> {
    _RawXLA.addV2(
      _RawXLA.maximum(features, _RawXLA.zerosLike(features)),
      _RawXLA.log1p(_RawXLA.exp(-_RawXLA.abs(features))))
  }

  /// Computes softplus gradients for a softplus operation.
  ///
  /// - Parameters:
  ///     - gradients: The backpropagated gradients to the corresponding softplus operation.
  ///     - features: The features passed as input to the corresponding softplus operation.
  ///
  /// - Output backprops: The gradients: `gradients / (1 + exp(-features))`.
  public static func softplusGrad<T: FloatingPoint & TensorFlowScalar>(
    gradients: Tensor<T>,
    features: Tensor<T>
  ) -> Tensor<T> {
    checkSameDevice(gradients, features)
    checkSamePrecision(gradients, features)
    let half = fullLike(0.5, features)
    return _RawXLA.mul(
      gradients, _RawXLA.addV2(half, _RawXLA.mul(half, _RawXLA.tanh(_RawXLA.mul(half, features)))))
  }

  /// Computes softsign: `features / (abs(features) + 1)`.
  public static func softsign<T: FloatingPoint & TensorFlowScalar>(
    features: Tensor<T>
  ) -> Tensor<T> {
    _RawXLA.div(features, (_RawXLA.addV2(_RawXLA.abs(features), _RawXLA.onesLike(features))))
  }

  /// Computes softsign gradients for a softsign operation.
  ///
  /// - Parameters:
  ///     - gradients: The backpropagated gradients to the corresponding softsign operation.
  ///     - features: The features passed as input to the corresponding softsign operation.
  ///
  /// - Output backprops: The gradients: `gradients / (1 + abs(features)) ** 2`.
  public static func softsignGrad<T: FloatingPoint & TensorFlowScalar>(
    gradients: Tensor<T>,
    features: Tensor<T>
  ) -> Tensor<T> {
    checkSameDevice(gradients, features)
    checkSamePrecision(gradients, features)
    return _RawXLA.div(
      gradients, _RawXLA.square(_RawXLA.addV2(_RawXLA.abs(features), _RawXLA.onesLike(features))))
  }

  /// Computes softmax cross entropy cost and gradients to backpropagate.
  ///
  /// Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
  /// a matrix of label probabilities, but rather a single label per row
  /// of features.  This label is considered to have probability 1.0 for the
  /// given row.
  ///
  /// Inputs are the logits, not probabilities.
  ///
  /// - Parameters:
  ///     - features: batch_size x num_classes matrix
  ///     - labels: batch_size vector with values in [0, num_classes).
  ///         This is the label for the given minibatch entry.
  ///
  /// - Outputs:
  ///     - loss: Per example loss (batch_size vector).
  ///     - backprop: backpropagated gradients (batch_size x num_classes matrix).
  public static func sparseSoftmaxCrossEntropyWithLogits<
    T: FloatingPoint & TensorFlowScalar,
    Tlabels: TensorFlowIndex
  >(
    features: Tensor<T>,
    labels: Tensor<Tlabels>
  ) -> (loss: Tensor<T>, backprop: Tensor<T>) {
    checkSameDevice(features.device, labels.device)
    let device = labels.device
    let tmp_output = _RawXLA.logSoftmax(logits: features)
    var one_hot = _RawXLA.neg(
      _RawXLA.oneHot(
        indices: labels, depth: Int64(features.shape[1]),
        onValue: Tensor<T>(1, on: device),
        offValue: Tensor<T>(0, on: device)))
    if features.isReducedPrecision {
      one_hot = one_hot.toReducedPrecision
    }
    let loss = _RawXLA.sum(
      _RawXLA.mul(one_hot, tmp_output),
      reductionIndices: [Int64(1)], keepDims: false)
    // one_hot stands in for the backprop gradient as it has the right shape
    // and will multiplied on the backwards pass.
    let backprop = logSoftmaxBackward(gradOutput: one_hot, output: tmp_output, dim: -1)
    return (loss: loss, backprop: backprop)
  }

  /// Splits a tensor into `num_split` tensors along one dimension.
  ///
  /// - Parameters:
  ///     - split_dim: 0-D.  The dimension along which to split.  Must be in the range
  ///         `[-rank(value), rank(value))`.
  ///     - value: The tensor to split.
  ///
  /// - Attr num_split: The number of ways to split.  Must evenly divide
  ///     `value.shape[split_dim]`.
  ///
  /// - Output output: They are identically shaped tensors, whose shape matches that of `value`
  ///     except along `axis`, where their sizes are
  ///     `values.shape[split_dim] / num_split`.
  public static func split<T: TensorFlowScalar>(
    splitDim: Tensor<Int32>,
    value: Tensor<T>,
    numSplit: Int64
  ) -> [Tensor<T>] {
    split(splitDim: Int(splitDim.scalarized()), value: value, numSplit: numSplit)
  }

  public static func split<T: TensorFlowScalar>(
    splitDim: Int,
    value: Tensor<T>,
    numSplit: Int64
  ) -> [Tensor<T>] {
    let canonicalSplitDim = canonicalDims([Int64(splitDim)], Int64(value.rank)).first!
    let splitDimSize = value.shape.dimensions[Int(canonicalSplitDim)]
    if Int64(splitDimSize) % numSplit != 0 {
      fatalError(
        "Number of ways to split should evenly divide the split dimension, but got splitDim "
          + "\(splitDim) (size = \(splitDimSize)) and numSplit \(numSplit)")
    }
    let chunkSize = Int(Int64(splitDimSize) / numSplit)
    let sizeSplits = Array(repeating: chunkSize, count: Int(numSplit))
    return splitV(value: value, sizeSplits: sizeSplits, splitDim: splitDim)
  }

  /// Splits a tensor into `num_split` tensors along one dimension.
  ///
  /// - Parameters:
  ///     - value: The tensor to split.
  ///     - size_splits: list containing the sizes of each output tensor along the split
  ///         dimension. Must sum to the dimension of value along split_dim.
  ///         Can contain one -1 indicating that dimension is to be inferred.
  ///     - split_dim: 0-D.  The dimension along which to split.  Must be in the range
  ///         `[-rank(value), rank(value))`.
  ///
  /// - Output output: Tensors whose shape matches that of `value`
  ///     except along `axis`, where their sizes are
  ///     `size_splits[i]`.
  public static func splitV<
    T: TensorFlowScalar,
    Tlen: TensorFlowIndex
  >(
    value: Tensor<T>,
    sizeSplits: Tensor<Tlen>,
    splitDim: Tensor<Int32>,
    numSplit: Int64
  ) -> [Tensor<T>] {
    guard sizeSplits.rank == 1, sizeSplits.shape.contiguousSize == numSplit else {
      fatalError(
        "shape of tensor describing the output must have dimension 1 and the same number of "
          + "elements as the output. Got \(sizeSplits.rank)-D and "
          + "\(sizeSplits.shape.contiguousSize) elements"
      )
    }
    return splitV(
      value: value, sizeSplits: sizeSplits.scalars.map { Int($0) },
      splitDim: Int(splitDim.scalarized()))
  }

  public static func splitV<
    T: TensorFlowScalar
  >(
    value: Tensor<T>,
    sizeSplits: [Int],
    splitDim: Int
  ) -> [Tensor<T>] {
    let inferredIndices = sizeSplits.indices.filter { sizeSplits[$0] == -1 }
    guard inferredIndices.count <= 1 else {
      fatalError(
        "Only one dimensions can have a value of -1. Second one found at dimension "
          + String(inferredIndices[1])
      )
    }
    let totalSplitSize = sizeSplits.filter { $0 != -1 }.reduce(0, +)
    let canonicalSplitDim = canonicalDims([Int64(splitDim)], Int64(value.rank))
      .first!
    let splitDimSize = value.shape.dimensions[Int(canonicalSplitDim)]
    guard
      (inferredIndices.count == 0 && totalSplitSize == splitDimSize)
        || (inferredIndices.count == 1 && totalSplitSize <= splitDimSize)
    else {
      fatalError(
        "Determined shape must either match input shape along split_dim exactly if fully "
          + "specified, or be less than the size of the input along split_dim if not fully "
          + "specified. Got: \(totalSplitSize)"
      )
    }
    var completeSizeSplits = sizeSplits.map { Int64($0) }
    if inferredIndices.count == 1 {
      completeSizeSplits[inferredIndices.first!] = Int64(splitDimSize) - Int64(totalSplitSize)
    }
    var offset: Int64 = 0
    let dim = Int64(splitDim)
    return completeSizeSplits.map { (size: Int64) -> Tensor<T> in
      let nextOffset = offset + size
      let result = slice(value, dim: dim, start: offset, end: nextOffset, stride: 1)
      offset = nextOffset
      return result
    }
  }

  /// Computes square of x element-wise.
  ///
  /// I.e., \\(y = x * x = x^2\\).
  public static func square<T: TensorFlowNumeric>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    _RawXLA.mul(x, x)
  }

  /// Returns (x - y)(x - y) element-wise.
  ///
  /// *NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
  /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  public static func squaredDifference<T: TensorFlowNumeric>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    checkSameDevice(x, y)
    checkSamePrecision(x, y)
    return _RawXLA.square(_RawXLA.sub(x, y))
  }

  /// Removes dimensions of size 1 from the shape of a tensor.
  ///
  /// Given a tensor `input`, this operation returns a tensor of the same type with
  /// all dimensions of size 1 removed. If you don't want to remove all size 1
  /// dimensions, you can remove specific size 1 dimensions by specifying
  /// `axis`.
  ///
  /// For example:
  ///
  /// ```
  /// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  /// shape(squeeze(t)) ==> [2, 3]
  /// ```
  ///
  /// Or, to remove specific size 1 dimensions:
  ///
  /// ```
  /// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  /// shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
  /// ```
  ///
  /// - Parameter input: The `input` to squeeze.
  ///
  /// - Attr squeeze_dims: If specified, only squeezes the dimensions listed. The dimension
  ///     index starts at 0. It is an error to squeeze a dimension that is not 1. Must
  ///     be in the range `[-rank(input), rank(input))`.
  ///
  /// - Output output: Contains the same data as `input`, but has one or more dimensions of
  ///     size 1 removed.
  public static func squeeze<T: TensorFlowScalar>(
    _ input: Tensor<T>,
    squeezeDims: [Int32]
  ) -> Tensor<T> {
    var output = input
    let shape = input.shape
    var dims = canonicalDims(squeezeDims.map { Int64($0) }, Int64(shape.rank))
    if dims.count == 0 {
      var total = 0
      for dim in 0..<shape.rank {
        if shape[dim] == 1 {
          output = squeeze(output, dim: Int64(dim - total))
          total += 1
        }
      }
      return output
    }
    // Go through dims in reverse order.
    dims.sort(by: >)
    for dim in dims {
      output = squeeze(output, dim: Int64(dim))
    }
    return output
  }

  /// Draws samples from a multinomial distribution.
  ///
  /// - Parameters:
  ///     - logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
  ///         represents the unnormalized log probabilities for all classes.
  ///     - num_samples: 0-D.  Number of independent samples to draw for each row slice.
  ///     - seed: 2 seeds (shape [2]).
  ///
  /// - Output output: 2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
  ///     contains the drawn class labels with range `[0, num_classes)`.
  @inlinable @inline(__always)
  public static func statelessMultinomial<
    T: TensorFlowNumeric,
    Tseed: TensorFlowIndex,
    OutputDtype: TensorFlowIndex
  >(
    logits: Tensor<T>,
    numSamples: Tensor<Int32>,
    seed: Tensor<Tseed>
  ) -> Tensor<OutputDtype> {
    fatalError("Implement statelessMultinomial")
  }

  /// Outputs deterministic pseudorandom values from a normal distribution.
  ///
  /// The generated values will have mean 0 and standard deviation 1.
  ///
  /// The outputs are a deterministic function of `shape` and `seed`.
  ///
  /// - Parameters:
  ///     - shape: The shape of the output tensor.
  ///     - seed: 2 seeds (shape [2]).
  ///
  /// - Attr dtype: The type of the output.
  ///
  /// - Output output: Random values with specified shape.
  public static func statelessRandomNormal<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
  >(
    shape: Tensor<T>,
    seed: Tensor<Tseed>,
    device: Device
  ) -> Tensor<Dtype> {
    tf_StatelessRandomNormal(
      shape.scalars.map { Int64($0) }, seed, dtype: Dtype.xlaTensorScalarType)
  }

  public static func statelessRandomNormal<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
  >(
    shape: Tensor<T>,
    seed: Tensor<Tseed>
  ) -> Tensor<Dtype> {
    let device = seed.device
    return statelessRandomNormal(shape: shape, seed: seed, device: device)
  }

  /// Outputs deterministic pseudorandom random values from a uniform distribution.
  ///
  /// The generated values follow a uniform distribution in the range `[0, 1)`. The
  /// lower bound 0 is included in the range, while the upper bound 1 is excluded.
  ///
  /// The outputs are a deterministic function of `shape` and `seed`.
  ///
  /// - Parameters:
  ///     - shape: The shape of the output tensor.
  ///     - seed: 2 seeds (shape [2]).
  ///
  /// - Attr dtype: The type of the output.
  ///
  /// - Output output: Random values with specified shape.
  public static func statelessRandomUniform<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
  >(
    shape: Tensor<T>,
    seed: Tensor<Tseed>,
    device: Device
  ) -> Tensor<Dtype> {
    tf_StatelessRandomUniform(
      shape.scalars.map { Int64($0) },
      seed, Tensor<Dtype>(0, on: device),
      Tensor<Dtype>(1, on: device))
  }

  public static func statelessRandomUniform<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
  >(
    shape: Tensor<T>,
    seed: Tensor<Tseed>
  ) -> Tensor<Dtype> {
    let device = seed.device
    return statelessRandomUniform(shape: shape, seed: seed, device: device)
  }

  /// Outputs deterministic pseudorandom random integers from a uniform distribution.
  ///
  /// The generated values follow a uniform distribution in the range `[minval, maxval)`.
  ///
  /// The outputs are a deterministic function of `shape`, `seed`, `minval`, and `maxval`.
  ///
  /// - Parameters:
  ///     - shape: The shape of the output tensor.
  ///     - seed: 2 seeds (shape [2]).
  ///     - minval: Minimum value (inclusive, scalar).
  ///     - maxval: Maximum value (exclusive, scalar).
  ///
  /// - Attr dtype: The type of the output.
  ///
  /// - Output output: Random values with specified shape.
  public static func statelessRandomUniformInt<
    Dtype: TensorFlowIndex,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
  >(
    shape: Tensor<T>,
    seed: Tensor<Tseed>,
    minval: Tensor<Dtype>,
    maxval: Tensor<Dtype>,
    device: Device
  ) -> Tensor<Dtype> {
    tf_StatelessRandomUniform(shape.scalars.map { Int64($0) }, seed, minval, maxval)
  }

  public static func statelessRandomUniformInt<
    Dtype: TensorFlowIndex,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
  >(
    shape: Tensor<T>,
    seed: Tensor<Tseed>,
    minval: Tensor<Dtype>,
    maxval: Tensor<Dtype>
  ) -> Tensor<Dtype> {
    let device = minval.device
    return statelessRandomUniformInt(
      shape: shape, seed: seed, minval: minval, maxval: maxval, device: device)
  }

  /// Outputs deterministic pseudorandom values from a truncated normal distribution.
  ///
  /// The generated values follow a normal distribution with mean 0 and standard
  /// deviation 1, except that values whose magnitude is more than 2 standard
  /// deviations from the mean are dropped and re-picked.
  ///
  /// The outputs are a deterministic function of `shape` and `seed`.
  ///
  /// - Parameters:
  ///     - shape: The shape of the output tensor.
  ///     - seed: 2 seeds (shape [2]).
  ///
  /// - Attr dtype: The type of the output.
  ///
  /// - Output output: Random values with specified shape.
  public static func statelessTruncatedNormal<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
  >(
    shape: Tensor<T>,
    seed: Tensor<Tseed>,
    device: Device
  ) -> Tensor<Dtype> {
    let minval = Tensor<Dtype>(Dtype.leastNormalMagnitude, on: device)
    let maxval = Tensor<Dtype>(1, on: device)
    let uniform = tf_StatelessRandomUniform(
      shape.scalars.map { Int64($0) },
      seed, minval, maxval)
    return truncatedNormal(uniform)
  }

  public static func statelessTruncatedNormal<
    Dtype: FloatingPoint & TensorFlowScalar,
    T: TensorFlowIndex,
    Tseed: TensorFlowIndex
  >(
    shape: Tensor<T>,
    seed: Tensor<Tseed>
  ) -> Tensor<Dtype> {
    let device = seed.device
    return statelessTruncatedNormal(shape: shape, seed: seed, device: device)
  }

  /// Return a strided slice from `input`.
  ///
  /// Note, most python users will want to use the Python `Tensor.__getitem__`
  /// or `Variable.__getitem__` rather than this op directly.
  ///
  /// The goal of this op is to produce a new tensor with a subset of
  /// the elements from the `n` dimensional `input` tensor. The subset is chosen using
  /// a sequence of `m` sparse range specifications encoded into the arguments
  /// of this function. Note, in some cases
  /// `m` could be equal to `n`, but this need not be the case. Each
  /// range specification entry can be one of the following:
  ///
  /// - An ellipsis (...). Ellipses are used to imply zero or more
  ///   dimensions of full-dimension selection and are produced using
  ///   `ellipsis_mask`. For example, `foo[...]` is the identity slice.
  ///
  /// - A new axis. This is used to insert a new shape=1 dimension and is
  ///   produced using `new_axis_mask`. For example, `foo[:, ...]` where
  ///   `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.
  ///
  ///
  /// - A range `begin:end:stride`. This is used to specify how much to choose from
  ///   a given dimension. `stride` can be any integer but 0.  `begin` is an integer
  ///   which represents the index of the first value to select while `end` represents
  ///   the index of the last value to select. The number of values selected in each
  ///   dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
  ///   `begin` and `end` can be negative where `-1` is the last element, `-2` is
  ///   the second to last. `begin_mask` controls whether to replace the explicitly
  ///   given `begin` with an implicit effective value of `0` if `stride > 0` and
  ///   `-1` if `stride < 0`. `end_mask` is analogous but produces the number
  ///   required to create the largest open interval. For example, given a shape
  ///   `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
  ///   not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
  ///   and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
  ///   first dimension of a tensor while dropping the last two (in the original
  ///   order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.
  ///
  /// - A single index. This is used to keep only elements that have a given
  ///   index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
  ///   shape `(6,)` tensor. This is encoded in `begin` and `end` and
  ///   `shrink_axis_mask`.
  ///
  /// Each conceptual range specification is encoded in the op's argument. This
  /// encoding is best understand by considering a non-trivial example. In
  /// particular,
  /// `foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as
  ///
  /// ```
  /// begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
  /// end = [2, 4, x, x, -3, x]
  /// strides = [1, 1, x, x, -1, 1]
  /// begin_mask = 1<<4 | 1 << 5 = 48
  /// end_mask = 1<<5 = 32
  /// ellipsis_mask = 1<<3 = 8
  /// new_axis_mask = 1<<2 4
  /// shrink_axis_mask = 1<<0
  /// ```
  ///
  /// In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
  /// the slice becomes (2, 1, 5, 5, 2, 5).
  /// Let us walk step by step through each argument specification.
  ///
  /// 1.  The first argument in the example slice is turned into `begin = 1` and
  /// `end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
  /// also set the appropriate bit in `shrink_axis_mask`.
  ///
  /// 2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
  /// zero bits contributed.
  ///
  /// 3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
  /// dimension in the final shape. Dummy values are contributed to begin,
  /// end and stride, while the new_axis_mask bit is set.
  ///
  /// 4. `...` grab the full ranges from as many dimensions as needed to
  /// fully specify a slice for every dimension of the input shape.
  ///
  /// 5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
  /// with a dimension that has shape `s` is converted to a positive index
  /// `s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
  /// is done internally so begin, end and strides receive x, -3, and -1.
  /// The appropriate begin_mask bit is set to indicate the start range is the
  /// full range (ignoring the x).
  ///
  /// 6. `:` indicates that the entire contents of the corresponding dimension
  /// is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
  /// receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
  /// `end_mask` are also set.
  ///
  /// *Requirements*:
  ///   `0 != strides[i] for i in [0, m)`
  ///   `ellipsis_mask must be a power of two (only one ellipsis)`
  ///
  /// - Parameters:
  ///     - begin: `begin[k]` specifies the offset into the `k`th range specification.
  ///         The exact dimension this corresponds to will be determined by context.
  ///         Out-of-bounds values will be silently clamped. If the `k`th bit of
  ///         `begin_mask` then `begin[k]` is ignored and the full range of the
  ///         appropriate dimension is used instead. Negative values causes indexing
  ///         to start from the highest element e.g. If `foo==[1,2,3]` then `foo[-1]==3`.
  ///     - end: `end[i]` is like `begin` with the exception that `end_mask` is
  ///         used to determine full ranges.
  ///     - strides: `strides[i]` specifies the increment in the `i`th specification
  ///         after extracting a given element. Negative indices will reverse
  ///         the original order. Out or range values are
  ///         clamped to `[0,dim[i]) if slice[i]>0` or `[-1,dim[i]-1] if slice[i] < 0`
  ///
  /// - Attrs:
  ///     - begin_mask: a bitmask where a bit i being 1 means to ignore the begin
  ///         value and instead use the largest interval possible. At runtime
  ///         begin[i] will be replaced with `[0, n-1)` if `stride[i] > 0` or
  ///         `[-1, n-1]` if `stride[i] < 0`
  ///     - end_mask: analogous to `begin_mask`
  ///     - ellipsis_mask: a bitmask where bit `i` being 1 means the `i`th
  ///         position is actually an ellipsis. One bit at most can be 1.
  ///         If `ellipsis_mask == 0`, then an implicit ellipsis mask of `1 << (m+1)`
  ///         is provided. This means that `foo[3:5] == foo[3:5, ...]`. An ellipsis
  ///         implicitly creates as many range specifications as necessary to fully
  ///         specify the sliced range for every dimension. For example for a 4-dimensional
  ///         tensor `foo` the slice `foo[2, ..., 5:8]` implies `foo[2, :, :, 5:8]`.
  ///     - new_axis_mask: a bitmask where bit `i` being 1 means the `i`th
  ///         specification creates a new shape 1 dimension. For example
  ///         `foo[:4, tf.newaxis, :2]` would produce a shape `(4, 1, 2)` tensor.
  ///     - shrink_axis_mask: a bitmask where bit `i` implies that the `i`th
  ///         specification should shrink the dimensionality. begin and end
  ///         must imply a slice of size 1 in the dimension. For example in
  ///         python one might do `foo[:, 3, :]` which would result in
  ///         `shrink_axis_mask` being 2.
  public static func stridedSlice<
    T: TensorFlowScalar,
    Index: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    begin: Tensor<Index>,
    end: Tensor<Index>,
    strides: Tensor<Index>,
    beginMask: Int64 = 0,
    endMask: Int64 = 0,
    ellipsisMask: Int64 = 0,
    newAxisMask: Int64 = 0,
    shrinkAxisMask: Int64 = 0
  ) -> Tensor<T> {
    let boundsAndStrides = XLATensor.computeIndexingBoundsAndStrides(
      inputSizes: input.shape.dimensions.map { Int64($0) }, begin: begin.scalars.map { Int64($0) },
      end: end.scalars.map { Int64($0) },
      strides: strides.scalars.map { Int64($0) }, beginMask: Int32(beginMask),
      endMask: Int32(endMask), ellipsisMask: Int32(ellipsisMask), newAxisMask: Int32(newAxisMask),
      shrinkAxisMask: Int32(shrinkAxisMask))
    var dimensionsToReverse: [Int64] = []
    var sliceBegin: [Int64] = []
    var sliceEnd: [Int64] = []
    var sliceStrides: [Int64] = []
    assert(
      boundsAndStrides.begin.count == boundsAndStrides.end.count
        && boundsAndStrides.begin.count == boundsAndStrides.strides.count)
    for i in 0..<boundsAndStrides.begin.count {
      if boundsAndStrides.strides[i] > 0 {
        sliceBegin.append(boundsAndStrides.begin[i])
        sliceEnd.append(Swift.max(boundsAndStrides.end[i], boundsAndStrides.begin[i]))
        sliceStrides.append(boundsAndStrides.strides[i])
      } else {
        // Negative stride: swap begin and end, add 1 because the interval is semi-open, and mark
        // the dimension to be reversed.
        sliceBegin.append(Int64(input.shape.dimensions[i]) - boundsAndStrides.begin[i] - 1)
        sliceEnd.append(
          Swift.max(
            Int64(input.shape.dimensions[i]) - boundsAndStrides.end[i] - 1,
            Int64(input.shape.dimensions[i]) - boundsAndStrides.begin[i] - 1))
        sliceStrides.append(-boundsAndStrides.strides[i])
        dimensionsToReverse.append(Int64(i))
      }
    }
    var result = input
    if !dimensionsToReverse.isEmpty {
      result = flip(result, dims: dimensionsToReverse)
    }
    result = xlaSlice(
      result, start_indices: sliceBegin, limit_indices: sliceEnd, strides: sliceStrides)
    return resize_value(result, dims: boundsAndStrides.finalSizes)
  }

  /// Returns the gradient of `StridedSlice`.
  ///
  /// Since `StridedSlice` cuts out pieces of its `input` which is size
  /// `shape`, its gradient will have the same shape (which is passed here
  /// as `shape`). The gradient will be zero in any element that the slice
  /// does not select.
  ///
  /// Arguments are the same as StridedSliceGrad with the exception that
  /// `dy` is the input gradient to be propagated and `shape` is the
  /// shape of `StridedSlice`'s `input`.
  public static func stridedSliceGrad<
    T: TensorFlowScalar,
    Index: TensorFlowIndex
  >(
    shape: Tensor<Index>,
    begin: Tensor<Index>,
    end: Tensor<Index>,
    strides: Tensor<Index>,
    dy: Tensor<T>,
    beginMask: Int64 = 0,
    endMask: Int64 = 0,
    ellipsisMask: Int64 = 0,
    newAxisMask: Int64 = 0,
    shrinkAxisMask: Int64 = 0
  ) -> Tensor<T> {
    let boundsAndStrides = XLATensor.computeIndexingBoundsAndStrides(
      inputSizes: shape.scalars.map { Int64($0) }, begin: begin.scalars.map { Int64($0) },
      end: end.scalars.map { Int64($0) },
      strides: strides.scalars.map { Int64($0) }, beginMask: Int32(beginMask),
      endMask: Int32(endMask), ellipsisMask: Int32(ellipsisMask), newAxisMask: Int32(newAxisMask),
      shrinkAxisMask: Int32(shrinkAxisMask))
    // Check to make sure dy is consistent with the original slice
    if dy.shape.dimensions != boundsAndStrides.finalSizes.map({ Int($0) }) {
      fatalError("Shape of dy was \(dy.shape.dimensions), expected \(boundsAndStrides.finalSizes)")
    }
    if shape.scalars.count != boundsAndStrides.processingSizes.count {
      fatalError("Input shape and processing shape must have same rank")
    }
    // Undo any new/shrink axes.
    var grad = resize_value(dy, dims: boundsAndStrides.processingSizes)
    // Pad the input gradients.
    var dimensionsToReverse: [Int64] = []
    var paddingConfig: [PaddingConfigDimension] = []
    assert(
      boundsAndStrides.begin.count == boundsAndStrides.end.count
        && boundsAndStrides.begin.count == boundsAndStrides.strides.count
        && boundsAndStrides.begin.count == boundsAndStrides.processingSizes.count)
    for i in 0..<boundsAndStrides.processingSizes.count {
      let processingDimSize = boundsAndStrides.processingSizes[i]
      if boundsAndStrides.strides[i] > 0 {
        let edgePaddingLow = boundsAndStrides.begin[i]
        let interiorPadding = boundsAndStrides.strides[i] - 1
        // Pad the upper dimension up to the expected input shape. (It's not sufficient simply to
        // use "end[i]" to compute the padding in cases where the stride does not divide evenly into
        // the interval between begin[i] and end[i].)
        let size = edgePaddingLow + processingDimSize + (processingDimSize - 1) * interiorPadding
        let edgePaddingHigh = Int64(shape.scalars[i]) - size
        paddingConfig.append(
          PaddingConfigDimension(
            edge_padding_low: edgePaddingLow, edge_padding_high: edgePaddingHigh,
            interior_padding: interiorPadding))
      } else {
        dimensionsToReverse.append(Int64(i))
        let edgePaddingHigh = Int64(shape.scalars[i]) - boundsAndStrides.begin[i] - 1
        let interiorPadding = -boundsAndStrides.strides[i] - 1
        // Pad the lower dimension up to the expected input shape.
        let size = edgePaddingHigh + processingDimSize + (processingDimSize - 1) * interiorPadding
        let edgePaddingLow = Int64(shape.scalars[i]) - size
        paddingConfig.append(
          PaddingConfigDimension(
            edge_padding_low: edgePaddingLow, edge_padding_high: edgePaddingHigh,
            interior_padding: interiorPadding))
      }
    }
    if !dimensionsToReverse.isEmpty {
      grad = flip(grad, dims: dimensionsToReverse)
    }
    return xlaPad(grad, paddingValue: 0, paddingConfig: paddingConfig)
  }

  /// Computes the sum of elements across dimensions of a tensor.
  ///
  /// Reduces `input` along the dimensions given in `axis`. Unless
  /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  /// `axis`. If `keep_dims` is true, the reduced dimensions are
  /// retained with length 1.
  ///
  /// - Parameters:
  ///     - input: The tensor to reduce.
  ///     - reduction_indices: The dimensions to reduce. Must be in the range
  ///         `[-rank(input), rank(input))`.
  ///
  /// - Attr keep_dims: If true, retain reduced dimensions with length 1.
  ///
  /// - Output output: The reduced tensor.
  public static func sum<
    T: TensorFlowNumeric,
    Tidx: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    reductionIndices: Tensor<Tidx>,
    keepDims: Bool = false
  ) -> Tensor<T> {
    sum(input, reductionIndices: reductionIndices.scalars.map { Int64($0) }, keepDims: keepDims)
  }

  /// Assign `value` to the sliced l-value reference of `input`.
  ///
  /// The values of `value` are assigned to the positions in the tensor `input` that
  /// are selected by the slice parameters. The slice parameters `begin` `end`
  /// `strides` etc. work exactly as in `StridedSlice`.
  ///
  /// NOTE this op currently does not support broadcasting and so `value`'s shape
  /// must be exactly the shape produced by the slice of `input`.
  public static func tensorStridedSliceUpdate<
    T: TensorFlowScalar,
    Index: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    begin: Tensor<Index>,
    end: Tensor<Index>,
    strides: Tensor<Index>,
    value: Tensor<T>,
    beginMask: Int64 = 0,
    endMask: Int64 = 0,
    ellipsisMask: Int64 = 0,
    newAxisMask: Int64 = 0,
    shrinkAxisMask: Int64 = 0
  ) -> Tensor<T> {
    checkSameDevice(input, value)
    checkSamePrecision(input, value)
    let boundsAndStrides = XLATensor.computeIndexingBoundsAndStrides(
      inputSizes: input.shape.dimensions.map { Int64($0) }, begin: begin.scalars.map { Int64($0) },
      end: end.scalars.map { Int64($0) },
      strides: strides.scalars.map { Int64($0) }, beginMask: Int32(beginMask),
      endMask: Int32(endMask), ellipsisMask: Int32(ellipsisMask), newAxisMask: Int32(newAxisMask),
      shrinkAxisMask: Int32(shrinkAxisMask))
    if boundsAndStrides.finalSizes.reduce(1, *) == 0 || input.shape.dimensions.reduce(1, *) == 0 {
      // DynamicUpdateSlice does not allow 0-element updates. We should probably check that
      // rhs_shape can be broadcast to final_shape, but that is probably better handled when
      // implementing broadcasting more generally.
      return input
    }
    if boundsAndStrides.finalSizes.map({ Int($0) }) != value.shape.dimensions {
      fatalError(
        "Sliced l-value shape \(boundsAndStrides.finalSizes) does not match r-value shape"
          + " \(value.shape.dimensions). Automatic broadcasting not yet implemented.")
    }
    var dimensionsToReverse: [Int64] = []
    var sliceBegin: [Int64] = []
    var sliceDims: [Int64] = []
    assert(
      boundsAndStrides.begin.count == boundsAndStrides.end.count
        && boundsAndStrides.begin.count == boundsAndStrides.strides.count)
    for i in 0..<boundsAndStrides.begin.count {
      // TODO(b/121179231): implement strides != 1
      if Swift.abs(boundsAndStrides.strides[i]) != 1 {
        fatalError("Strides != 1 or -1 are not yet implemented")
      }
      if boundsAndStrides.strides[i] > 0 {
        sliceBegin.append(boundsAndStrides.begin[i])
        sliceDims.append(boundsAndStrides.end[i] - boundsAndStrides.begin[i])
      } else {
        sliceBegin.append(boundsAndStrides.end[i] + 1)
        sliceDims.append(boundsAndStrides.begin[i] - boundsAndStrides.end[i])
        dimensionsToReverse.append(Int64(i))
      }
    }
    var rhs = value
    if !dimensionsToReverse.isEmpty {
      rhs = flip(rhs, dims: dimensionsToReverse)
    }
    rhs = resize_value(rhs, dims: sliceDims)
    return updateSlice(input: input, source: rhs, baseIndices: sliceBegin)
  }

  /// Constructs a tensor by tiling a given tensor.
  ///
  /// This operation creates a new tensor by replicating `input` `multiples` times.
  /// The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
  /// and the values of `input` are replicated `multiples[i]` times along the 'i'th
  /// dimension. For example, tiling `[a b c d]` by `[2]` produces
  /// `[a b c d a b c d]`.
  ///
  /// - Parameters:
  ///     - input: 1-D or higher.
  ///     - multiples: 1-D. Length must be the same as the number of dimensions in `input`
  public static func tile<
    T: TensorFlowScalar,
    Tmultiples: TensorFlowIndex
  >(
    _ input: Tensor<T>,
    multiples: Tensor<Tmultiples>
  ) -> Tensor<T> {
    guard multiples.rank == 1 else {
      fatalError("Expected multiples to be 1-D, but got shape \(multiples.shape)")
    }
    return tile(input, multiples: multiples.scalars.map { Int($0) })
  }

  public static func tile<
    T: TensorFlowScalar
  >(
    _ input: Tensor<T>,
    multiples: [Int]
  ) -> Tensor<T> {
    guard input.rank == multiples.count else {
      fatalError(
        "Expected multiples argument to be a vector of length \(input.rank) but got length "
          + String(multiples.count)
      )
    }
    for (index, multiply) in multiples.enumerated() {
      guard multiply >= 0 else {
        fatalError("Expected multiples[\(index)] >= 0, but got \(multiply)")
      }
    }
    return tile(input, multiples: multiples.map { Int64($0) })
  }

  /// Transfer a tensor to a different device.
  public static func toDevice<T: TensorFlowScalar>(_ x: Tensor<T>, _ device: Device) -> Tensor<T> {
    Tensor(_xla: XLATensor.to(x.xlaTensor, device, T.self))
  }

  /// Shuffle dimensions of x according to a permutation.
  ///
  /// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
  ///   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
  public static func transpose<
    T: TensorFlowScalar,
    Tperm: TensorFlowIndex
  >(
    _ x: Tensor<T>,
    perm: Tensor<Tperm>
  ) -> Tensor<T> {
    return permute(x, dims: perm.scalars.map(Int64.init))
  }

  public static func transpose<
    T: TensorFlowScalar
  >(
    _ x: Tensor<T>,
    perm: [Int]
  ) -> Tensor<T> {
    return permute(x, dims: perm.map(Int64.init))
  }

  /// Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.
  ///
  /// Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
  /// For example, given a tensor of shape `(A, B, C, D)`;
  ///
  /// If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
  ///   and each tensor in `output` will have shape `(B, C, D)`. (Note that the
  ///   dimension unpacked along is gone, unlike `split`).
  ///
  /// If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
  ///   and each tensor in `output` will have shape `(A, C, D)`.
  /// Etc.
  ///
  /// This is the opposite of `pack`.
  ///
  /// - Parameter value: 1-D or higher, with `axis` dimension size equal to `num`.
  ///
  /// - Attr axis: Dimension along which to unpack.  Negative values wrap around, so the
  ///     valid range is `[-R, R)`.
  ///
  /// - Output output: The list of tensors unpacked from `value`.
  public static func unpack<T: TensorFlowScalar>(
    value: Tensor<T>,
    num: Int64,
    axis: Int64 = 0
  ) -> [Tensor<T>] {
    return (0..<num).map { select(value, dim: axis, index: $0) }
  }

  /// Computes the sum along segments of a tensor.
  ///
  /// Read
  /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  /// for an explanation of segments.
  ///
  /// Computes a tensor such that
  /// \\(output[i] = \sum_{j...} data[j...]\\) where the sum is over tuples `j...` such
  /// that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
  /// need not be sorted and need not cover all values in the full
  /// range of valid values.
  ///
  /// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
  /// If the given segment ID `i` is negative, the value is dropped and will not be
  /// added to the sum of the segment.
  ///
  /// `num_segments` should equal the number of distinct segment IDs.
  ///
  /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  /// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
  /// </div>
  ///
  /// ``` python
  /// c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  /// tf.unsorted_segment_sum(c, tf.constant([0, 1, 0]), num_segments=2)
  /// # ==> [[ 5,  5, 5, 5],
  /// #       [5,  6, 7, 8]]
  /// ```
  ///
  ///
  /// - Parameter segment_ids: A tensor whose shape is a prefix of `data.shape`.
  ///
  /// - Output output: Has same shape as data, except for the first `segment_ids.rank`
  ///     dimensions, which are replaced with a single dimension which has size
  ///     `num_segments`.
  public static func unsortedSegmentSum<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex,
    Tnumsegments: TensorFlowIndex
  >(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>,
    numSegments: Tensor<Tnumsegments>
  ) -> Tensor<T> {
    unsortedSegmentSum(
      data: data, segmentIds: segmentIds, numSegments: Int(numSegments.scalarized()))
  }

  public static func unsortedSegmentSum<
    T: TensorFlowNumeric,
    Tindices: TensorFlowIndex
  >(
    data: Tensor<T>,
    segmentIds: Tensor<Tindices>,
    numSegments: Int
  ) -> Tensor<T> {
    checkSameDevice(data.device, segmentIds.device)
    if segmentIds.rank > data.rank {
      fatalError(
        "UnsortedSegmentSum requires that indices' rank be less than or equal to data's rank.")
    }
    // Validate that segmentIds shape is a prefix of data shape.
    for dim in 0..<segmentIds.rank {
      if data.shape.dimensions[dim] != segmentIds.shape.dimensions[dim] {
        fatalError(
          "UnsortedSegmentSum requires indices shape to be prefix of data_shape, but dimension"
            + " \(dim) differs: \(data.shape.dimensions[dim]) vs. "
            + String(segmentIds.shape.dimensions[dim]))
      }
    }
    return tf_UnsortedSegmentSum(data, indicies: segmentIds, numSegments: Int64(numSegments))
  }

  /// Returns 0 if x == 0, and x / y otherwise, elementwise.
  public static func xdivy<T: FloatingPoint & TensorFlowScalar>(
    _ x: Tensor<T>,
    _ y: Tensor<T>
  ) -> Tensor<T> {
    checkSameDevice(x, y)
    checkSamePrecision(x, y)
    let (axla, bxla) = XLATensor.broadcast_tensors(x.xlaTensor, y.xlaTensor)
    let xbroadcast = Tensor<T>(_xla: axla)
    let ybroadcast = Tensor<T>(_xla: bxla)
    let zero = _RawXLA.zerosLike(xbroadcast)
    return _RawXLA.select(
      condition: _RawXLA.equal(xbroadcast, zero),
      t: zero,
      e: _RawXLA.div(xbroadcast, ybroadcast))
  }

  /// Returns a tensor of zeros with the same shape and type as x.
  ///
  /// - Parameter x: a tensor of type T.
  ///
  /// - Output y: a tensor of the same shape and type as x but filled with zeros.
  public static func zerosLike<T: TensorFlowScalar>(
    _ x: Tensor<T>
  ) -> Tensor<T> {
    return fullLike(0, x)
  }

  // Currently only used for deterministic testing.
  public static func rand(_ dims: [Int], _ seed: Int) -> Tensor<Float> {
    Tensor(_xla: XLATensor.rand(dims.map { Int64($0) }, Int64(seed)))
  }
}
