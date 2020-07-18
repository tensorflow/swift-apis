
@available(
  *, deprecated, renamed: "_Raw",
  message:
    """
  'Raw' has been renamed to '_Raw' to indicate that it is not a guaranteed/stable API.
  """
)
public typealias Raw = _Raw

#if USING_X10_BACKEND
  public enum _Raw {
    static let generatedTensorFlowVersion = "2.1.0"
    static let generatedTensorFlowGitVersion = "v2.1.0-rc2-17-ge5bf8de"

    public typealias DataFormat = _RawTFEager.DataFormat
    public typealias DataFormat2 = _RawTFEager.DataFormat2
    public typealias Padding = _RawTFEager.Padding
    public typealias Padding1 = _RawTFEager.Padding1

    @inlinable @inline(__always)
    public static func abs<T: TensorFlowNumeric>(
      _ x: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func addV2<T: TensorFlowNumeric>(
      _ x: Tensor<T>,
      _ y: Tensor<T>
    ) -> Tensor<T> {
      if commonBackend(x.handle.backend, y.handle.backend) == .TF_EAGER {
        return _TFE_Op("AddV2", 1).execute()
      } else {
        fatalError()
      }
    }

    @inlinable @inline(__always)
    public static func broadcastGradientArgs<T: TensorFlowIndex>(
      s0: Tensor<T>,
      s1: Tensor<T>
    ) -> (r0: Tensor<T>, r1: Tensor<T>) {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func ceil<T: FloatingPoint & TensorFlowScalar>(
      _ x: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func floor<T: FloatingPoint & TensorFlowScalar>(
      _ x: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func maximum<T: TensorFlowNumeric>(
      _ x: Tensor<T>,
      _ y: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func minimum<T: TensorFlowNumeric>(
      _ x: Tensor<T>,
      _ y: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func pack<T: TensorFlowScalar>(
      _ values: [Tensor<T>],
      axis: Int64 = 0
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func round<T: TensorFlowNumeric>(
      _ x: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func rsqrt<T: FloatingPoint & TensorFlowScalar>(
      _ x: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func rsqrtGrad<T: FloatingPoint & TensorFlowScalar>(
      _ y: Tensor<T>,
      dy: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func select<T: TensorFlowScalar>(
      condition: Tensor<Bool>,
      t: Tensor<T>,
      e: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func sigmoid<T: FloatingPoint & TensorFlowScalar>(
      _ x: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func sigmoidGrad<T: FloatingPoint & TensorFlowScalar>(
      _ y: Tensor<T>,
      dy: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func sign<T: TensorFlowNumeric>(
      _ x: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func softmax<T: FloatingPoint & TensorFlowScalar>(
      logits: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func softmaxCrossEntropyWithLogits<T: FloatingPoint & TensorFlowScalar>(
      features: Tensor<T>,
      labels: Tensor<T>
    ) -> (loss: Tensor<T>, backprop: Tensor<T>) {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func softplus<T: FloatingPoint & TensorFlowScalar>(
      features: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func softplusGrad<T: FloatingPoint & TensorFlowScalar>(
      gradients: Tensor<T>,
      features: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func softsign<T: FloatingPoint & TensorFlowScalar>(
      features: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func softsignGrad<T: FloatingPoint & TensorFlowScalar>(
      gradients: Tensor<T>,
      features: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func square<T: TensorFlowNumeric>(
      _ x: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }

    @inlinable @inline(__always)
    public static func squaredDifference<T: TensorFlowNumeric>(
      _ x: Tensor<T>,
      _ y: Tensor<T>
    ) -> Tensor<T> {
      fatalError()
    }
  }
#else
  public typealias _Raw = _RawTFEager
#endif
