import x10_device

fileprivate func l2Norm(_ x: Tensor<Float>) -> Tensor<Float> {
  return sqrt(x.squared().sum())
}

/// Layerwise adaptive rate scaling (LARS) optimizer.
///
/// https://arxiv.org/abs/1708.03888
/// Yang You, Igor Gitman, Boris Ginsburg
public class LARS<Model: EuclideanDifferentiable & KeyPathIterable>: Optimizer
where
  Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions
    & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float
{
  public typealias Model = Model

  /// The learning rate.
  public var learningRate: Float

  public var keys: [WritableKeyPath<Model.TangentVector, Tensor<Float>>]

  /// The momentum factor. It accelerates stochastic gradient descent in the relevant direction
  /// and dampens oscillations.
  public var momentum: Float

  /// The trust coefficient for trust ratio computation
  public var trustCoefficient: Float

  /// Use Nesterov momentum if true.
  public var nesterov: Bool

  /// The velocity state of the model.
  public var velocity: Model.TangentVector = .zero

  /// Weight decay coefficient to apply.
  public var weightDecay: Float

  /// Epsilon used to avoid dividing by zero
  public var epsilon: Float

  public var step: Int = 0

  public init(
    for model: __shared Model,
    learningRate: Float = 0.01,
    keys: [WritableKeyPath<Model.TangentVector, Tensor<Float>>],
    momentum: Float = 0.9,
    trustCoefficient: Float = 0.001,
    nesterov: Bool = false,
    epsilon: Float = 0.0,
    weightDecay: Float = 0.0
  ) {
    precondition(learningRate >= 0, "Learning rate must be non-negative")
    precondition(momentum >= 0, "Momentum must be non-negative")
    precondition(weightDecay >= 0, "Weight decay must be non-negative")

    self.learningRate = learningRate
    self.keys = keys
    self.momentum = momentum
    self.trustCoefficient = trustCoefficient
    self.nesterov = nesterov
    self.velocity = model.differentiableVectorView
    for kp in velocity.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      velocity[keyPath: kp] = Tensor<Float>(zerosLike: velocity[keyPath: kp])
    }
    self.epsilon = epsilon
    self.weightDecay = weightDecay
  }

  public func update(_ model: inout Model, along direction: Model.TangentVector) {
    step += 1
    // Initialize step to `direction` (instead of `TangentVector.zero`) because otherwise the keypath returns an index error
    var step = direction
    for kp in keys {
      let param = model.differentiableVectorView[keyPath: kp]
      let device = param.device
      let paramNorm = l2Norm(param)
      let gradNorm = l2Norm(direction[keyPath: kp])
      let trustRatio = trustCoefficient * paramNorm / (gradNorm + weightDecay * paramNorm + epsilon)
      let clippedTrustRatio = _Raw.select(
        condition: (paramNorm + gradNorm) .> 0, t: trustRatio, e: Tensor<Float>(1.0, on: device))
      let decayedGrad =
        weightDecay == 0
        ? direction[keyPath: kp] : direction[keyPath: kp] + param.scaled(by: weightDecay)
      let scaledGrad = learningRate * clippedTrustRatio * decayedGrad
      velocity[keyPath: kp] = velocity[keyPath: kp] * momentum + scaledGrad
      step[keyPath: kp] =
        -(nesterov ? (scaledGrad + momentum * velocity[keyPath: kp]) : velocity[keyPath: kp])
    }
    model.move(along: step)
  }

  public required init(copying other: LARS, to device: Device) {
    learningRate = other.learningRate
    keys = other.keys
    momentum = other.momentum
    trustCoefficient = other.trustCoefficient
    nesterov = other.nesterov
    velocity = .init(copying: other.velocity, to: device)
    weightDecay = other.weightDecay
    epsilon = other.epsilon
  }
}
