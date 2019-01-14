// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

public protocol Optimizer : AnyObject {
    associatedtype Model: Layer
    associatedtype Scalar: FloatingPoint
    var learningRate: Scalar { get }
    func fit(_ model: inout Model, along gradient: Model.CotangentVector)
}

public class SGD<Model: Layer, Scalar: BinaryFloatingPoint & TensorFlowScalar>: Optimizer
    where Model.AllDifferentiableVariables: AdditiveArithmetic,
          Model.CotangentVector == Model.AllDifferentiableVariables {
    public let learningRate: Scalar
    public let momentum: Scalar
    public let decay: Scalar
    public let nesterov: Bool

    public init(
        learningRate: Scalar = 0.01,
        momentum: Scalar = 0,
        decay: Scalar = 0,
        nesterov: Bool = false
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(momentum >= 0, "Momentum must be non-negative")
        precondition(decay >= 0, "Weight decay must be non-negative")

        self.learningRate = learningRate
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov
    }

    private var velocity = Model.AllDifferentiableVariables.zero

    public func fit(_ model: inout Model, along gradients: Model.CotangentVector) {
        for kp in model.allDifferentiableVariables
                       .recursivelyAllWritableKeyPaths(to: Tensor<Scalar>.self) {
            velocity[keyPath: kp] =
                momentum * velocity[keyPath: kp] - learningRate * gradients[keyPath: kp]
            let modelKP = (\Model.allDifferentiableVariables).appending(path: kp)
            if nesterov {
                model[keyPath: modelKP] += velocity[keyPath: kp]
            } else {
                model[keyPath: modelKP] = model[keyPath: modelKP] + momentum *
                    velocity[keyPath: kp] - learningRate * gradients[keyPath: kp]
            }
        }
    }
}

public class RiemannSGD<Model: Layer, Scalar: FloatingPoint> : Optimizer
    where Model.TangentVector: VectorNumeric, Model.TangentVector.Scalar == Scalar {
    public var learningRate: Scalar

    public init(learningRate: Scalar) {
        self.learningRate = learningRate
    }

    public func fit(_ model: inout Model, along gradient: Model.CotangentVector) {
        model = model.moved(along: learningRate * (.zero - model.tangentVector(from: gradient)))
    }
}
