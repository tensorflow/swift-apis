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


public extension Tensor where Scalar: TensorFlowNumeric {
    /// Returns the batched diagonal part of a batched tensor.
    ///
    /// For example:
    ///
    /// ```
    /// // 't' is [[1, 0, 0, 0]
    /// //         [0, 2, 0, 0]
    /// //         [0, 0, 3, 0]
    /// //         [0, 0, 0, 4]]
    /// t.diagonalPart()
    /// // [1, 2, 3, 4]
    /// ```
    @inlinable
    @differentiable(wrt: self, vjp: _vjpDiagonalPart where Scalar: TensorFlowFloatingPoint)
    var diagonalPart: Tensor {
        _Raw.matrixDiagPart(self)
    }
    
    /// Constructs a diagonal array.
    ///
    /// For example:
    ///
    /// ```
    /// // 't' is [1, 2, 3, 4]
    ///
    /// t.diagonal()
    /// // [[1, 0, 0, 0]
    /// //  [0, 2, 0, 0]
    /// //  [0, 0, 3, 0]
    /// //  [0, 0, 0, 4]]
    /// ```
    @inlinable
    @differentiable(wrt: self, vjp: _vjpDiagonal where Scalar: TensorFlowFloatingPoint)
    var diagonal: Tensor {
        _Raw.matrixDiag(diagonal: self)
    }
    
    
    /// Returns a copy of a innermost tensor defined by a central band boundaries.
    ///
    /// For example:
    ///
    /// ```
    /// // 't' is [[ 0,  1,  2, 3]
    /// //         [-1,  0,  1, 2]
    /// //         [-2, -1,  0, 1]
    /// //         [-3, -2, -1, 0]]
    ///
    /// t.bandPart(1, -1)
    /// // [[ 0,  1,  2, 3]
    /// //  [-1,  0,  1, 2]
    /// //  [ 0, -1,  0, 1]
    /// //  [ 0,  0, -1, 0]]
    ///
    /// t.bandPart(2, 1)
    /// // [[ 0,  1,  0, 0]
    /// //  [-1,  0,  1, 0]
    /// //  [-2, -1,  0, 1]
    /// //  [ 0, -2, -1, 0]]
    /// ```
    @inlinable
    @differentiable(wrt: self, vjp: _vjpBandPart where Scalar: TensorFlowFloatingPoint)
    func bandPart(lowerCount: Int, upperCount: Int) -> Tensor {
        let lower = Tensor<Int32>(Int32(numLower))
        let upper = Tensor<Int32>(Int32(numUpper))
        return _Raw.matrixBandPart(self, numLower: lower, numUpper: upper)
    }
}


internal extension Tensor where Scalar: TensorFlowFloatingPoint {
    @inlinable
    func _vjpDiagonalPart() -> (Tensor, (Tensor) -> Tensor) {
        (diagonalPart(), { $0.diagonal() })
    }
    
    @inlinable
    func _vjpDiagonal() -> (Tensor, (Tensor) -> Tensor) {
        (diagonal(), { $0.diagonalPart() })
    }
    
    @inlinable
    func _vjpBandPart(_ numLower: Int, _ numUpper: Int) -> (Tensor, (Tensor) -> Tensor) {
        (bandPart(numLower, numUpper), { $0.bandPart(numLower, numUpper) })
    }
}
