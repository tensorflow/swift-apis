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

/// An embedding layer.
///
/// `Embedding` is effectively a lookup table that maps indices from a fixed vocabulary to fix sized
/// (dense) vector representations.
/// E.g. `[[0], [3]] -> [[0.25, 0.1], [0.6, -0.2]]`
public struct Embedding<Scalar: TensorFlowFloatingPoint>: Layer {
    /// A helper structure containing the indices to be passed into the `Embedding` layer.
    ///
    /// - Note: Often times, `Embedding` is followed by a `Flatten` and a `Dense` layer. When this 
    ///   is the case, ensure that all input sequences of indices have the same dimension.
    /// - Note: This structure is needed to conform `Embedding` to the Layer protocol.
    public struct Input: Differentiable {
        /// Sequences of indices that will be passed into the layer.
        @noDerivative var indices: Tensor<Int32>
    }
    /// A learnable lookup table that maps vocabulary indices to their dense vector representations.
    public var embeddings: Tensor<Scalar>

    /// Creates an `Embedding` layer with randomly initialized embeddings of shape 
    /// `(vocabularySize, embeddingSize)` so that each vocabulary index is given a vector 
    /// representation.
    ///
    /// - Parameters:
    ///   - vocabularySize: The number of distinct indices (words) in the vocabulary. This number
    ///     should be the `largest integer index + 1`.
    ///   - embeddingSize: The number of entries in a single embedding vector representation.
    public init(vocabularySize: Int, embeddingSize: Int) {
        self.embeddings = Tensor(randomUniform: [vocabularySize, embeddingSize])
    }

    /// Creates an `Embedding` layer from the provided embeddings. Useful for introducing 
    /// pretrained embeddings into a model.
    /// 
    /// - Parameter embeddings: The pretrained embeddings table.
    public init(from embeddings: Tensor<Scalar>) {
        self.embeddings = embeddings
    }

    /// Replaces each index in the input with corresponding dense vector representations.
    ///
    /// - Parameter
    ///   - input: The indices that will be mapped to their vector representations.
    /// - Returns:
    ///   - output: The tensor created by replacing input indices with their vector representations.
    @differentiable
    public func callAsFunction(_ input: Input) -> Tensor<Scalar> {
        return embeddings.gathering(atIndices: input.indices)
    }
}
