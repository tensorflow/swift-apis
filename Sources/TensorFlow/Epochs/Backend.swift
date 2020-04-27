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

import Foundation

extension Collection {
  /// Returns `self.map(transform)`, computed in parallel on chunks of self 
  /// of size `minBatchSize` or `minBatchSize + 1`.
  ///
  /// - Requires: `transform` is safe to call from multiple threads.
  func concurrentMap<B>(
    minBatchSize: Int = 1,
    _ transform: (Element) -> B
  ) -> [B] {
    precondition(minBatchSize >= 1)
    let n = self.count
    let batchCount = (n + minBatchSize - 1) / minBatchSize
    if batchCount < 2 { return self.map(transform) }

    return Array(unsafeUninitializedCapacity: n) {
      uninitializedMemory, resultCount in
      resultCount = n
      let baseAddress = uninitializedMemory.baseAddress!

      DispatchQueue.concurrentPerform(iterations: batchCount) { b in
        let startOffset = b * n / batchCount
        let endOffset = (b + 1) * n / batchCount
        var sourceIndex = index(self.startIndex, offsetBy: startOffset)
        for p in baseAddress + startOffset..<baseAddress + endOffset {
          p.initialize(to: transform(self[sourceIndex]))
          formIndex(after: &sourceIndex)
        }
      }
    }
  }
}
