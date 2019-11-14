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

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

//===------------------------------------------------------------------------------------------===//
// Hashing
//===------------------------------------------------------------------------------------------===//

internal extension FixedWidthInteger {
    init(bytes: ArraySlice<UInt8>, startingAt index: Int) {
        if bytes.isEmpty { self.init(0); return }
        let count = bytes.count
        self.init(0)
        for i in 0..<MemoryLayout<Self>.size {
            let j = (MemoryLayout<Self>.size - i - 1) * 8
            self |= count > 0 ? Self(bytes[index.advanced(by: i)]) << j : 0
        }
    }

    func bytes(count byteCount: Int = MemoryLayout<Self>.size) -> [UInt8] {
        let actualByteCount = Swift.min(MemoryLayout<Self>.size, byteCount)
        var littleEndianValue = littleEndian
        return withUnsafePointer(to: &littleEndianValue) {
            $0.withMemoryRebound(to: UInt8.self, capacity: actualByteCount) { pointer in
                var bytes = [UInt8](repeating: 0, count: byteCount)
                for i in 0..<actualByteCount {
                    bytes[byteCount - 1 - i] = (pointer + i).pointee
                }
                return bytes
            }
        }
    }
}

internal extension Array where Element == UInt8 {
    /// - Note: The SHA1 hash is only 20 bytes long and so only the first 20 bytes of the returned
    ///   `SIMD32<UInt8>` are non-zero.
    func sha1() -> SIMD32<UInt8> {
        let blockSize = 64
        var accumulated = self
        let lengthInBits = accumulated.count * 8
        let lengthBytes = lengthInBits.bytes(count: blockSize / 8)
        
        // Step 1: Append padding.
        let msgLength = accumulated.count
        // Append one bit (`UInt8` with one bit) to the message.
        accumulated.append(0x80)
        // Append `0` bits until the length of `accumulated` in bits is 448 (mod 512).
        let max = blockSize * 7 / 8
        accumulated += [UInt8](
        repeating: 0,
        count: msgLength % blockSize < max ?
            max - 1 - (msgLength % blockSize) :
            blockSize + max - 1 - (msgLength % blockSize))

        // Step 2: Append the message length as a 64-bit representation of `lengthInBits`.
        accumulated += lengthBytes

        // Step 3: Process the array bytes.
        var accumulatedHash = SIMD8<UInt32>([
            0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0, 0x00, 0x00, 0x00])
        var index = 0
        while index < accumulated.count {
            let chunk = accumulated[index..<(index + blockSize)]
            index += blockSize

            // Break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15, in big-endian format.
            // Extend the sixteen 32-bit words into eighty 32-bit words:
            var w = [UInt32](repeating: 0, count: 80)
            for x in w.indices {
                switch x {
                case 0...15:
                    let start = chunk.startIndex.advanced(by: x * 4)
                    w[x] = UInt32(bytes: chunk, startingAt: start)
                    break
                default:
                    let term = w[x - 3] ^ w[x - 8] ^ w[x - 14] ^ w[x - 16]
                    w[x] = term << 1 ^ term >> 31
                    break
                }
            }

            var hashCopy = accumulatedHash
            for j in w.indices {
                var f: UInt32 = 0
                var k: UInt32 = 0
                switch j {
                case 0...19:
                    f = (hashCopy[1] & hashCopy[2]) | (~hashCopy[1] & hashCopy[3])
                    k = 0x5a827999
                    break
                case 20...39:
                    f = hashCopy[1] ^ hashCopy[2] ^ hashCopy[3]
                    k = 0x6ed9eba1
                    break
                case 40...59:
                    f = (hashCopy[1] & hashCopy[2]) | 
                        (hashCopy[1] & hashCopy[3]) |
                        (hashCopy[2] & hashCopy[3])
                    k = 0x8f1bbcdc
                    break
                default:
                    f = hashCopy[1] ^ hashCopy[2] ^ hashCopy[3]
                    k = 0xca62c1d6
                    break
                }
                let temp = hashCopy[0] << 5 ^ hashCopy[0] >> 27
                let t0 = temp &+ f &+ hashCopy[4] &+ w[j] &+ k
                hashCopy[4] = hashCopy[3]
                hashCopy[3] = hashCopy[2]
                hashCopy[2] = hashCopy[1] << 30 ^ hashCopy[1] >> 2
                hashCopy[1] = hashCopy[0]
                hashCopy[0] = t0
            }
            accumulatedHash &+= hashCopy
        }

        // Step 4: Return the computed hash.
        var result = SIMD32<UInt8>()
        var position = 0
        for index in accumulatedHash.indices {
            let h = accumulatedHash[index]
            result[position + 0] = UInt8((h >> 24) & 0xff)
            result[position + 1] = UInt8((h >> 16) & 0xff)
            result[position + 2] = UInt8((h >> 8) & 0xff)
            result[position + 3] = UInt8(h & 0xff)
            position += 4
        }

        return result
    }

    func sha512() -> SIMD64<UInt8> {
        // First we define some useful constants.
        let blockSize = 128
        let k: [UInt64] = [
            0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
            0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
            0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
            0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
            0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
            0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
            0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
            0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
            0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
            0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
            0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
            0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
            0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
            0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
            0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
            0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
            0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
            0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
            0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
            0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817]

        var accumulated = self
        let lengthInBits = accumulated.count * 8
        let lengthBytes = lengthInBits.bytes(count: blockSize / 8)
        
        // Step 1: Append padding.
        let msgLength = accumulated.count
        // Append one bit (`UInt8` with one bit) to the message.
        accumulated.append(0x80)
        // Append `0` bits until the length of `accumulated` in bits is 448 (mod 512).
        let max = blockSize * 7 / 8
        accumulated += [UInt8](
            repeating: 0,
            count: msgLength % blockSize < max ?
                max - 1 - (msgLength % blockSize) :
                blockSize + max - 1 - (msgLength % blockSize))

        // Step 2: Append the message length as a 64-bit representation of `lengthInBits`.
        accumulated += lengthBytes

        // Step 3: Process the array bytes.
        var accumulatedHash = SIMD8<UInt64>(
            0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1, 
            0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179)
        var index = 0
        while index < accumulated.count {
            let chunk = accumulated[index..<(index + blockSize)]
            index += blockSize

            // Break chunk into sixteen 64-bit words w[j], 0 ≤ j ≤ 15, in big-endian format.
            // Extend the sixteen 64-bit words into eighty 64-bit words:
            var w = [UInt64](repeating: 0, count: k.count)
            for x in w.indices {
                switch x {
                case 0...15:
                    let start = chunk.startIndex.advanced(by: x * 8)
                    w[x] = UInt64(bytes: chunk, startingAt: start)
                    break
                default:
                    let s0Term0 = ((w[x - 15] >> 1 ^ w[x - 15]) >> 6 ^ w[x - 15]) >> 1
                    let s0Term1 = (w[x - 15] << 7 ^ w[x - 15]) << 56
                    let s0 = s0Term0 ^ s0Term1
                    let s1Term0 = ((w[x - 2] >> 42 ^ w[x - 2]) >> 13 ^ w[x - 2]) >> 6
                    let s1Term1 = (w[x - 2] << 42 ^ w[x - 2]) << 3
                    let s1 = s1Term0 ^ s1Term1
                    w[x] = w[x - 16] &+ s0 &+ w[x - 7] &+ s1
                    break
                }
            }

            var hashCopy = accumulatedHash
            for j in w.indices {
                let s0Term0 = ((hashCopy[0] >> 5 ^ hashCopy[0]) >> 6 ^ hashCopy[0]) >> 28
                let s0Term1 = ((hashCopy[0] << 6 ^ hashCopy[0]) << 5 ^ hashCopy[0]) << 25
                let s0 = s0Term0 ^ s0Term1
                let s1Term0 = ((hashCopy[4] >> 23 ^ hashCopy[4]) >> 4 ^ hashCopy[4]) >> 14
                let s1Term1 = ((hashCopy[4] << 4 ^ hashCopy[4]) << 23 ^ hashCopy[4]) << 23
                let s1 = s1Term0 ^ s1Term1
                let maj = (hashCopy[0] & hashCopy[1]) ^ 
                    (hashCopy[0] & hashCopy[2]) ^
                    (hashCopy[1] & hashCopy[2])
                let t2 = s0 &+ maj
                let ch = (hashCopy[4] & hashCopy[5]) ^ (~hashCopy[4] & hashCopy[6])
                let t1 = hashCopy[7] &+ s1 &+ ch &+ k[j] &+ w[j]
                hashCopy[7] = hashCopy[6]
                hashCopy[6] = hashCopy[5]
                hashCopy[5] = hashCopy[4]
                hashCopy[4] = hashCopy[3] &+ t1
                hashCopy[3] = hashCopy[2]
                hashCopy[2] = hashCopy[1]
                hashCopy[1] = hashCopy[0]
                hashCopy[0] = t1 &+ t2
            }
            accumulatedHash &+= hashCopy
        }

        // Step 4: Return the computed hash.
        var result = SIMD64<UInt8>()
        var position = 0
        for index in accumulatedHash.indices {
            let h = accumulatedHash[index]
            result[position + 0] = UInt8((h >> 56) & 0xff)
            result[position + 1] = UInt8((h >> 48) & 0xff)
            result[position + 2] = UInt8((h >> 40) & 0xff)
            result[position + 3] = UInt8((h >> 32) & 0xff)
            result[position + 4] = UInt8((h >> 24) & 0xff)
            result[position + 5] = UInt8((h >> 16) & 0xff)
            result[position + 6] = UInt8((h >> 8) & 0xff)
            result[position + 7] = UInt8(h & 0xff)
            position += 8
        }

        return result
    }
}
