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

import CTensorFlow

//===------------------------------------------------------------------------------------------===//
// Runtime Checkers
//===------------------------------------------------------------------------------------------===//

/// These checks run in both debug and release modes (while assert() only runs in debug mode), to
/// help shake out more bugs and facilitate debugging in the early project phases. It can be
/// replaced with plain assert() later, when we have a more mature code base.
@usableFromInline
internal func internalConsistencyCheck(
    _ predicate: Bool,
    _ errMessage: String = "TF runtime assertion failure",
    file: StaticString = #file,
    line: UInt = #line
) {
    guard predicate else {
        fatalError(errMessage, file: file, line: line)
    }
}

@usableFromInline
internal func checkOk(
    _ s: CTFStatus?,
    file: StaticString = #file,
    line: UInt = #line
) {
    internalConsistencyCheck(
        TF_GetCode(s) == TF_OK,
        String(cString: TF_Message(s)),
        file: file,
        line: line)
}

//===------------------------------------------------------------------------------------------===//
// Type Aliases
//===------------------------------------------------------------------------------------------===//

// Before assigning a C pointer to one of the pointer type aliases below, caller should check that
// the pointer is not NULL.

/// The `TF_Session *` type.
@usableFromInline
internal typealias CTFSession = OpaquePointer

/// The `TF_Status *` type.
@usableFromInline
internal typealias CTFStatus = OpaquePointer

/// The `TF_Graph*` type.
@usableFromInline
internal typealias CTFGraph = OpaquePointer

/// The `TF_Function*` type.
@usableFromInline
internal typealias CTFFunction = OpaquePointer

/// The `TF_Tensor *` type.
@usableFromInline
internal typealias CTensor = OpaquePointer

/// The `TF_TensorHandle *` type.
///
/// - Note: This is public so that compiler generated code can read/write tensor handles when
///   calling runtime APIs.
public typealias CTensorHandle = OpaquePointer

/// The `TFE_Context *` type.
@usableFromInline
internal typealias CTFEContext = OpaquePointer

/// The `TFE_Op *` type.
@usableFromInline
internal typealias CTFEOp = OpaquePointer

/// The `TF_OperationDescription *` type.
@usableFromInline
internal typealias CTFOperationDescription = OpaquePointer

/// The `TFE_TraceContext *` type.
@usableFromInline
internal typealias CTFETraceContext = OpaquePointer

//===------------------------------------------------------------------------------------------===//
// Logging
//===------------------------------------------------------------------------------------------===//

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
@usableFromInline internal let stderr = __stderrp
@usableFromInline internal  let stdout = __stdoutp
#endif

/// Log to standard error.
@usableFromInline
internal func logToStderr(_ message: StaticString) {
    message.utf8Start.withMemoryRebound(to: Int8.self, capacity: message.utf8CodeUnitCount) {
        _ = fputs($0, stderr)
    }
}

/// Log to standard error.
@usableFromInline
internal func logToStderr(_ message: String) {
    _ = fputs(message, stderr)
}

@usableFromInline
internal func debugLog(
    _ message: @autoclosure () -> String,
    file: StaticString = #file,
    line: UInt = #line
) {
    if _RuntimeConfig.printsDebugLog {
        print("[\(file):\(line)] \(message())")
        // This helps dump more log before a crash.
        fflush(stdout)
    }
}

//===------------------------------------------------------------------------------------------===//
// File Writing
//===------------------------------------------------------------------------------------------===//

/// Given the address of a `TF_Buffer` and a file path, write the buffer's contents to the file.
@usableFromInline
internal func writeContents(of buffer: UnsafePointer<TF_Buffer>, toFile path: String) {
    let fp = fopen(path, "w+")
    fwrite(buffer.pointee.data, /*size*/ 1, /*count*/ buffer.pointee.length, fp)
    fclose(fp)
}

//===------------------------------------------------------------------------------------------===//
// Unit Test Utilities
//===------------------------------------------------------------------------------------------===//

// TODO: Consider revising the call sites where this is necessary to only need UnsafeMutablePointer
// to optional when it is the actual c-api call site.
extension UnsafeMutablePointer where Pointee == CTensorHandle? {
    @usableFromInline
    init(_ other: UnsafeMutablePointer<CTensorHandle>) {
        self.init(other._rawValue)
    }

    @usableFromInline
    init?(_ other: UnsafeMutablePointer<CTensorHandle>?) {
        guard let unwrapped = other else { return nil }
        self.init(unwrapped)
    }
}

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
        var littleEndianValue = littleEndian
        return withUnsafePointer(to: &littleEndianValue) { pointer -> [UInt8] in
            let bytesPointer = UnsafeMutablePointer<UInt8>(OpaquePointer(pointer))
            var bytes = [UInt8](repeating: 0, count: byteCount)
            for i in 0..<Swift.min(MemoryLayout<Self>.size, byteCount) {
                bytes[byteCount - 1 - i] = (bytesPointer + i).pointee
            }
            return bytes
        }
    }

    fileprivate func rotate(rightBy count: Self) -> Self {
        (self >> count) | (self << (Self(MemoryLayout<Self>.size) * 8 - count))
    }
}

internal extension Array where Element == UInt8 {
    func sha512() -> [UInt8] {
        // First we define some useful constants.
        let blockSize = 128
        let digestLength = 64
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
        // Append one bit(`UInt8` with one bit) to the message.
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
            
            // Break chunk into sixteen 64-bit words M[j], 0 ≤ j ≤ 15, in big-endian format.
            // Extend the sixteen 64-bit words into eighty 64-bit words:
            var M = [UInt64](repeating: 0, count: k.count)
            for x in k.indices {
                switch x {
                case 0...15:
                    let start = chunk.startIndex.advanced(by: x * 8)
                    M[x] = UInt64(bytes: chunk, startingAt: start)
                    break
                default:
                    let s0 = M[x - 15].rotate(rightBy: 1) ^ 
                        M[x - 15].rotate(rightBy: 8) ^ 
                        (M[x - 15] >> 7)
                    let s1 = M[x - 2].rotate(rightBy: 19) ^ 
                        M[x - 2].rotate(rightBy: 61) ^ 
                        (M[x - 2] >> 6)
                    M[x] = M[x - 16] &+ s0 &+ M[x - 7] &+ s1
                    break
                }
            }

            var hashCopy = accumulatedHash
            for j in k.indices {
                let s0 = hashCopy[0].rotate(rightBy: 28) ^
                    hashCopy[0].rotate(rightBy: 34) ^
                    hashCopy[0].rotate(rightBy: 39)
                let maj = (hashCopy[0] & hashCopy[1]) ^ 
                    (hashCopy[0] & hashCopy[2]) ^
                    (hashCopy[1] & hashCopy[2])
                let t2 = s0 &+ maj
                let s1 = hashCopy[4].rotate(rightBy: 14) ^
                    hashCopy[4].rotate(rightBy: 18) ^
                    hashCopy[4].rotate(rightBy: 41)
                let ch = (hashCopy[4] & hashCopy[5]) ^ ((~hashCopy[4]) & hashCopy[6])
                let t1 = hashCopy[7] &+ s1 &+ ch &+ k[j] &+ UInt64(M[j])
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
        var result = [UInt8](repeating: 0, count: digestLength)
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
