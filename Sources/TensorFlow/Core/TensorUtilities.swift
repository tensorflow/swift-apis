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
#elseif os(Windows)
  import ucrt
#else
  import Glibc
#endif

//===------------------------------------------------------------------------------------------===//
// Hashing
//===------------------------------------------------------------------------------------------===//

extension FixedWidthInteger {
  init(bytes: ArraySlice<UInt8>, startingAt index: Int) {
    if bytes.isEmpty {
      self.init(0)
      return
    }
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

extension Array where Element == UInt8 {
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
      count: msgLength % blockSize < max
        ? max - 1 - (msgLength % blockSize) : blockSize + max - 1 - (msgLength % blockSize))

    // Step 2: Append the message length as a 64-bit representation of `lengthInBits`.
    accumulated += lengthBytes

    // Step 3: Process the array bytes.
    var accumulatedHash = SIMD8<UInt32>([
      0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476, 0xc3d2_e1f0, 0x00, 0x00, 0x00,
    ])
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
          k = 0x5a82_7999
          break
        case 20...39:
          f = hashCopy[1] ^ hashCopy[2] ^ hashCopy[3]
          k = 0x6ed9_eba1
          break
        case 40...59:
          f =
            (hashCopy[1] & hashCopy[2]) | (hashCopy[1] & hashCopy[3]) | (hashCopy[2] & hashCopy[3])
          k = 0x8f1b_bcdc
          break
        default:
          f = hashCopy[1] ^ hashCopy[2] ^ hashCopy[3]
          k = 0xca62_c1d6
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
      0x428a_2f98_d728_ae22, 0x7137_4491_23ef_65cd, 0xb5c0_fbcf_ec4d_3b2f, 0xe9b5_dba5_8189_dbbc,
      0x3956_c25b_f348_b538, 0x59f1_11f1_b605_d019, 0x923f_82a4_af19_4f9b, 0xab1c_5ed5_da6d_8118,
      0xd807_aa98_a303_0242, 0x1283_5b01_4570_6fbe, 0x2431_85be_4ee4_b28c, 0x550c_7dc3_d5ff_b4e2,
      0x72be_5d74_f27b_896f, 0x80de_b1fe_3b16_96b1, 0x9bdc_06a7_25c7_1235, 0xc19b_f174_cf69_2694,
      0xe49b_69c1_9ef1_4ad2, 0xefbe_4786_384f_25e3, 0x0fc1_9dc6_8b8c_d5b5, 0x240c_a1cc_77ac_9c65,
      0x2de9_2c6f_592b_0275, 0x4a74_84aa_6ea6_e483, 0x5cb0_a9dc_bd41_fbd4, 0x76f9_88da_8311_53b5,
      0x983e_5152_ee66_dfab, 0xa831_c66d_2db4_3210, 0xb003_27c8_98fb_213f, 0xbf59_7fc7_beef_0ee4,
      0xc6e0_0bf3_3da8_8fc2, 0xd5a7_9147_930a_a725, 0x06ca_6351_e003_826f, 0x1429_2967_0a0e_6e70,
      0x27b7_0a85_46d2_2ffc, 0x2e1b_2138_5c26_c926, 0x4d2c_6dfc_5ac4_2aed, 0x5338_0d13_9d95_b3df,
      0x650a_7354_8baf_63de, 0x766a_0abb_3c77_b2a8, 0x81c2_c92e_47ed_aee6, 0x9272_2c85_1482_353b,
      0xa2bf_e8a1_4cf1_0364, 0xa81a_664b_bc42_3001, 0xc24b_8b70_d0f8_9791, 0xc76c_51a3_0654_be30,
      0xd192_e819_d6ef_5218, 0xd699_0624_5565_a910, 0xf40e_3585_5771_202a, 0x106a_a070_32bb_d1b8,
      0x19a4_c116_b8d2_d0c8, 0x1e37_6c08_5141_ab53, 0x2748_774c_df8e_eb99, 0x34b0_bcb5_e19b_48a8,
      0x391c_0cb3_c5c9_5a63, 0x4ed8_aa4a_e341_8acb, 0x5b9c_ca4f_7763_e373, 0x682e_6ff3_d6b2_b8a3,
      0x748f_82ee_5def_b2fc, 0x78a5_636f_4317_2f60, 0x84c8_7814_a1f0_ab72, 0x8cc7_0208_1a64_39ec,
      0x90be_fffa_2363_1e28, 0xa450_6ceb_de82_bde9, 0xbef9_a3f7_b2c6_7915, 0xc671_78f2_e372_532b,
      0xca27_3ece_ea26_619c, 0xd186_b8c7_21c0_c207, 0xeada_7dd6_cde0_eb1e, 0xf57d_4f7f_ee6e_d178,
      0x06f0_67aa_7217_6fba, 0x0a63_7dc5_a2c8_98a6, 0x113f_9804_bef9_0dae, 0x1b71_0b35_131c_471b,
      0x28db_77f5_2304_7d84, 0x32ca_ab7b_40c7_2493, 0x3c9e_be0a_15c9_bebc, 0x431d_67c4_9c10_0d4c,
      0x4cc5_d4be_cb3e_42b6, 0x597f_299c_fc65_7e2a, 0x5fcb_6fab_3ad6_faec, 0x6c44_198c_4a47_5817,
    ]

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
      count: msgLength % blockSize < max
        ? max - 1 - (msgLength % blockSize) : blockSize + max - 1 - (msgLength % blockSize))

    // Step 2: Append the message length as a 64-bit representation of `lengthInBits`.
    accumulated += lengthBytes

    // Step 3: Process the array bytes.
    var accumulatedHash = SIMD8<UInt64>(
      0x6a09_e667_f3bc_c908, 0xbb67_ae85_84ca_a73b, 0x3c6e_f372_fe94_f82b, 0xa54f_f53a_5f1d_36f1,
      0x510e_527f_ade6_82d1, 0x9b05_688c_2b3e_6c1f, 0x1f83_d9ab_fb41_bd6b, 0x5be0_cd19_137e_2179)
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
        let maj =
          (hashCopy[0] & hashCopy[1]) ^ (hashCopy[0] & hashCopy[2]) ^ (hashCopy[1] & hashCopy[2])
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
