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
