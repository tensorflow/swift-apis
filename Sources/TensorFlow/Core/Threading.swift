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

// A thread that runs the provided body.
class Thread {
#if os(Windows)
    typealias Handle = HANDLE
#else
#if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
    typealias Handle = pthread_t?
#else
    typealias Handle = pthread_t
#endif
#endif

    static func _invalidHandle() -> Handle {
#if os(Windows)
      return INVALID_HANDLE_VALUE
#else
#if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      return nil
#else
      return pthread_t()
#endif
#endif
    }

    // NOTE: the `Procedure` abstraction is required as the use of the
    // `Unmanaged` requires a `class` type for the reference counting purposes.
    class Procedure {
        let body: () -> ()

        init(_ body: @escaping () -> ()) {
            self.body = body
        }

        func run() {
            self.body()
        }
    }

    var thread: Handle = _invalidHandle()

    init(perform body: @escaping () -> ()) {
        let context = Unmanaged.passRetained(Procedure(body)).toOpaque()
#if os(Windows)
        let status = _beginthreadex(nil, 0, {
            let procedure: Thread.Procedure =
                Unmanaged.fromOpaque($0!).takeRetainedValue()
            procedure.run()
            return 0
        }, context, 0, nil)
#else
        let status = pthread_create(&self.thread, nil, {
            // Set the cancelability of the detached thread.
            pthread_setcanceltype(Int32(PTHREAD_CANCEL_DEFERRED), nil)

#if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
            let procedure: Thread.Procedure =
                Unmanaged.fromOpaque($0).takeRetainedValue()
#else
            let procedure: Thread.Procedure =
                Unmanaged.fromOpaque($0!).takeRetainedValue()
#endif
            procedure.run()
            return nil
        }, context)
#endif
        internalConsistencyCheck(status == 0)
    }

    func join() {
#if os(Windows)
        let result = WaitForSingleObject(thread, INFINITE)
        internalConsistencyCheck(result == WAIT_OBJECT_0)
        CloseHandle(self.thread)
#else
#if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
        let status = pthread_join(thread!, nil)
#else
        let status = pthread_join(thread, nil)
#endif
        internalConsistencyCheck(status == 0)
#endif
    }
}

public func _runOnNDevices(_ n: Int, perform body: @escaping (Int) -> ()) {
    var threads = [] as [Thread]
    for i in 0..<n {
        threads.append(Thread {
            body(i)
        })
    }
    for t in threads {
        t.join()
    }
}
