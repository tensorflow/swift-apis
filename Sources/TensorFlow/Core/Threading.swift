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
  import WinSDK
#else
  import Glibc
#endif

struct ThreadLocalStorage {
  struct Key {
    #if os(Windows)
      var _value: DWORD
    #else
      var _value: pthread_key_t
    #endif

    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      typealias KeyDestructor = @convention(c) (UnsafeMutableRawPointer) -> Void
    #else
      typealias KeyDestructor = @convention(c) (UnsafeMutableRawPointer?) -> Void
    #endif

    init(destructor: KeyDestructor?) {
      #if os(Windows)
        _value = FlsAlloc(destructor)
      #else
        _value = pthread_key_t()
        pthread_key_create(&_value, destructor)
      #endif
    }
  }

  public static func get(for key: Key) -> UnsafeMutableRawPointer? {
    #if os(Windows)
      return FlsGetValue(key._value)
    #else
      return pthread_getspecific(key._value)
    #endif
  }

  public static func set(value: UnsafeMutableRawPointer?, for key: Key) {
    #if os(Windows)
      FlsSetValue(key._value, value)
    #else
      pthread_setspecific(key._value, value)
    #endif
  }
}

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
    let body: () -> Void

    init(_ body: @escaping () -> Void) {
      self.body = body
    }

    func run() {
      self.body()
    }
  }

  var thread: Handle = _invalidHandle()

  init(perform body: @escaping () -> Void) {
    let context = Unmanaged.passRetained(Procedure(body)).toOpaque()
    #if os(Windows)
      let status = _beginthreadex(
        nil, 0,
        {
          let procedure: Thread.Procedure =
            Unmanaged.fromOpaque($0!).takeRetainedValue()
          procedure.run()
          return 0
        }, context, 0, nil)
    #else
      let status = pthread_create(
        &self.thread, nil,
        {
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

/// A portable mutex for synchronization of a shared resource.
class Mutex {
  #if os(Windows)
    typealias MutexType = SRWLOCK
  #else
    typealias MutexType = pthread_mutex_t
  #endif

  var _mutex: MutexType

  init() {
    _mutex = MutexType()
    #if os(Windows)
      InitializeSRWLock(&_mutex)
    #else
      pthread_mutex_init(&_mutex, nil)
    #endif
  }

  deinit {
    #if os(Windows)
      // SRWLOCKs do not need explicit destruction
    #else
      pthread_mutex_destroy(&_mutex)
    #endif
  }

  // Acquire the mutex.
  //
  // Calling this function will block until it is safe to access the resource
  // that the mutex is protecting, locking the mutex indicating ownership of
  // the shared resource.
  //
  // Returns 0 on success.
  func acquire() -> Int32 {
    #if os(Windows)
      AcquireSRWLockExclusive(&_mutex)
      return 0
    #else
      return pthread_mutex_lock(&_mutex)
    #endif
  }

  // Release the mutex.
  //
  // Calling this function unlocks the mutex, relinquishing control of the
  // shared resource.
  //
  // Returns 0 on success.
  func release() -> Int32 {
    #if os(Windows)
      ReleaseSRWLockExclusive(&_mutex)
      return 0
    #else
      return pthread_mutex_unlock(&_mutex)
    #endif
  }
}

public func _runOnNDevices(_ n: Int, perform body: @escaping (Int) -> Void) {
  var threads = [] as [Thread]
  for i in 0..<n {
    threads.append(
      Thread {
        body(i)
      })
  }
  for t in threads {
    t.join()
  }
}
