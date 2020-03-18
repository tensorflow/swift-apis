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

#if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
  import Darwin
#elseif os(Windows)
  import ucrt
#else
  import Glibc
#endif

#if !COMPILING_TENSORFLOW_STDLIB_MODULE
  import Tensor
#endif

/// A value that indicates the phase of using a machine learning model.
public enum LearningPhase {
  case training
  case inference
}

/// A context that stores thread-local contextual information used by deep learning APIs such as
/// layers.
///
/// Use `Context.local` to retrieve the current thread-local context.
///
/// Examples:
///
/// * Set the current learning phase to training so that layers like `BatchNorm` will
///   compute mean and variance when applied to inputs.
///
///   ```swift
///   Context.local.learningPhase = .training
///   ```
/// * Set the current learning phase to inference so that layers like `Dropout` will not drop out
///   units when applied to inputs.
///
///   ```swift
///   Context.local.learningPhase = .inference
///   ```
public struct Context {
  /// The learning phase.
  public var learningPhase: LearningPhase = .inference

  /// The random seed.
  ///
  /// - Note: Whenever obtained, the random seed is also updated so that future stateless 
  ///   random TensorFlow op executions will result in non-deterministic results.
  public var randomSeed: TensorFlowSeed {
    mutating get {
      let seed = _randomSeed
      _randomSeed = (seed.0, seed.1 + 1)
      return seed
    }
    set { _randomSeed = newValue }
  }

  private var _randomSeed: TensorFlowSeed = randomSeedForTensorFlow()

  /// The random number generator.
  internal var randomNumberGenerator: AnyRandomNumberGenerator =
    AnyRandomNumberGenerator(PhiloxRandomNumberGenerator(uint64Seed: UInt64(time(nil))))

  internal var globalTensorCount: Int = 0

  /// Creates a context with default properties.
  public init() {}

  /// The current thread-local context.
  ///
  /// - Note: Accessing this property is thread-safe.
  public static var local: Context {
    _read { yield ContextManager.local.currentContext }
    _modify { yield &ContextManager.local.currentContext }
  }
}

/// Calls the given closure within a context that has everything identical to the current context
/// except for the given learning phase.
///
/// - Parameters:
///   - context: A context that will be set before the closure gets called and restored after the
///     closure returns.
///   - body: A nullary closure. If the closure has a return value, that value is also used as the
///     return value of the `withContext(_:_:)` function.
/// - Returns: The return value, if any, of the `body` closure.
public func withContext<R>(_ context: Context, _ body: () throws -> R) rethrows -> R {
  ContextManager.local.push(context)
  defer { ContextManager.local.popContext() }
  return try body()
}

/// Calls the given closure within a context that has everything identical to the current context
/// except for the given learning phase.
///
/// - Parameters:
///   - learningPhase: A learning phase that will be set before the closure gets called and restored
///     after the closure returns.
///   - body: A nullary closure. If the closure has a return value, that value is also used as the
///     return value of the `withLearningPhase(_:_:)` function.
/// - Returns: The return value, if any, of the `body` closure.
public func withLearningPhase<R>(
  _ learningPhase: LearningPhase,
  _ body: () throws -> R
) rethrows -> R {
  var context = ContextManager.local.currentContext
  context.learningPhase = learningPhase
  return try withContext(context, body)
}

/// Calls the given closure within a context that has everything identical to the current context
/// except for the given random seed.
///
/// - Parameters:
///   - randomSeed: A random seed that will be set before the closure gets called and restored
///     after the closure returns.
///   - body: A nullary closure. If the closure has a return value, that value is also used as the
///     return value of the `withRandomSeedForTensorFlow(_:_:)` function.
/// - Returns: The return value, if any, of the `body` closure.
public func withRandomSeedForTensorFlow<R>(
  _ randomSeed: TensorFlowSeed,
  _ body: () throws -> R
) rethrows -> R {
  var context = ContextManager.local.currentContext
  context.randomSeed = randomSeed
  return try withContext(context, body)
}

/// Calls the given closure within a context that has everything identical to the current context
/// except for the given random number generator.
///
/// - Parameters:
///   - randomNumberGenerator: A random number generator that will be set before the closure gets 
///     called and restored after the closure returns.
///   - body: A nullary closure. If the closure has a return value, that value is also used as the
///     return value of the `withRandomNumberGeneratorForTensorFlow(_:_:)` function.
/// - Returns: The return value, if any, of the `body` closure.
public func withRandomNumberGeneratorForTensorFlow<G: RandomNumberGenerator, R>(
  _ randomNumberGenerator: inout G,
  _ body: () throws -> R
) rethrows -> R {
  var context = ContextManager.local.currentContext
  context.randomNumberGenerator = AnyRandomNumberGenerator(randomNumberGenerator)
  return try withContext(context, body)
}

/// A manager that maintains and provides safe access to thread-local `Context` values.
private final class ContextManager {
  var contextStack: [Context] = [Context()]

  /// The data key for the singleton `Context` in the current thread.
  static let key: ThreadLocalStorage.Key =
    ThreadLocalStorage.Key {
      #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
        Unmanaged<ContextManager>.fromOpaque($0).release()
      #else
        Unmanaged<ContextManager>.fromOpaque($0!).release()
      #endif
    }

  /// The thread-local singleton.
  static var local: ContextManager {
    if let address = ThreadLocalStorage.get(for: key) {
      return Unmanaged<ContextManager>.fromOpaque(address)
        .takeUnretainedValue()
    }

    let context = ContextManager()
    ThreadLocalStorage.set(
      value: Unmanaged.passRetained(context).toOpaque(),
      for: key)
    return context
  }

  /// Pushes the given context to the context stack.
  func push(_ context: Context) {
    contextStack.append(context)
  }

  /// Pops a context out of a stack.
  ///
  /// - Precondition: The context stack must contain more than `1` contexts.
  func popContext() {
    assert(
      contextStack.count > 1,
      "Internal error: Only 1 context is available. Popping is not allowed.")
    contextStack.removeLast()
  }

  /// The most recent context.
  var currentContext: Context {
    _read {
      assert(!contextStack.isEmpty, "Internal error: No contexts exist.")
      yield contextStack[contextStack.endIndex - 1]
    }
    _modify {
      assert(!contextStack.isEmpty, "Internal error: No contexts exist.")
      yield &contextStack[contextStack.endIndex - 1]
    }
  }
}
