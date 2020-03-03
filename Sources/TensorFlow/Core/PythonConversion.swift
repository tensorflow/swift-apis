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

#if canImport(PythonKit)

  import PythonKit

  /// The `numpy` Python module.
  /// Note: Global variables are lazy, so the following declaration won't produce
  // a Python import error until it is first used.
  private let np = Python.import("numpy")

  private func debugLogNumpyError(_ message: String) {
    #if ENABLE_NUMPY_LOGGING
      debugLog("NumPy conversion error: " + message)
    #endif
  }

  extension ShapedArray: ConvertibleFromNumpyArray
  where Scalar: NumpyScalarCompatible {
    /// Creates a `ShapedArray` with the same shape and scalars as the specified
    /// `numpy.ndarray` instance.
    ///
    /// - Parameter numpyArray: The `numpy.ndarray` instance to convert.
    /// - Precondition: The `numpy` Python package must be installed.
    /// - Precondition: `numpyArray` must have a compatible scalar `dtype`.
    public init?(numpy numpyArray: PythonObject) {
      // Check if input is a `numpy.ndarray` instance.
      guard Python.isinstance(numpyArray, np.ndarray) == true else {
        debugLogNumpyError(
          """
          PythonObject input has type '\(Python.type(numpyArray))' and is not \
          an instance of 'numpy.ndarray'.
          """)
        return nil
      }
      // Check if the dtype of the `ndarray` is compatible with the `Scalar`
      // type.
      guard Scalar.numpyScalarTypes.contains(numpyArray.dtype) else {
        debugLogNumpyError(
          """
          'numpy.ndarray' dtype '\(numpyArray.dtype)' is incompatible with \
          Swift type '\(Scalar.self)'.
          """)
        return nil
      }

      let pyShape = numpyArray.__array_interface__["shape"]
      guard let shape = [Int](pyShape) else {
        debugLogNumpyError("cannot access shape of 'numpy.ndarray' instance.")
        return nil
      }

      // Make sure that the array is contiguous in memory. This does a copy if
      // the array is not already contiguous in memory.
      let contiguousNumpyArray = np.ascontiguousarray(numpyArray)

      guard
        let ptrVal =
          UInt(contiguousNumpyArray.__array_interface__["data"].tuple2.0)
      else {
        debugLogNumpyError("cannot access data of 'numpy.ndarray' instance.")
        return nil
      }
      // Note: `ptr` is not nil even if the `ndarray` is empty (i.e. has a shape
      // of `(0,)`).
      guard let ptr = UnsafePointer<Scalar>(bitPattern: ptrVal) else {
        fatalError("'numpy.ndarray' data pointer was nil")
      }
      // This code avoids calling `init<S: Sequence>(shape: [Int], scalars: S)`,
      // which inefficiently copies scalars one by one. Instead,
      // `init(shape: [Int], scalars: [Scalar])` is called, which efficiently
      // does a `memcpy` of the entire `scalars` array.
      // Unecessary copying is minimized.
      let dummyPointer = UnsafeMutablePointer<Scalar>.allocate(capacity: 1)
      let scalarCount = shape.reduce(1, *)
      var scalars: [Scalar] = Array(repeating: dummyPointer.move(), count: scalarCount)
      dummyPointer.deallocate()
      scalars.withUnsafeMutableBufferPointer { buffPtr in
        buffPtr.baseAddress!.assign(from: ptr, count: scalarCount)
      }
      self.init(shape: shape, scalars: scalars)
    }
  }

  extension Tensor: ConvertibleFromNumpyArray where Scalar: NumpyScalarCompatible {
    /// Creates a tensor with the same shape and scalars as the specified
    /// `numpy.ndarray` instance.
    ///
    /// - Parameter numpyArray: The `numpy.ndarray` instance to convert.
    /// - Precondition: The `numpy` Python package must be installed.
    /// - Returns: `numpyArray` converted to an `Array`. Returns `nil` if
    ///   `numpyArray` does not have a compatible scalar `dtype`.
    public init?(numpy numpyArray: PythonObject) {
      // Check if input is a `numpy.ndarray` instance.
      guard Python.isinstance(numpyArray, np.ndarray) == true else {
        debugLogNumpyError(
          """
          PythonObject input has type '\(Python.type(numpyArray))' and is not \
          an instance of 'numpy.ndarray'.
          """)
        return nil
      }
      // Check if the dtype of the `ndarray` is compatible with the `Scalar`
      // type.
      guard Scalar.numpyScalarTypes.contains(numpyArray.dtype) else {
        debugLogNumpyError(
          """
          'numpy.ndarray' dtype '\(numpyArray.dtype)' is incompatible with \
          Swift type '\(Scalar.self)'.
          """)
        return nil
      }

      let pyShape = numpyArray.__array_interface__["shape"]
      guard let dimensions = [Int](pyShape) else {
        debugLogNumpyError("cannot access shape of 'numpy.ndarray' instance.")
        return nil
      }
      let shape = TensorShape(dimensions)

      // Make sure that the array is contiguous in memory. This does a copy if
      // the array is not already contiguous in memory.
      let contiguousNumpyArray = np.ascontiguousarray(numpyArray)

      guard let ptrVal = UInt(contiguousNumpyArray.__array_interface__["data"].tuple2.0)
      else {
        debugLogNumpyError("cannot access data of 'numpy.ndarray' instance.")
        return nil
      }
      // Note: `ptr` is not nil even if the `ndarray` is empty (i.e. has a shape
      // of `(0,)`).
      guard let ptr = UnsafePointer<Scalar>(bitPattern: ptrVal) else {
        fatalError("'numpy.ndarray' data pointer was nil")
      }
      let buffPtr = UnsafeBufferPointer(start: ptr, count: Int(shape.contiguousSize))
      self.init(shape: shape, scalars: buffPtr)
    }
  }

  extension ShapedArray where Scalar: NumpyScalarCompatible {
    /// Creates a `numpy.ndarray` instance with the same shape and scalars as
    /// this `ShapedArray`.
    ///
    /// - Precondition: The `numpy` Python package must be installed.
    public func makeNumpyArray() -> PythonObject {
      return scalars.makeNumpyArray().reshape(shape)
    }
  }

  extension Tensor where Scalar: NumpyScalarCompatible {
    /// Creates a `numpy.ndarray` instance with the same shape and scalars as
    /// this tensor.
    ///
    /// - Precondition: The `numpy` Python package must be installed.
    public func makeNumpyArray() -> PythonObject { return array.makeNumpyArray() }
  }

  extension TensorShape: PythonConvertible {
    public var pythonObject: PythonObject {
      return dimensions.pythonObject
    }
  }

#endif  // canImport(PythonKit)
