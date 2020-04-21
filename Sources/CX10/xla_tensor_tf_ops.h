/*
 * Copyright 2020 TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef X10_XLA_TENSOR_TF_OPS_H_
#define X10_XLA_TENSOR_TF_OPS_H_

#include "xla_tensor_wrapper.h"

#if !defined(XLA_API)
#define XLA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

XLA_API OpaqueXLATensor* tf_AvgPool(OpaqueXLATensor* value, Int64ArrayRef ksize,
                                    Int64ArrayRef strides,
                                    enum TFPadding padding,
                                    enum TFDataFormat data_format);

XLA_API OpaqueXLATensor* tf_AvgPoolGrad(Int64ArrayRef origInputShape,
                                        OpaqueXLATensor* grad,
                                        Int64ArrayRef ksize,
                                        Int64ArrayRef strides,
                                        enum TFPadding padding,
                                        enum TFDataFormat data_format);

XLA_API OpaqueXLATensor* tf_MaxPool(OpaqueXLATensor* input, Int64ArrayRef ksize,
                                    Int64ArrayRef strides,
                                    enum TFPadding padding,
                                    enum TFDataFormat data_format);

XLA_API OpaqueXLATensor* tf_MaxPoolGrad(OpaqueXLATensor* input,
                                        OpaqueXLATensor* grad,
                                        Int64ArrayRef ksize,
                                        Int64ArrayRef strides,
                                        enum TFPadding padding);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // X10_XLA_TENSOR_TF_OPS_H_
