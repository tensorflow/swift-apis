#ifndef X10_XLA_TENSOR_TF_OPS_H_
#define X10_XLA_TENSOR_TF_OPS_H_

#include "swift_bindings/xla_tensor_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

OpaqueXLATensor* tf_AvgPool(OpaqueXLATensor* value, Int64ArrayRef ksize,
                            Int64ArrayRef strides, enum TFPadding padding,
                            enum TFDataFormat data_format);

OpaqueXLATensor* tf_AvgPoolGrad(Int64ArrayRef origInputShape,
                                OpaqueXLATensor* grad, Int64ArrayRef ksize,
                                Int64ArrayRef strides, enum TFPadding padding,
                                enum TFDataFormat data_format);

OpaqueXLATensor* tf_MaxPool(OpaqueXLATensor* input, Int64ArrayRef ksize,
                            Int64ArrayRef strides, enum TFPadding padding,
                            enum TFDataFormat data_format);

OpaqueXLATensor* tf_MaxPoolGrad(OpaqueXLATensor* input, OpaqueXLATensor* grad,
                                Int64ArrayRef ksize, Int64ArrayRef strides,
                                enum TFPadding padding);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // X10_XLA_TENSOR_TF_OPS_H_
