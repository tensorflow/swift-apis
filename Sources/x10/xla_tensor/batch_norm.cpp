// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/tf2xla/xla_tensor/batch_norm.h"

#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"

namespace swift_xla {
namespace {

xla::XlaOp VarianceRecover(xla::XlaOp invstd, float eps_value) {
  const xla::Shape& invstd_shape = XlaHelpers::ShapeOfXlaOp(invstd);
  xla::XlaOp eps = XlaHelpers::ScalarValue(
      eps_value, invstd_shape.element_type(), invstd.builder());
  xla::XlaOp one_over_invstd =
      xla::One(invstd.builder(), invstd_shape.element_type()) / invstd;
  return one_over_invstd * one_over_invstd - eps;
}

}  // namespace

xla::XlaOp BatchNormVarianceInvert(xla::XlaOp variance, float eps_value) {
  const xla::Shape& variance_shape = XlaHelpers::ShapeOfXlaOp(variance);
  xla::XlaOp eps = XlaHelpers::ScalarValue(
      eps_value, variance_shape.element_type(), variance.builder());
  return xla::Rsqrt(variance + eps);
}

BatchNormOutput BuildBatchNormTraining(xla::XlaOp input, xla::XlaOp weight,
                                       xla::XlaOp bias, float eps_value) {
  xla::XlaOp outputs = xla::BatchNormTraining(input, weight, bias, eps_value,
                                              /*feature_index=*/1);
  xla::XlaOp output = xla::GetTupleElement(outputs, 0);
  xla::XlaOp batch_mean = xla::GetTupleElement(outputs, 1);
  xla::XlaOp batch_variance = xla::GetTupleElement(outputs, 2);
  return {output, batch_mean, batch_variance};
}

xla::XlaOp BuildBatchNormInference(xla::XlaOp input, xla::XlaOp weight,
                                   xla::XlaOp bias, xla::XlaOp mean,
                                   xla::XlaOp variance, float eps_value) {
  return xla::BatchNormInference(input, weight, bias, mean, variance, eps_value,
                                 /*feature_index=*/1);
}

BatchNormGrads BuildBatchNormBackward(xla::XlaOp grad, xla::XlaOp input,
                                      xla::XlaOp weight, xla::XlaOp save_mean,
                                      xla::XlaOp save_invstd, bool training,
                                      float eps_value) {
  xla::XlaOp grads = xla::BatchNormGrad(input, weight, save_mean,
                                        VarianceRecover(save_invstd, eps_value),
                                        grad, eps_value, /*feature_index=*/1);
  xla::XlaOp grad_input = xla::GetTupleElement(grads, 0);
  xla::XlaOp grad_weight = xla::GetTupleElement(grads, 1);
  xla::XlaOp grad_bias = xla::GetTupleElement(grads, 2);
  return {grad_input, grad_weight, grad_bias};
}

}  // namespace swift_xla
