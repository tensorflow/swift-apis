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

#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace swift_xla {

struct BatchNormOutput {
  xla::XlaOp output;
  xla::XlaOp batch_mean;
  xla::XlaOp batch_variance;
};

struct BatchNormGrads {
  xla::XlaOp grad_input;
  xla::XlaOp grad_weight;
  xla::XlaOp grad_bias;
};

xla::XlaOp BatchNormVarianceInvert(xla::XlaOp variance, float eps_value);

BatchNormOutput BuildBatchNormTraining(xla::XlaOp input, xla::XlaOp weight,
                                       xla::XlaOp bias, float eps_value);

xla::XlaOp BuildBatchNormInference(xla::XlaOp input, xla::XlaOp weight,
                                   xla::XlaOp bias, xla::XlaOp mean,
                                   xla::XlaOp variance, float eps_value);

BatchNormGrads BuildBatchNormBackward(xla::XlaOp grad, xla::XlaOp input,
                                      xla::XlaOp weight, xla::XlaOp save_mean,
                                      xla::XlaOp save_invstd, bool training,
                                      float eps_value);

}  // namespace swift_xla
