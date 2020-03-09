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

xla::XlaOp BuildTriu(xla::XlaOp input, xla::int64 diagonal);

xla::XlaOp BuildTril(xla::XlaOp input, xla::int64 diagonal);

xla::XlaOp BuildDiagonal(xla::XlaOp input, xla::int64 offset, xla::int64 dim1,
                         xla::int64 dim2);

xla::XlaOp BuildDiagonalViewUpdate(xla::XlaOp target, xla::XlaOp input,
                                   xla::int64 offset, xla::int64 dim1,
                                   xla::int64 dim2);

xla::XlaOp BuildInverse(xla::XlaOp input);

}  // namespace swift_xla
