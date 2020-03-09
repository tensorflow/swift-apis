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

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"

namespace swift_xla {

class ShapeBuilder {
 public:
  explicit ShapeBuilder(xla::PrimitiveType type) : type_(type) {}

  ShapeBuilder& Add(const xla::Shape& shape, xla::int64 dim);

  ShapeBuilder& Add(const xla::Shape& shape,
                    absl::Span<const xla::int64> dimensions);

  ShapeBuilder& Add(xla::int64 size);

  xla::Shape Build() const;

 private:
  struct ShapeDim {
    const xla::Shape* shape = nullptr;
    xla::int64 dim_or_size = -1;
  };

  xla::PrimitiveType type_;
  std::vector<ShapeDim> dims_;
};

}  // namespace swift_xla
