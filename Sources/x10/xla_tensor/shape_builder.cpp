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

#include "tensorflow/compiler/tf2xla/xla_tensor/shape_builder.h"

#include "tensorflow/compiler/xla/shape_util.h"

namespace swift_xla {

ShapeBuilder& ShapeBuilder::Add(const xla::Shape& shape, xla::int64 dim) {
  dims_.push_back({&shape, dim});
  return *this;
}

ShapeBuilder& ShapeBuilder::Add(const xla::Shape& shape,
                                absl::Span<const xla::int64> dimensions) {
  dims_.reserve(dimensions.size());
  for (auto dim : dimensions) {
    dims_.push_back({&shape, dim});
  }
  return *this;
}

ShapeBuilder& ShapeBuilder::Add(xla::int64 size) {
  dims_.push_back({nullptr, size});
  return *this;
}

xla::Shape ShapeBuilder::Build() const {
  std::vector<xla::int64> dimensions;
  dimensions.reserve(dims_.size());
  for (auto& sdim : dims_) {
    if (sdim.shape != nullptr) {
      dimensions.push_back(sdim.shape->dimensions(sdim.dim_or_size));
    } else {
      dimensions.push_back(sdim.dim_or_size);
    }
  }
  xla::Shape shape = xla::ShapeUtil::MakeShape(type_, dimensions);
  for (xla::int64 i = 0; i < shape.rank(); ++i) {
    const ShapeDim& sdim = dims_[i];
    if (sdim.shape != nullptr) {
      shape.set_dynamic_dimension(
          i, sdim.shape->is_dynamic_dimension(sdim.dim_or_size));
    }
  }
  return shape;
}

}  // namespace swift_xla
