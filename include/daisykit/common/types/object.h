// Copyright 2021 The DaisyKit Authors.
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

#ifndef DAISYKIT_COMMON_TYPES_OBJECT_H_
#define DAISYKIT_COMMON_TYPES_OBJECT_H_

#include "daisykit/common/types/box.h"

namespace daisykit {
namespace types {

/// Object data type.
/// Used in object detection models
struct Object : Box {
  Object() {}
  Object(float x, float y, float w, float h, int class_id, float confidence)
      : Box(x, y, w, h), class_id(class_id), confidence(confidence) {}
  int class_id;      /// Object class ID
  float confidence;  /// Detection score
};

}  // namespace types
}  // namespace daisykit

#endif
