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

#ifndef DAISYKIT_COMMON_TYPES_FACE_SEARCH_RESULT_H_
#define DAISYKIT_COMMON_TYPES_FACE_SEARCH_RESULT_H_

#include <string>
#include <vector>

namespace daisykit {
namespace types {

/// Face comparison result struct
/// Contains the ID of the face and the min distance of the face to all faces in
/// the database
class FaceSearchResult {
 public:
  int id;              /// ID of the face
  float min_distance;  /// Min distance of the face to all faces in the database
};

}  // namespace types
}  // namespace daisykit

#endif
