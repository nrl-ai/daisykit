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

#ifndef DAISYKIT_COMMON_TYPES_FEATURE_SET_H_
#define DAISYKIT_COMMON_TYPES_FEATURE_SET_H_
#include <string>
#include <vector>
namespace daisykit {
namespace types {

/// FeatureSet is a struct that contains a feature vector and its id and name
/// Used in face recognition
struct FeatureSet {
  int id;                      /// ID of the feature
  std::string name;            /// Name of the feature
  std::vector<float> feature;  /// Feature vector
};

}  // namespace types
}  // namespace daisykit

#endif
