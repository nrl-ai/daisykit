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
#ifndef MODELS_FACE_MANAGER_H_
#define MODELS_FACE_MANAGER_H_
#include <assert.h>
#include <iostream>
#include <vector>
#include "daisykit/common/types.h"
#include "hnswlib.h"

using idx_t = hnswlib::labeltype;
namespace daisykit {
namespace models {
class FaceManager {
 public:
  FaceManager(std::string path_data, int max_size, int dim, int k,
              float threshold);
  ~FaceManager();
  bool InsertFeature(const std::vector<float> feature, const std::string name,
                     const int id);
  void DeleteName(std::string name);
  bool Search(std::vector<daisykit::types::FaceInfor>& result,
              std::vector<float> feature);

 private:
  void LoadLabel(std::vector<std::string>& labels, int& length,
                 std::string path);
  void SaveLabel(std::vector<std::string>& labels, std::string path);
  void InsertLabel(const std::string name, std::string path);
  void SaveData(std::string path);
  void LoadData(std::string path);
  void InsertData(std::string path, const daisykit::types::FeatureSet newf);
  int GetIndexName(std::string name, std::vector<std::string> labels);
  void ReLoadHNSW();
  void WriteFeatureSet(std::ofstream& stream,
                       const daisykit::types::FeatureSet& feature_set);
  daisykit::types::FeatureSet ReadFeatureSet(std::ifstream& stream);

  int max_size_;
  int dim_;
  int k_;
  float threshold_;
  std::vector<daisykit::types::FeatureSet> data_;
  std::string path_data_;
  int length_ = 0;
  std::shared_ptr<hnswlib::L2Space> space_;
  hnswlib::AlgorithmInterface<float>* alg_hnsw_;
};
}  // namespace models
}  // namespace daisykit
#endif
