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
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include "daisykit/common/types.h"
#include "hnswlib.h"

using idx_t = hnswlib::labeltype;
namespace daisykit {
namespace models {
class FaceManager {
 public:
  FaceManager(const std::string& save_path, int max_size, int topk, int dim,
              float threshold);
  FaceManager(const std::string& config_path);
  ~FaceManager();

  int GetNumDatas();
  bool Insert(const std::vector<float>& feature, int& inserted_id);
  bool InsertMultiple(const std::vector<std::vector<float>>& features,
                      std::vector<int>& inserted_ids);
  bool Search(const std::vector<float>& feature,
              std::vector<daisykit::types::FaceSearchResult>& result);

  bool InsertMultiple(
      const std::vector<std::vector<float>>& features,
      std::vector<std::vector<daisykit::types::FaceSearchResult>>& results);
  bool DeleteByIds(const std::vector<int>& ids);
  bool DeleteById(const int id);
  float threshold_;
  bool LoadData();

 private:
  bool SaveData();

 private:
  int max_size_;
  int dim_;
  int topk_;
  std::string save_path_;
  std::shared_ptr<hnswlib::L2Space> space_;
  hnswlib::HierarchicalNSW<float>* alg_hnsw_;
};
}  // namespace models
}  // namespace daisykit
#endif
