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
#include "daisykit/models/face_recognition/face_manager.h"

namespace daisykit {
namespace models {
FaceManager::FaceManager(const std::string& save_path, int max_size, int topk,
                         int dim, float threshold) {
  save_path_ = save_path;
  max_size_ = max_size;
  topk_ = topk;
  dim_ = dim;
  threshold_ = threshold;
  space_ = std::make_shared<hnswlib::L2Space>(dim_);
  alg_hnsw_ = new hnswlib::HierarchicalNSW<float>(&*space_, max_size_);
}
FaceManager::FaceManager(const std::string& config_path) {
  // TODO
}

FaceManager::~FaceManager() = default;

bool CompareFaceinfor(const daisykit::types::FaceSearchResult& f1,
                      const daisykit::types::FaceSearchResult& f2) {
  return (f1.min_distance < f2.min_distance);
}

bool CheckExistsFile(const std::string& name) {
  std::ifstream f(name.c_str());
  return f.good();
}

// Load and Save new data
bool FaceManager::SaveData() {
  alg_hnsw_->saveIndex(save_path_);
  return true;
}

bool FaceManager::LoadData() {
  if (!CheckExistsFile(save_path_)) return false;
  alg_hnsw_->loadIndex(save_path_, &*space_, max_size_);
  return true;
}

int FaceManager::GetNumDatas() { return alg_hnsw_->cur_element_count; }

bool FaceManager::Insert(const std::vector<float>& feature, int& inserted_id) {
  int num_data = GetNumDatas();
  alg_hnsw_->addPoint(feature.data(), num_data);
  inserted_id = num_data;
  SaveData();
  return true;
}

bool FaceManager::InsertMultiple(
    const std::vector<std::vector<float>>& features,
    std::vector<int>& inserted_ids) {
  int num_data = GetNumDatas();
  inserted_ids.clear();
  inserted_ids.resize(features.size());
  for (int i = 0; i < features.size(); i++) {
    alg_hnsw_->addPoint(features[i].data(), num_data + i);
    inserted_ids[i] = num_data + i;
  }
  SaveData();
  return true;
}

bool FaceManager::DeleteById(const int id) {
  alg_hnsw_->markDelete(id);
  return true;
}

bool FaceManager::DeleteByIds(const std::vector<int>& ids) {
  for (int i = 0; i < ids.size(); i++) {
    DeleteById(ids[i]);
  }
  return true;
}

bool FaceManager::Search(
    const std::vector<float>& feature,
    std::vector<daisykit::types::FaceSearchResult>& result) {
  int index;
  int num_data = GetNumDatas();
  result.clear();
  if (num_data == 0)
    return false;
  else {
    const void* p = feature.data();
    int topk;
    if (topk_ >= num_data)
      topk = num_data;
    else
      topk = topk_;
    auto gd = alg_hnsw_->searchKnn(p, topk);
    while (!gd.empty()) {
      if (gd.top().first <= threshold_) {
        daisykit::types::FaceSearchResult faceif;
        index = gd.top().second;
        faceif.min_distance = gd.top().first;
        faceif.id = index;
        result.emplace_back(faceif);
      }
      gd.pop();
    }
    if (topk_ > 1) std::sort(result.begin(), result.end(), CompareFaceinfor);
  }
  return true;
}

bool FaceManager::InsertMultiple(
    const std::vector<std::vector<float>>& features,
    std::vector<std::vector<daisykit::types::FaceSearchResult>>& results) {
  int num_data = GetNumDatas();
  results.clear();
  if (num_data == 0)
    return false;
  else {
    for (int i = 0; i < features.size(); i++) {
      std::vector<daisykit::types::FaceSearchResult> result;
      Search(features[i], result);
      results.emplace_back(result);
    }
    return true;
  }
}

}  // namespace models
}  // namespace daisykit
