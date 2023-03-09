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
FaceManager::FaceManager(std::string path_data, int max_size, int dim, int k,
                         float threshold) {
  path_data_ = path_data;
  max_size_ = max_size;
  dim_ = dim;
  k_ = k;
  threshold_ = threshold;
  space_ = std::make_shared<hnswlib::L2Space>(dim_);
  alg_hnsw_ = new hnswlib::HierarchicalNSW<float>(&*space_, max_size_);
  LoadData(path_data);
}

FaceManager::~FaceManager() = default;

bool CompareFaceinfor(daisykit::types::FaceInfor f1,
                      daisykit::types::FaceInfor f2) {
  return (f1.distance < f2.distance);
}

// Load and Save new data
void FaceManager::SaveData(std::string path) {
  std::ofstream output_file(path, std::ios::binary);
  for (int i = 0; i < length_; ++i) {
    WriteFeatureSet(output_file, data_[i]);
  }
  output_file.close();
}

void FaceManager::LoadData(std::string path) {
  data_.clear();
  length_ = 0;
  std::ifstream input_file(path, std::ios::binary);
  while (input_file.peek() != EOF) {
    daisykit::types::FeatureSet f = ReadFeatureSet(input_file);
    data_.emplace_back(f);
    length_ += 1;
  }
  for (size_t i = 0; i < length_; ++i) {
    alg_hnsw_->addPoint(data_[i].feature.data(), i);
  }
  std::cout << "Load " << length_ << " face\n";
}

void FaceManager::InsertData(std::string path,
                             const daisykit::types::FeatureSet newf) {
  std::ofstream output_file(path, std::ios::app | std::ios::binary);
  WriteFeatureSet(output_file, newf);
  output_file.close();
}

bool FaceManager::InsertFeature(const std::vector<float> feature,
                                const std::string name, const int id) {
  daisykit::types::FeatureSet newf;
  newf.feature = feature;
  newf.id = id;
  newf.name = name;

  alg_hnsw_->addPoint(feature.data(), length_);
  data_.emplace_back(newf);
  InsertData(path_data_, newf);

  length_ += 1;
  return true;
}
void FaceManager::ReLoadHNSW() {
  delete alg_hnsw_;
  alg_hnsw_ = new hnswlib::HierarchicalNSW<float>(&*space_, max_size_);
  for (size_t i = 0; i < length_; ++i) {
    alg_hnsw_->addPoint(data_[i].feature.data(), i);
  }
}

void FaceManager::DeleteName(std::string name) {
  for (int i = data_.size() - 1; i > 0; i--) {
    if (data_[i].name == name) {
      data_.erase(data_.begin() + i, data_.begin() + i + 1);
      length_ -= 1;
    }
  }
  SaveData(path_data_);
  ReLoadHNSW();
}

bool FaceManager::Search(std::vector<daisykit::types::FaceInfor>& result,
                         std::vector<float> feature) {
  int index;
  result.clear();
  if (length_ == 0)
    return false;
  else {
    const void* p = feature.data();
    auto gd = alg_hnsw_->searchKnn(p, k_);
    while (!gd.empty()) {
      if (gd.top().first <= threshold_) {
        daisykit::types::FaceInfor faceif;
        index = gd.top().second;
        faceif.distance = gd.top().first;
        faceif.name = data_[index].name;
        faceif.id = data_[index].id;
        result.emplace_back(faceif);
      }
      gd.pop();
    }
    std::sort(result.begin(), result.end(), CompareFaceinfor);
    return true;
  }
}

void FaceManager::WriteFeatureSet(
    std::ofstream& stream, const daisykit::types::FeatureSet& feature_set) {
  char name[512];
  float feature[dim_];
  int id;
  strncpy(name, feature_set.name.c_str(), feature_set.name.size());
  id = feature_set.id;
  std::copy(feature_set.feature.begin(), feature_set.feature.end(), feature);
  name[feature_set.name.size()] = '\0';
  stream.write((char*)&name, sizeof(name));
  stream.write((char*)&id, sizeof(id));
  stream.write((char*)&feature, sizeof(feature));
}

daisykit::types::FeatureSet FaceManager::ReadFeatureSet(std::ifstream& stream) {
  char name[512];
  float feature[dim_];
  int id;
  stream.read((char*)&name, sizeof(name));
  stream.read((char*)&id, sizeof(id));
  stream.read((char*)&feature, sizeof(feature));
  daisykit::types::FeatureSet f;
  f.name = name;
  f.id = id;
  f.feature = std::vector<float>(
      feature, feature + sizeof(feature) / sizeof(feature[0]));
  return f;
}
}  // namespace models
}  // namespace daisykit
