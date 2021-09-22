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

#include "daisykitsdk/common/io/data_reader.h"

#include <iostream>

namespace daisykit {
namespace io {

DataReader::DataReader() { read_from_android_asset_ = false; }

#ifdef __ANDROID__
DataReader::DataReader(AAssetManager* asset_manager) {
  asset_manager_ = asset_manager;
  read_from_android_asset_ = true;
}
#endif

DataReader::DataReader(const DataReader& a) {
#ifdef __ANDROID__
  asset_manager_ = a.asset_manager_;
  read_from_android_asset_ = a.read_from_android_asset_;
#endif
}

int DataReader::Read(const std::string& path, char** data) const {
  // Read from Android asset
  if (read_from_android_asset_) {
#ifdef __ANDROID__
    return ReadAndroidAsset(path, data);
#endif
    return -1;
  }

  // Read from file
  return ReadFile(path, data);
}

int DataReader::ReadFile(const std::string& path, char** data) const {
  FILE* p_file;
  long l_size;
  char* buffer;
  size_t result;

  // Open file
  p_file = fopen(path.c_str(), "rb");
  if (p_file == nullptr) {
    std::cerr << "Error opening file: " << path << std::endl;
    return -1;
  }

  // Obtain file size
  fseek(p_file, 0, SEEK_END);
  l_size = ftell(p_file);
  rewind(p_file);

  // Allocate memory to contain the whole file
  buffer = (char*)malloc(sizeof(char) * l_size);
  if (buffer == nullptr) {
    std::cout << "Memory allocation failed on reading file: " << path
              << std::endl;
    return -2;
  }

  // Copy the file into the buffer
  result = fread(buffer, 1, l_size, p_file);
  if (result != l_size) {
    std::cout << "Error on reading file: " << path << std::endl;
    return -3;
  }

  // The whole file is now loaded in the memory buffer.
  // Close the file pointer
  fclose(p_file);
  *data = buffer;
  return 0;
}

#ifdef __ANDROID__
int DataReader::ReadAndroidAsset(const std::string& path, char** data) const {
  if (asset_manager_ == nullptr) {
    std::cout << "This reader was not initialized with asset manager, thus "
                 "could not read file from assets."
              << std::endl;
    return -1;
  }

  // Prepare buffer
  AAsset* asset_file =
      AAssetManager_open(asset_manager_, path.c_str(), AASSET_MODE_BUFFER);
  size_t file_length = AAsset_getLength(asset_file);
  char* buffer = (char*)malloc(file_length);
  if (buffer == nullptr) {
    std::cout << "Memory allocation failed on reading file: " << path
              << std::endl;
    return -2;
  }

  // Read file data
  AAsset_read(asset_file, data, file_length);
  AAsset_close(asset_file);

  // Final check data
  if (buffer == nullptr) {
    std::cout << "Error on asset file: " << path << std::endl;
    return -3;
  }

  *data = buffer;
  return 0;
}
#endif

}  // namespace io
}  // namespace daisykit
