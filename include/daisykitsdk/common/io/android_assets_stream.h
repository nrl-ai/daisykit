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

#ifndef DAISYKIT_IO_ANDROID_ASSETS_STREAM_H_
#define DAISYKIT_IO_ANDROID_ASSETS_STREAM_H_

#include <fstream>
#include <streambuf>
#include <string>
#include <vector>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

class AssetStreambuf : public std::streambuf {
 public:
  AssetStreambuf(AAssetManager* manager, const std::string& filename);
  virtual ~AssetStreambuf();

  std::streambuf::int_type underflow() override;
  std::streambuf::int_type overflow(std::streambuf::int_type value) override;

  int sync() override;

 private:
  AAssetManager* manager;
  AAsset* asset;
  std::vector<char> buffer;
};

class AssetIStream : public std::istream {
 public:
  AssetIStream(AAssetManager* manager, const std::string& file);
  AssetIStream(const std::string& file);

  virtual ~AssetIStream();

  static void SetAssetManager(AAssetManager* m);

 private:
  static AAssetManager* manager;
};

#endif