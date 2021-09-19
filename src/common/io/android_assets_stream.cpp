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

#include "daisykitsdk/common/io/android_assets_stream.h"

AssetStreambuf::AssetStreambuf(AAssetManager* manager,
                               const std::string& filename)
    : manager(manager) {
  asset = AAssetManager_open(manager, filename.c_str(), AASSET_MODE_STREAMING);
  buffer.resize(1024);

  setg(0, 0, 0);
  setp(&buffer.front(), &buffer.front() + buffer.size());
}

AssetStreambuf::~AssetStreambuf() {
  sync();
  AAsset_close(asset);
}

std::streambuf::int_type AssetStreambuf::underflow() {
  auto bufferPtr = &buffer.front();
  auto counter = AAsset_read(asset, bufferPtr, buffer.size());

  if (counter == 0) return traits_type::eof();
  if (counter < 0)  // error, what to do now?
    return traits_type::eof();

  setg(bufferPtr, bufferPtr, bufferPtr + counter);

  return traits_type::to_int_type(*gptr());
}

std::streambuf::int_type AssetStreambuf::overflow(
    std::streambuf::int_type value) {
  return traits_type::eof();
};

int AssetStreambuf::sync() {
  std::streambuf::int_type result = overflow(traits_type::eof());

  return traits_type::eq_int_type(result, traits_type::eof()) ? -1 : 0;
}

AssetIStream::AssetIStream(AAssetManager* manager, const std::string& file)
    : std::istream(new AssetStreambuf(manager, file)) {}

AssetIStream::AssetIStream(const std::string& file)
    : std::istream(new AssetStreambuf(manager, file)) {}

AssetIStream::~AssetIStream() { delete rdbuf(); }

void AssetIStream::setAssetManager(AAssetManager* m) { manager = m; }

AAssetManager* AssetIStream::manager;
