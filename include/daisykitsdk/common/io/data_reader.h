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

#ifndef DAISYKIT_IO_DATA_READER_H_
#define DAISYKIT_IO_DATA_READER_H_

#include <string>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

namespace daisykit {
namespace io {

/// Data Reader for DaisyKit core.
/// Should be implemented for cross platform content reading. Add other
/// processors such as encryption/decryption module in the future.
class DataReader {
 public:
  /// Constructors for data reader to read from file
  DataReader();
  // Copy constructor.
  DataReader(const DataReader& a);

#ifdef __ANDROID__
  /// Construct data reader to read from Android asset.
  DataReader(AAssetManager* asset_manager);
#endif

  /// General reading function.
  /// This method reads file from `path`, and put read data into `data` buffer.
  /// Return 0 on success, otherwise return a negative number. Reading source
  /// from file or Android asset depending on the constructor.
  int Read(const std::string& path, char** data) const;

  /// Read the whole file into memory.
  /// This method reads file from `path`, and put read data into `data` buffer.
  /// Return 0 on success, otherwise return a negative number.
  int ReadFile(const std::string& path, char** data) const;

#ifdef __ANDROID__
  /// Read the whole file into memory from an Android asset.
  /// This API requires AAssetManager to get file from Android assets.
  /// It reads file from `path`, and put read data into `data` buffer.
  /// Return 0 on success, otherwise return a negative number.
  int DataReader::ReadAndroidAsset(const std::string& path, char** data) const;
#endif

 private:
#ifdef __ANDROID__
  AAssetManager* asset_manager_;
#endif
  bool read_from_android_asset_;
};

}  // namespace io
}  // namespace daisykit

#endif