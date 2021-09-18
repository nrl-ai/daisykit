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

#ifndef DAISYKIT_GRAPHS_CORE_PACKET_H_
#define DAISYKIT_GRAPHS_CORE_PACKET_H_

#include "daisykitsdk/common/utils/timer.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <memory>
#include <mutex>

namespace daisykit {
namespace graphs {

/// Packet is a data holder, used in DaisyKit graph as the wrapper message for
/// data transmission between nodes.
class Packet {
 public:
  /// Create an empty packet
  Packet();

  /// Create a new packet with data and timestamp
  Packet(std::shared_ptr<void> data, utils::TimePoint timestamp);

  /// Get data from packet.
  /// Return nullptr if no data in the packet.
  template <typename T>
  std::shared_ptr<T> GetData() {
    /// If no data
    if (!data_available_) {
      return nullptr;
    }
    const std::lock_guard<std::mutex> lock(data_mutex_);
    return std::static_pointer_cast<T>(data_);
  }

  /// Get data from packet and its timestamp.
  /// Return 0 if successfully, otherwise return -1.
  template <typename T>
  bool GetData(std::shared_ptr<T>& data, utils::TimePoint& timestamp) {
    /// If no data
    if (!data_available_) {
      return -1;
    }
    const std::lock_guard<std::mutex> lock(data_mutex_);
    data = std::static_pointer_cast<T>(data_);
    timestamp = timestamp_;
    return 0;
  }

  /// Create a packet from pointer.
  template <typename T>
  static std::shared_ptr<Packet> Adopt(T* ptr) {
    std::shared_ptr<T> data(ptr);
    utils::TimePoint timestamp = utils::Timer::GetCurrentTime();
    std::shared_ptr<Packet> packet = std::make_shared<Packet>(
        std::static_pointer_cast<void>(data), timestamp);
    return packet;
  }

  /// Make a packet from a shared pointer to the data.
  template <typename T>
  static std::shared_ptr<Packet> MakePacket(std::shared_ptr<T> data) {
    utils::TimePoint timestamp = utils::Timer::GetCurrentTime();
    std::shared_ptr<Packet> packet = std::make_shared<Packet>(
        std::static_pointer_cast<void>(data), timestamp);
    return packet;
  }

  /// Create a packet containing an object of type T initialized with the
  /// provided arguments. Similar to std::make_shared.
  template <typename T,
            typename std::enable_if<!std::is_array<T>::value>::type* = nullptr,
            typename... Args>
  static std::shared_ptr<Packet> MakePacket(
      Args&&... args) {  /// NOLINT(build/c++11)
    return Adopt(new T(std::forward<Args>(args)...));
  }

 private:
  std::mutex data_mutex_;
  std::shared_ptr<void> data_;
  utils::TimePoint timestamp_;
  std::atomic<bool> data_available_;
};

typedef std::shared_ptr<Packet> PacketPtr;

}  // namespace graphs
}  // namespace daisykit

#endif