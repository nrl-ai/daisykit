#ifndef DAISYKIT_GRAPHS_CORE_PACKET_H_
#define DAISYKIT_GRAPHS_CORE_PACKET_H_

#include <daisykitsdk/utils/timer.h>

#include <chrono>
#include <ctime>
#include <memory>
#include <mutex>

namespace daisykit {
namespace graphs {

class Packet {
 public:
  Packet(){};
  Packet(std::shared_ptr<void> data, utils::TimePoint timestamp) {
    // We dont use SetData() here because using mutex lock
    // in the constructor causes crashing
    data_ = data;
    timestamp_ = timestamp;
  }

  void SetData(std::shared_ptr<void> data, utils::TimePoint timestamp) {
    const std::lock_guard<std::mutex> lock(data_mutex_);
    data_ = data;
    timestamp_ = timestamp;
  }

  void GetData(std::shared_ptr<void>& data, utils::TimePoint& timestamp) {
    const std::lock_guard<std::mutex> lock(data_mutex_);
    data = data_;
    timestamp = timestamp_;
  }

 private:
  std::mutex data_mutex_;
  std::shared_ptr<void> data_;
  utils::TimePoint timestamp_;
};

typedef std::shared_ptr<Packet> PacketPtr;

}  // namespace graphs
}  // namespace daisykit

#endif