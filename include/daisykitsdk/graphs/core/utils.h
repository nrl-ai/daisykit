#ifndef DAISYKIT_GRAPHS_CORE_UTILS_H_
#define DAISYKIT_GRAPHS_CORE_UTILS_H_

#include <daisykitsdk/graphs/core/node.h>
#include <daisykitsdk/utils/timer.h>

#include <chrono>
#include <memory>

namespace daisykit {
namespace graphs {

void Packet2CvMat(PacketPtr packet, cv::Mat& output) {
  if (packet == nullptr) {
    output = cv::Mat();
    return;
  }
  std::shared_ptr<void> data;
  std::time_t timestamp;
  packet->GetData(data, timestamp);
  output = *std::static_pointer_cast<cv::Mat>(data);
}

void CvMat2Packet(const cv::Mat& input, PacketPtr& packet, bool copy = false) {
  utils::TimePoint timestamp = utils::Timer::GetCurrentTime();
  std::shared_ptr<cv::Mat> output_mat = std::make_shared<cv::Mat>();
  *output_mat = copy ? input.clone() : input;
  packet = std::make_shared<Packet>(std::static_pointer_cast<void>(output_mat),
                                    timestamp);
}

}  // namespace graphs
}  // namespace daisykit

#endif