#ifndef DAISYKIT_FLOWS_PUSHUP_COUNTER_FLOW_H_
#define DAISYKIT_FLOWS_PUSHUP_COUNTER_FLOW_H_

#include <atomic>
#include <iostream>
#include <string>
#include <vector>

#include <daisykitsdk/common/types.h>
#include <daisykitsdk/examples/fitness/pushup_analyzer.h>
#include <daisykitsdk/models/action_classifier.h>
#include <daisykitsdk/models/body_detector.h>
#include <daisykitsdk/models/pose_detector.h>
#include <daisykitsdk/utils/img_proc/img_utils.h>
#include <daisykitsdk/utils/visualizer/viz_utils.h>
#include <daisykitsdk/thirdparties/json.hpp>

namespace daisykit {
namespace flows {
class PushupCounterFlow {
 public:
  PushupCounterFlow(const std::string& config_str);
#ifdef __ANDROID__
  PushupCounterFlow(AAssetManager* mgr, const std::string& config_str);
#endif
  void Process(cv::Mat& rgb);
  void DrawResult(cv::Mat& rgb);

 private:
  std::mutex input_lock_;
  std::mutex output_lock_;

  std::vector<common::Object> bodies_;
  std::mutex bodies_lock_;

  std::vector<std::vector<common::Keypoint>> keypoints_;
  std::mutex keypoints_lock_;

  std::atomic<int> num_pushups_;
  std::atomic<float> is_pushup_score_;

  models::BodyDetector* body_detector_;
  models::PoseDetector* pose_detector_;
  models::ActionClassifier* action_classifier_;
  examples::PushupAnalyzer* pushup_analyzer_;
};

}  // namespace flows
}  // namespace daisykit

#endif