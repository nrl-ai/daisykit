#include <daisykitsdk/flows/pushup_counter_flow.h>

using namespace daisykit::flows;

PushupCounterFlow::PushupCounterFlow(const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  body_detector_ =
      new models::BodyDetector(config["person_detection_model"]["model"],
                               config["person_detection_model"]["weights"]);
  pose_detector_ =
      new models::PoseDetector(config["human_pose_model"]["model"],
                               config["human_pose_model"]["weights"]);
  action_classifier_ = new models::ActionClassifier(
      config["action_recognition_model"]["model"],
      config["action_recognition_model"]["weights"], true);
  pushup_analyzer_ = new examples::PushupAnalyzer();
}

#ifdef __ANDROID__
PushupCounterFlow::PushupCounterFlow(AAssetManager* mgr,
                                     const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  body_detector_ =
      new models::BodyDetector(mgr, config["person_detection_model"]["model"],
                               config["person_detection_model"]["weights"]);
  pose_detector_ =
      new models::PoseDetector(mgr, config["human_pose_model"]["model"],
                               config["human_pose_model"]["weights"]);
  action_classifier_ = new models::ActionClassifier(
      mgr, config["action_recognition_model"]["model"],
      config["action_recognition_model"]["weights"], true);
  pushup_analyzer_ = new examples::PushupAnalyzer();
}
#endif

void PushupCounterFlow::Process(cv::Mat& rgb) {
  // Detect human pose
  std::vector<common::Object> bodies = body_detector_->Detect(rgb);
  {
    const std::lock_guard<std::mutex> lock(bodies_lock_);
    bodies_ = bodies;
  }

  // Detect keypoints
  std::vector<std::vector<common::Keypoint>> keypoints =
      pose_detector_->DetectMulti(rgb, bodies);
  {
    const std::lock_guard<std::mutex> lock(keypoints_lock_);
    keypoints_ = keypoints;
  }

  // Recognize action and count pushups
  float score;
  common::Action action = action_classifier_->Classify(rgb, score);
  is_pushup_score_ = score;
  num_pushups_ =
      pushup_analyzer_->CountPushups(rgb, action == common::Action::kPushup);
}

void PushupCounterFlow::DrawResult(cv::Mat& rgb) {
  // Draw body bounding boxes
  {
    const std::lock_guard<std::mutex> lock(bodies_lock_);
    for (auto body : bodies_) {
      cv::rectangle(rgb, cv::Rect(body.x, body.y, body.w, body.h),
                    cv::Scalar(0, 255, 0), 2);
    }
  }

  // Draw body keypoints
  {
    const std::lock_guard<std::mutex> lock(keypoints_lock_);
    for (auto kp_single : keypoints_) {
      pose_detector_->DrawKeypoints(rgb, kp_single);
    }
  }

  // Draw pushups counting
  if (is_pushup_score_ > 0.5) {
    utils::visualizer::VizUtils::DrawLabel(
        rgb, "Is pushing: " + std::to_string(is_pushup_score_),
        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, 10,
        cv::Scalar(0, 0, 0), cv::Scalar(0, 255, 0));
  } else {
    utils::visualizer::VizUtils::DrawLabel(
        rgb, "Is pushing: " + std::to_string(is_pushup_score_),
        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, 10,
        cv::Scalar(0, 0, 0), cv::Scalar(220, 220, 220));
  }
  utils::visualizer::VizUtils::DrawLabel(
      rgb, std::string("PushUps: ") + std::to_string(num_pushups_),
      cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 1.2, 2, 10,
      cv::Scalar(255, 255, 255), cv::Scalar(255, 0, 0));
}