#include <daisykitsdk/flows/face_detector_flow.h>

using namespace daisykit::flows;

FaceDetectorFlow::FaceDetectorFlow(const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  face_detector_ = new models::FaceDetector(
      config["face_detection_model"]["model"],
      config["face_detection_model"]["weights"],
      config["face_detection_model"]["input_width"],
      config["face_detection_model"]["input_height"],
      config["face_detection_model"]["score_threshold"],
      config["face_detection_model"]["iou_threshold"]);
}

#ifdef __ANDROID__
FaceDetectorFlow::FaceDetectorFlow(AAssetManager* mgr,
                                   const std::string& config_str) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  face_detector_ = new models::FaceDetector(
      mgr,
      config["face_detection_model"]["model"],
      config["face_detection_model"]["weights"],
      config["face_detection_model"]["input_width"],
      config["face_detection_model"]["input_height"],
      config["face_detection_model"]["score_threshold"],
      config["face_detection_model"]["iou_threshold"]);
}
#endif

void FaceDetectorFlow::Process(cv::Mat& rgb) {
  // Detect faces
  std::vector<common::Face> faces = face_detector_->Detect(rgb);
  {
    const std::lock_guard<std::mutex> lock(faces_lock_);
    faces_ = faces;
  }
}

void FaceDetectorFlow::DrawResult(cv::Mat& rgb) {
  // Draw face bounding boxes
  {
    const std::lock_guard<std::mutex> lock(faces_lock_);
    for (auto face : faces_) {
      cv::rectangle(rgb, cv::Rect(face.x, face.y, face.w, face.h),
                    cv::Scalar(0, 255, 0), 2);
    }
  }
}