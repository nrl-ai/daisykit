#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <ai_models/pose_detector.h>
#include <ai_models/action_classifier.h>
#include <ai_models/body_detector.h>
#include <analyzers/pushup_analyzer.h>

using namespace cv;
using namespace std;

static BodyDetector* body_detector = 0;
static PoseDetector* pose_detector = 0;
static ActionClassifier* action_classifier = 0;
static PushupAnalyzer* pushup_analyzer = 0;

int main(int, char **) {
  body_detector = new BodyDetector("../data/models/person_detector.param",
                                    "../data/models/person_detector.bin");
  pose_detector = new PoseDetector("../data/models/Ultralight-Nano-SimplePose.param",
                                    "../data/models/Ultralight-Nano-SimplePose.bin");
  action_classifier = new ActionClassifier("../data/models/action_classifier.param",
                                    "../data/models/action_classifier.bin", true);
  pushup_analyzer = new PushupAnalyzer();

  Mat frame;
  VideoCapture cap(0);

  while (1) {
    cap >> frame;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
  
    // Detect human pose
    std::vector<Object> bodies = body_detector->detect(rgb);

    // Detect keypoints
    std::vector<std::vector<Keypoint>> keypoints = pose_detector->detect_multi(rgb, bodies);

    // Recognize action and count pushups
    float is_pushup_score;
    Action action = action_classifier->classify(rgb, is_pushup_score);
    bool is_pushup = action==Action::kPushup;
    int n_pushups = pushup_analyzer->count_pushups(rgb, is_pushup);

    // Draw result
    for (auto body : bodies) {
      cv::rectangle(frame, cv::Rect(body.x, body.y, body.w, body.h),
                    cv::Scalar(0, 255, 0), 2);
    }
    for (auto kp_single : keypoints) {
      pose_detector->draw_pose(frame, kp_single);
    }
    if (action == Action::kPushup) {
      cv::putText(frame, "PUSHING UP - " + std::to_string(is_pushup_score) + " - " + std::to_string(n_pushups), cv::Point(20, 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    } else {
      cv::putText(frame, "NOT PUSHING UP - " + std::to_string(is_pushup_score) + " - " + std::to_string(n_pushups), cv::Point(20, 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
    }

    imshow("Image", frame);
    waitKey(1);
  }

  return 0;
}