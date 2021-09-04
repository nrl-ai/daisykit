#include <daisykitsdk/models/pose_detector_blazepose.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int, char **) {
  PoseDetectorBlazepose *model =
      new PoseDetectorBlazepose("../data/models/pose_landmark_lite.param",
                           "../data/models/pose_landmark_lite.bin");

  Mat frame;
  VideoCapture cap(0);

  while (1) {
    cap >> frame;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
  
    // Detect keypoints
    std::vector<Keypoint> keypoints = model->detect(rgb, 0, 0);

    imshow("Image", frame);
    waitKey(1);
  }

  return 0;
}