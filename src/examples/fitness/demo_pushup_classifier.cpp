#include <daisykitsdk/models/action_classifier.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace daisykit::common;
using namespace daisykit::models;


int main(int, char **) {
  ActionClassifier *model =
      new ActionClassifier("models/action_classification/is_pushup.param",
                           "models/action_classification/is_pushup.bin");

  Mat frame;
  VideoCapture cap(0);

  while (1) {
    cap >> frame;
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    float confidence;
    auto action = model->Classify(rgb, confidence);

    cv::Scalar color(0, 255, 0);
    if (confidence < 0.9) {
      color = cv::Scalar(0, 0, 255);
    }
    cv::putText(frame, std::to_string(confidence), cv::Point(100, 100), cv::FONT_HERSHEY_PLAIN, 1.0, color, 2);

    imshow("Image", frame);
    waitKey(1);
  }

  return 0;
}