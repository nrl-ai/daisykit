#ifndef DAISYKIT_MODELS_FACE_RECOGNITION_ANCHOR_GENERATOR_H_
#define DAISYKIT_MODELS_FACE_RECOGNITION_ANCHOR_GENERATOR_H_

#include <iostream>
#include <vector>
#include "anchor_cfg.h"
#include "net.h"
#include "opencv2/opencv.hpp"

namespace daisykit {
namespace models {

class CRect2f {
 public:
  CRect2f(float x1, float y1, float x2, float y2) {
    val[0] = x1;
    val[1] = y1;
    val[2] = x2;
    val[3] = y2;
  }

  float& operator[](int i) { return val[i]; }

  float operator[](int i) const { return val[i]; }

  float val[4];
};

class Anchor {
 public:
  Anchor() {}

  ~Anchor() {}

  bool operator<(const Anchor& t) const { return score_ < t.score_; }

  bool operator>(const Anchor& t) const { return score_ > t.score_; }

  float& operator[](int i) {
    assert(0 <= i && i <= 4);
    if (i == 0) return finalbox_.x;
    if (i == 1) return finalbox_.y;
    if (i == 2) return finalbox_.width;
    if (i == 3) return finalbox_.height;
  };

  float operator[](int i) const {
    assert(0 <= i && i <= 4);

    if (i == 0) return finalbox_.x;
    if (i == 1) return finalbox_.y;
    if (i == 2) return finalbox_.width;
    if (i == 3) return finalbox_.height;
  };

  cv::Rect_<float> anchor_;
  float reg_[4];
  cv::Point center_;
  float score_;
  std::vector<cv::Point2f> pts_;
  cv::Rect_<float> finalbox_;
};

class AnchorGenerator {
 public:
  AnchorGenerator();
  ~AnchorGenerator();

  // init different anchors
  int Init(int stride, const AnchorCfg& cfg, bool dense_anchor_);

  // anchor plane
  int Generate(int fwidth, int fheight, int stride, float step,
               std::vector<int>& size, std::vector<float>& ratio,
               bool dense_anchor_);

  // filter anchors and return valid anchors
  int FilterAnchor(ncnn::Mat& cls, ncnn::Mat& reg, ncnn::Mat& pts,
                   std::vector<Anchor>& result, float cls_threshold);
  int anchor_num_;  // anchor type num

 private:
  void RatioEnum(const CRect2f& anchor, const std::vector<float>& ratios,
                 std::vector<CRect2f>& ratio_anchors);

  void ScaleEnum(const std::vector<CRect2f>& ratio_anchor,
                 const std::vector<float>& scales,
                 std::vector<CRect2f>& scale_anchors);

  void BboxPred(const CRect2f& anchor, const CRect2f& delta,
                cv::Rect_<float>& box);

  void LandmarkPred(const CRect2f anchor, const std::vector<cv::Point2f>& delta,
                    std::vector<cv::Point2f>& pts);

  std::vector<std::vector<Anchor>> anchor_planes_;  // corrspont to channels

  std::vector<int> anchor_size_;
  std::vector<float> anchor_ratio_;
  float anchor_step_;  // scale step
  int anchor_stride_;  // anchor tile stride
  int feature_w_;      // feature map width
  int feature_h_;      // feature map height

  std::vector<CRect2f> preset_anchors_;
};

}  // namespace models
}  // namespace daisykit
#endif
