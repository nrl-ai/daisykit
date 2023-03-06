#include "daisykit/models/face_recognition/anchor_generator.h"
namespace daisykit {
namespace models {
AnchorGenerator::AnchorGenerator() {}

AnchorGenerator::~AnchorGenerator() {}

int AnchorGenerator::Init(int stride, const AnchorCfg& cfg,
                          bool dense_anchor_) {
  CRect2f base_anchor(0, 0, cfg.base_size_ - 1, cfg.base_size_ - 1);
  std::vector<CRect2f> ratio_anchors;
  RatioEnum(base_anchor, cfg.ratios_, ratio_anchors);
  ScaleEnum(ratio_anchors, cfg.scales_, preset_anchors_);

  if (dense_anchor_) {
    assert(stride % 2 == 0);
    int num = preset_anchors_.size();
    for (int i = 0; i < num; ++i) {
      CRect2f anchor = preset_anchors_[i];
      preset_anchors_.push_back(
          CRect2f(anchor[0] + int(stride / 2), anchor[1] + int(stride / 2),
                  anchor[2] + int(stride / 2), anchor[3] + int(stride / 2)));
    }
  }
  anchor_stride_ = stride;
  anchor_num_ = preset_anchors_.size();
  return anchor_num_;
}

int AnchorGenerator::FilterAnchor(ncnn::Mat& cls, ncnn::Mat& reg,
                                  ncnn::Mat& pts, std::vector<Anchor>& result,
                                  float cls_threshold) {
  assert(cls.c == anchor_num_ * 2);
  assert(reg.c == anchor_num_ * 4);
  int pts_length = 0;

  assert(pts.c % anchor_num_ == 0);
  pts_length = pts.c / anchor_num_ / 2;

  int w = cls.w;
  int h = cls.h;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int id = i * w + j;
      for (int a = 0; a < anchor_num_; ++a) {
        if (cls.channel(anchor_num_ + a)[id] >= cls_threshold) {
          CRect2f box(j * anchor_stride_ + preset_anchors_[a][0],
                      i * anchor_stride_ + preset_anchors_[a][1],
                      j * anchor_stride_ + preset_anchors_[a][2],
                      i * anchor_stride_ + preset_anchors_[a][3]);
          CRect2f delta(reg.channel(a * 4 + 0)[id], reg.channel(a * 4 + 1)[id],
                        reg.channel(a * 4 + 2)[id], reg.channel(a * 4 + 3)[id]);

          Anchor res;
          res.anchor_ = cv::Rect_<float>(box[0], box[1], box[2], box[3]);
          BboxPred(box, delta, res.finalbox_);

          res.score_ = cls.channel(anchor_num_ + a)[id];
          res.center_ = cv::Point(j, i);

          std::vector<cv::Point2f> pts_delta(pts_length);
          for (int p = 0; p < pts_length; ++p) {
            pts_delta[p].x = pts.channel(a * pts_length * 2 + p * 2)[id];
            pts_delta[p].y = pts.channel(a * pts_length * 2 + p * 2 + 1)[id];
          }

          LandmarkPred(box, pts_delta, res.pts_);

          result.push_back(res);
        }
      }
    }
  }
  return 0;
}

void AnchorGenerator::RatioEnum(const CRect2f& anchor,
                                const std::vector<float>& ratios,
                                std::vector<CRect2f>& ratio_anchors) {
  float w = anchor[2] - anchor[0] + 1;
  float h = anchor[3] - anchor[1] + 1;
  float x_ctr = anchor[0] + 0.5 * (w - 1);
  float y_ctr = anchor[1] + 0.5 * (h - 1);

  ratio_anchors.clear();
  float sz = w * h;
  for (int s = 0; s < ratios.size(); ++s) {
    float r = ratios[s];
    float size_ratios = sz / r;
    float ws = std::sqrt(size_ratios);
    float hs = ws * r;
    ratio_anchors.push_back(
        CRect2f(x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)));
  }
}

void AnchorGenerator::ScaleEnum(const std::vector<CRect2f>& ratio_anchor,
                                const std::vector<float>& scales,
                                std::vector<CRect2f>& scale_anchors) {
  scale_anchors.clear();
  for (int a = 0; a < ratio_anchor.size(); ++a) {
    CRect2f anchor = ratio_anchor[a];
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5 * (w - 1);
    float y_ctr = anchor[1] + 0.5 * (h - 1);

    for (int s = 0; s < scales.size(); ++s) {
      float ws = w * scales[s];
      float hs = h * scales[s];
      scale_anchors.push_back(
          CRect2f(x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                  x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)));
    }
  }
}

void AnchorGenerator::BboxPred(const CRect2f& anchor, const CRect2f& delta,
                               cv::Rect_<float>& box) {
  float w = anchor[2] - anchor[0] + 1;
  float h = anchor[3] - anchor[1] + 1;
  float x_ctr = anchor[0] + 0.5 * (w - 1);
  float y_ctr = anchor[1] + 0.5 * (h - 1);

  float dx = delta[0];
  float dy = delta[1];
  float dw = delta[2];
  float dh = delta[3];

  float pred_ctr_x = dx * w + x_ctr;
  float pred_ctr_y = dy * h + y_ctr;
  float pred_w = std::exp(dw) * w;
  float pred_h = std::exp(dh) * h;

  box = cv::Rect_<float>(
      pred_ctr_x - 0.5 * (pred_w - 1.0), pred_ctr_y - 0.5 * (pred_h - 1.0),
      pred_ctr_x + 0.5 * (pred_w - 1.0), pred_ctr_y + 0.5 * (pred_h - 1.0));
}

void AnchorGenerator::LandmarkPred(const CRect2f anchor,
                                   const std::vector<cv::Point2f>& delta,
                                   std::vector<cv::Point2f>& pts) {
  float w = anchor[2] - anchor[0] + 1;
  float h = anchor[3] - anchor[1] + 1;
  float x_ctr = anchor[0] + 0.5 * (w - 1);
  float y_ctr = anchor[1] + 0.5 * (h - 1);

  pts.resize(delta.size());
  for (int i = 0; i < delta.size(); ++i) {
    pts[i].x = delta[i].x * w + x_ctr;
    pts[i].y = delta[i].y * h + y_ctr;
  }
}
}  // namespace models
}  // namespace daisykit
