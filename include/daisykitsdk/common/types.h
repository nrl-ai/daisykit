#ifndef DAISYKIT_COMMON_TYPES_H_
#define DAISYKIT_COMMON_TYPES_H_

#include <opencv2/opencv.hpp>

namespace daisykit {
namespace common {
struct Keypoint {
  float x;
  float y;
  float prob;
};

struct Object {
  float x;
  float y;
  float w;
  float h;
  int class_id;
  float confidence;
};

struct Face {
  float x;
  float y;
  float w;
  float h;
  float confidence;
  std::vector<Keypoint> landmark;
};

enum Action { kUnknown = 0, kPushup = 1 };

}  // namespace common
}  // namespace daisykit

#endif