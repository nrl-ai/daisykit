#ifndef DAISYKIT_COMMON_TYPES_H_
#define DAISYKIT_COMMON_TYPES_H_

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

enum Action { kUnknown = 0, kPushup = 1 };

}  // namespace common
}  // namespace daisykit

#endif