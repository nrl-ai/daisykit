#ifndef DEFINES_
#define DEFINES_

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

#endif