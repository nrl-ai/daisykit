// Copyright 2021 The DaisyKit Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License

#ifndef FACE_ALIGNMENT_H_
#define FACE_ALIGNMENT_H_

#include <opencv2/opencv.hpp>
#include "daisykit/common/types.h"

namespace daisykit {
namespace models {
class FaceAligner {
 public:
  FaceAligner();
  ~FaceAligner();
  void AlignFace(const cv::Mat& img, daisykit::types::FaceExtended& face);
  void AlignMutipleFaces(const cv::Mat& img,
                         std::vector<daisykit::types::FaceExtended>& faces);

 private:
  cv::Mat MeanAxis0(const cv::Mat& src);
  cv::Mat ElementwiseMinus(const cv::Mat& A, const cv::Mat& B);
  cv::Mat VarAxis0(const cv::Mat& src);
  int MatrixRank(cv::Mat M);
  cv::Mat SimilarTransform(const cv::Mat& src, const cv::Mat& dst);
};

}  // namespace models
}  // namespace daisykit
#endif
