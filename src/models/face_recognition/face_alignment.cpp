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
// limitations under the License.

#include "daisykit/models/face_recognition/face_alignment.h"

namespace daisykit {
namespace models {
FaceAligner::FaceAligner(){};
FaceAligner::~FaceAligner(){};

void FaceAligner::AlignFace(const cv::Mat& img,
                            daisykit::types::FaceExtended& face) {
  float points_src[5][2] = {face.landmark[0].x, face.landmark[0].y,
                            face.landmark[1].x, face.landmark[1].y,
                            face.landmark[2].x, face.landmark[2].y,
                            face.landmark[3].x, face.landmark[3].y,
                            face.landmark[4].x, face.landmark[4].y};
  cv::Mat src_mat(5, 2, CV_32FC1, points_src);
  float points_dst[5][2] = {{30.2946f + 8.0f, 51.6963f},
                            {65.5318f + 8.0f, 51.5014f},
                            {48.0252f + 8.0f, 71.7366f},
                            {33.5493f + 8.0f, 92.3655f},
                            {62.7299f + 8.0f, 92.2041f}};

  cv::Mat dst_mat(5, 2, CV_32FC1, points_dst);
  cv::Mat transform = SimilarTransform(src_mat, dst_mat);
  face.aligned_face.create(112, 112, CV_32FC3);
  cv::Mat transfer_mat = transform(cv::Rect(0, 0, 3, 2));
  cv::warpAffine(img.clone(), face.aligned_face, transfer_mat,
                 cv::Size(112, 112), 1, 0, 0);
}
void FaceAligner::AlignMutipleFaces(
    const cv::Mat& img, std::vector<daisykit::types::FaceExtended>& faces) {
  if (faces.size() == 0) return;
  for (int i = 0; i < faces.size(); i++) {
    AlignFace(img, faces[i]);
  }
}

cv::Mat FaceAligner::MeanAxis0(const cv::Mat& src) {
  int num = src.rows;
  int dim = src.cols;
  cv::Mat output(1, dim, CV_32FC1);
  for (int i = 0; i < dim; i++) {
    float sum = 0;
    for (int j = 0; j < num; j++) {
      sum += src.at<float>(j, i);
    }
    output.at<float>(0, i) = sum / num;
  }
  return output;
}

cv::Mat FaceAligner::ElementwiseMinus(const cv::Mat& A, const cv::Mat& B) {
  cv::Mat output(A.rows, A.cols, A.type());
  assert(B.cols == A.cols);
  if (B.cols == A.cols) {
    for (int i = 0; i < A.rows; i++) {
      for (int j = 0; j < B.cols; j++) {
        output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
      }
    }
  }
  return output;
}

cv::Mat FaceAligner::VarAxis0(const cv::Mat& src) {
  cv::Mat temp_ = ElementwiseMinus(src, MeanAxis0(src));
  cv::multiply(temp_, temp_, temp_);
  return MeanAxis0(temp_);
}

int FaceAligner::MatrixRank(cv::Mat M) {
  cv::Mat w, u, vt;
  cv::SVD::compute(M, w, u, vt);
  cv::Mat1b nonZeroSingularValues = w > 0.0001;
  int rank = countNonZero(nonZeroSingularValues);
  return rank;
}

/*
References: "Least-squares estimation of transformation parameters between two
point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573 Anthor: Jack
Yu
*/
cv::Mat FaceAligner::SimilarTransform(const cv::Mat& src, const cv::Mat& dst) {
  int num = src.rows;
  int dim = src.cols;
  cv::Mat src_mean = MeanAxis0(src);
  cv::Mat dst_mean = MeanAxis0(dst);
  cv::Mat src_demean = ElementwiseMinus(src, src_mean);
  cv::Mat dst_demean = ElementwiseMinus(dst, dst_mean);
  cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
  cv::Mat d(dim, 1, CV_32F);
  d.setTo(1.0f);
  if (cv::determinant(A) < 0) {
    d.at<float>(dim - 1, 0) = -1;
  }
  cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
  cv::Mat U, S, V;
  cv::SVD::compute(A, S, U, V);

  int rank = MatrixRank(A);
  if (rank == 0) {
    assert(rank == 0);

  } else if (rank == dim - 1) {
    if (cv::determinant(U) * cv::determinant(V) > 0) {
      T.rowRange(0, dim).colRange(0, dim) = U * V;
    } else {
      int s = d.at<float>(dim - 1, 0) = -1;
      d.at<float>(dim - 1, 0) = -1;

      T.rowRange(0, dim).colRange(0, dim) = U * V;
      cv::Mat diag_ = cv::Mat::diag(d);
      cv::Mat twp = diag_ * V;
      cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
      cv::Mat C = B.diag(0);
      T.rowRange(0, dim).colRange(0, dim) = U * twp;
      d.at<float>(dim - 1, 0) = s;
    }
  } else {
    cv::Mat diag_ = cv::Mat::diag(d);
    cv::Mat twp = diag_ * V.t();
    cv::Mat res = U * twp;
    T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
  }
  cv::Mat var_ = VarAxis0(src_demean);
  float val = cv::sum(var_).val[0];
  cv::Mat res;
  cv::multiply(d, S, res);
  float scale = 1.0 / val * cv::sum(res).val[0];
  T.rowRange(0, dim).colRange(0, dim) =
      -T.rowRange(0, dim).colRange(0, dim).t();
  cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim);  // T[:dim, :dim]
  cv::Mat temp2 = src_mean.t();
  cv::Mat temp3 = temp1 * temp2;
  cv::Mat temp4 = scale * temp3;
  T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
  T.rowRange(0, dim).colRange(0, dim) *= scale;
  return T;
}

}  // namespace models
}  // namespace daisykit
