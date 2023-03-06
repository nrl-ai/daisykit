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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define SWAP(a, b) \
  do {             \
    float tmp;     \
    tmp = a;       \
    a = b;         \
    b = tmp;       \
  } while (0)
namespace daisykit {
namespace models {
static double PYTHAG(double a, double b) {
  double at = fabs(a), bt = fabs(b), ct, result;

  if (at > bt) {
    ct = bt / at;
    result = at * sqrt(1.0 + ct * ct);
  } else if (bt > 0.0) {
    ct = at / bt;
    result = bt * sqrt(1.0 + ct * ct);
  } else
    result = 0.0;
  return (result);
}

int DSVD(float a[][2], int m, int n, float* w, float v[][2]) {
  int flag, i, its, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  double* rv1;
  double rv1_buf[8];

  if (m < n) {
    fprintf(stderr, "#rows must be > #cols \n");
    return (0);
  }
  rv1 = rv1_buf;
  for (i = 0; i < n; i++) {
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs((double)a[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          a[k][i] = (float)((double)a[k][i] / scale);
          s += ((double)a[k][i] * (double)a[k][i]);
        }
        f = (double)a[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        a[i][i] = (float)(f - g);
        if (i != n - 1) {
          for (j = l; j < n; j++) {
            for (s = 0.0, k = i; k < m; k++)
              s += ((double)a[k][i] * (double)a[k][j]);
            f = s / h;
            for (k = i; k < m; k++) a[k][j] += (float)(f * (double)a[k][i]);
          }
        }
        for (k = i; k < m; k++) a[k][i] = (float)((double)a[k][i] * scale);
      }
    }
    w[i] = (float)(scale * g);
    g = s = scale = 0.0;
    if (i < m && i != n - 1) {
      for (k = l; k < n; k++) scale += fabs((double)a[i][k]);
      if (scale) {
        for (k = l; k < n; k++) {
          a[i][k] = (float)((double)a[i][k] / scale);
          s += ((double)a[i][k] * (double)a[i][k]);
        }
        f = (double)a[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        a[i][l] = (float)(f - g);
        for (k = l; k < n; k++) rv1[k] = (double)a[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < n; k++)
              s += ((double)a[j][k] * (double)a[i][k]);
            for (k = l; k < n; k++) a[j][k] += (float)(s * rv1[k]);
          }
        }
        for (k = l; k < n; k++) a[i][k] = (float)((double)a[i][k] * scale);
      }
    }
    anorm = MAX(anorm, (fabs((double)w[i]) + fabs(rv1[i])));
  }

  for (i = n - 1; i >= 0; i--) {
    if (i < n - 1) {
      if (g) {
        for (j = l; j < n; j++)
          v[j][i] = (float)(((double)a[i][j] / (double)a[i][l]) / g);
        for (j = l; j < n; j++) {
          for (s = 0.0, k = l; k < n; k++)
            s += ((double)a[i][k] * (double)v[k][j]);
          for (k = l; k < n; k++) v[k][j] += (float)(s * (double)v[k][i]);
        }
      }
      for (j = l; j < n; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }
  for (i = n - 1; i >= 0; i--) {
    l = i + 1;
    g = (double)w[i];
    if (i < n - 1)
      for (j = l; j < n; j++) a[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != n - 1) {
        for (j = l; j < n; j++) {
          for (s = 0.0, k = l; k < m; k++)
            s += ((double)a[k][i] * (double)a[k][j]);
          f = (s / (double)a[i][i]) * g;
          for (k = i; k < m; k++) a[k][j] += (float)(f * (double)a[k][i]);
        }
      }
      for (j = i; j < m; j++) a[j][i] = (float)((double)a[j][i] * g);
    } else {
      for (j = i; j < m; j++) a[j][i] = 0.0;
    }
    ++a[i][i];
  }

  for (k = n - 1; k >= 0; k--) {
    for (its = 0; its < 30; its++) {
      flag = 1;
      for (l = k; l >= 0; l--) {
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs((double)w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = (double)w[i];
            h = PYTHAG(f, g);
            w[i] = (float)h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = (double)a[j][nm];
              z = (double)a[j][i];
              a[j][nm] = (float)(y * c + z * s);
              a[j][i] = (float)(z * c - y * s);
            }
          }
        }
      }
      z = (double)w[k];
      if (l == k) {
        if (z < 0.0) {
          w[k] = (float)(-z);
          for (j = 0; j < n; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        fprintf(stderr, "No convergence after 30,000! iterations \n");
        return (0);
      }

      x = (double)w[l];
      nm = k - 1;
      y = (double)w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = (double)w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < n; jj++) {
          x = (double)v[jj][j];
          z = (double)v[jj][i];
          v[jj][j] = (float)(x * c + z * s);
          v[jj][i] = (float)(z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = (float)z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = (double)a[jj][j];
          z = (double)a[jj][i];
          a[jj][j] = (float)(y * c + z * s);
          a[jj][i] = (float)(z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = (float)x;
    }
  }
  return (1);
}

static float FrobNorm(float* f, int n) {
  float sum = 0;

  for (int i = 0; i < n; i++) sum += f[i] * f[i];

  return sqrt(sum);
}

static void MatrixDot(float* a, float* b, float* c, int m, int n, int k) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      c[i * n + j] = 0;
      for (int l = 0; l < k; l++) {
        c[i * n + j] += a[i * k + l] * b[l * n + j];
      }
    }
}

static void ComputeAffineMatrix(float cov[][2], float sigma,
                                float trans_m[][2]) {
  float u[2][2];
  float w[2];
  float v[2][2];
  float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
  float s[2][2] = {{1, 0}, {0, 1}};
  float c = 1.0;
  float r0[2][2];
  float r[2][2];

  u[0][0] = cov[0][0];
  u[0][1] = cov[0][1];

  u[1][0] = cov[1][0];
  u[1][1] = cov[1][1];

  DSVD(u, 2, 2, (float*)w, v);

  SWAP(u[0][0], u[0][1]);
  SWAP(u[1][0], u[1][1]);
  SWAP(w[0], w[1]);
  SWAP(v[0][0], v[0][1]);
  SWAP(v[1][0], v[1][1]);

  if (det < 0) {
    if (w[0] < w[1])
      s[1][1] = -1;
    else
      s[0][0] = -1;
  }

  MatrixDot((float*)u, (float*)s, (float*)r0, 2, 2, 2);
  MatrixDot((float*)r0, (float*)v, (float*)r, 2, 2, 2);

  if (sigma) {
    float diag[2][2];
    float trace[2][2];

    diag[0][0] = w[0];
    diag[0][1] = 0;
    diag[1][0] = 0;
    diag[1][1] = w[1];

    MatrixDot((float*)diag, (float*)s, (float*)trace, 2, 2, 2);

    c = 1.0 / sigma * (trace[0][0] + trace[1][1]);
  }

  trans_m[0][0] = r[0][0] * c;
  trans_m[0][1] = r[0][1] * c;
  trans_m[1][0] = r[1][0] * c;
  trans_m[1][1] = r[1][1] * c;
}

static int GetProbeVec(float* landmark, int landmark_number, float* probe_vec,
                       int probe_size, int desired_size) {
  int i;

  float mean_face_x[] = {0.224152, 0.75610125, 0.490127, 0.254149, 0.726104};
  float mean_face_y[] = {0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233};
  float from_mean[2] = {0, 0};
  float to_mean[2] = {0, 0};
  float sigma_from = 0;
  float sigma_to = 0;
  float cov[2][2] = {{0, 0}, {0, 0}};
  float trans_m[2][2];
  float padding = 0.37;

  if (landmark_number != 5 || probe_size != 2) return -1;

  for (i = 0; i < 5; i++) {
    mean_face_x[i] =
        (padding + mean_face_x[i]) / (2 * padding + 1) * desired_size;
    mean_face_y[i] =
        (padding + mean_face_y[i]) / (2 * padding + 1) * desired_size;
  }

  for (i = 0; i < 5; i++) {
    from_mean[0] += landmark[i];
    from_mean[1] += landmark[i + 5];
    to_mean[0] += mean_face_x[i];
    to_mean[1] += mean_face_y[i];
  }

  from_mean[0] = from_mean[0] / 5;
  from_mean[1] = from_mean[1] / 5;

  to_mean[0] = to_mean[0] / 5;
  to_mean[1] = to_mean[1] / 5;

  for (i = 0; i < 5; i++) {
    float gap_from[2];
    float gap_to[2];
    float distance;

    gap_from[0] = landmark[i] - from_mean[0];
    gap_from[1] = landmark[i + 5] - from_mean[1];

    distance = FrobNorm(gap_from, 2);

    sigma_from += distance * distance;

    gap_to[0] = mean_face_x[i] - to_mean[0];
    gap_to[1] = mean_face_y[i] - to_mean[1];

    distance = FrobNorm(gap_to, 2);

    sigma_to += distance * distance;

    cov[0][0] += gap_to[0] * gap_from[0];
    cov[0][1] += gap_to[0] * gap_from[1];
    cov[1][0] += gap_to[1] * gap_from[0];
    cov[1][1] += gap_to[1] * gap_from[1];
  }

  sigma_from = sigma_from / 5;
  sigma_to = sigma_to / 5;

  cov[0][0] /= 5;
  cov[0][1] /= 5;
  cov[1][0] /= 5;
  cov[1][1] /= 5;

  ComputeAffineMatrix(cov, sigma_from, trans_m);

  probe_vec[0] = trans_m[0][0];
  probe_vec[1] = trans_m[1][0];

  return 0;
}

int CalScaleAndAngle(float* landmark, int landmark_number, int desired_size,
                     float* scale, float* angle) {
  float probe_vec[2];

  if (GetProbeVec(landmark, landmark_number, probe_vec, 2, desired_size) < 0)
    return -1;

  scale[0] = FrobNorm(probe_vec, 2);
  angle[0] = 180.0 / M_PI * atan2(probe_vec[1], probe_vec[0]);

  return 0;
}

int GetAlignedFace(cv::Mat& img, float* landmark, int landmark_number,
                   int desired_size, cv::Mat& out) {
  float scale;
  float angle;
  float from_center[2];
  float to_center[2];

  if (CalScaleAndAngle(landmark, landmark_number, desired_size, &scale,
                       &angle) < 0) {
    return -1;
  }

  to_center[0] = desired_size * 0.4;
  to_center[1] = desired_size * 0.5;

  from_center[0] = (landmark[0] + landmark[1]) / 2;
  from_center[1] = (landmark[5] + landmark[6]) / 2;

  cv::Mat rot_mat = cv::getRotationMatrix2D(
      cv::Point2f(from_center[0], from_center[1]), -1 * angle, scale);

  float ex = to_center[0] - from_center[0];
  float ey = to_center[1] - from_center[1];

  rot_mat.at<double>(0, 2) += ex;
  rot_mat.at<double>(1, 2) += ey;

  cv::warpAffine(img, out, rot_mat, cv::Size(desired_size, desired_size));
  rot_mat.release();

  return 0;
}
}  // namespace models
}  // namespace daisykit
