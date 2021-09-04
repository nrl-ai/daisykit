#ifndef GRAPH_H
#define GRAPH_H

template <typename T>
cv::Mat plot_graph(std::vector<T>& vals, int YRange[2], vector<int> signals,
                   int n_last_values = -1) {
  auto begin_elm = n_last_values == -1 ? vals.begin()
                                       : vals.size() > n_last_values
                                             ? vals.end() - n_last_values
                                             : vals.begin();
  auto it = minmax_element(begin_elm, vals.end());
  float scale = 1. / ceil(*it.second - *it.first);
  float bias = *it.first;
  int rows = YRange[1] - YRange[0] + 1;

  int graph_y_size = n_last_values != -1
                         ? std::min(n_last_values, (int)vals.size())
                         : vals.size();
  cv::Mat image = Mat::zeros(rows, graph_y_size, CV_8UC3);
  image.setTo(0);

  int k = 0;
  if (n_last_values != -1) {
    if (vals.size() > n_last_values) {
      k = vals.size() - n_last_values;
    }
  }

  for (int i = k; i < (int)vals.size() - 1; i++) {
    cv::line(image,
             cv::Point(i - k, rows - 1 - (vals[i] - bias) * scale * YRange[1]),
             cv::Point(i + 1 - k,
                       rows - 1 - (vals[i + 1] - bias) * scale * YRange[1]),
             Scalar(255, 0, 0), 1);
  }

  for (int i = k; i < (int)signals.size() - 1; i++) {
    if (signals[i] == 0) {
      continue;
    }
    int y = signals[i] == 1 ? (int)(rows / 2) - 20 : (int)(rows / 2) + 20;
    cv::circle(image, cv::Point(i - k, y), 2, cv::Scalar(0, 0, 255), cv::FILLED,
               cv::LINE_8);
  }

  return image;
}

#endif  // SHARKPUSHUPS_GRAPH_H
