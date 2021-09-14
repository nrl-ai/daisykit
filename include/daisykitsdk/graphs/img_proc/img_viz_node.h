#ifndef DAISYKIT_GRAPHS_IMG_PROC_IMG_VIZ_NODE_H_
#define DAISYKIT_GRAPHS_IMG_PROC_IMG_VIZ_NODE_H_

#include <daisykitsdk/graphs/core/node.h>
#include <daisykitsdk/graphs/core/utils.h>

#include <memory>

namespace daisykit {
namespace graphs {

class ImgVizNode : public Node {
 public:
  using Node::Node;  // For constructor inheritance

  void Process(PacketPtr in_packet, PacketPtr& out_packet) {}

  void Tick() {
    WaitForData();

    std::map<std::string, PacketPtr> inputs;
    if (PrepareInputs(inputs) != 0) {
      std::cerr << GetNodeName() << ": Error on preparing inputs." << std::endl;
      return;
    }

    if (inputs.count("binary") > 0) {
      PacketPtr binary_input = inputs["binary"];
      cv::Mat binary;
      Packet2CvMat(binary_input, binary);
      cv::imshow("Binary", binary);
      cv::waitKey(1);
    }

    if (inputs.count("gray") > 0) {
      PacketPtr gray_input = inputs["gray"];
      cv::Mat gray;
      Packet2CvMat(gray_input, gray);
      cv::imshow("Gray", gray);
      cv::waitKey(1);
    }
  }
};

}  // namespace graphs
}  // namespace daisykit

#endif