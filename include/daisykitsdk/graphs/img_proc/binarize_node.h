#ifndef DAISYKIT_GRAPHS_IMG_PROC_BINARIZE_NODE_H_
#define DAISYKIT_GRAPHS_IMG_PROC_BINARIZE_NODE_H_

#include <daisykitsdk/graphs/core/node.h>
#include <daisykitsdk/graphs/core/utils.h>

#include <chrono>
#include <memory>

namespace daisykit {
namespace graphs {

class BinarizeNode : public Node {
 public:
  using Node::Node;  // For constructor inheritance
  void Process(std::shared_ptr<Packet> in_packet,
               std::shared_ptr<Packet>& out_packet) {
    // Convert packet to processing format: cv::Mat
    cv::Mat img;
    Packet2CvMat(in_packet, img);

    // Process
    cv::Mat thres;
    cv::threshold(img, thres, 127, 255, cv::THRESH_BINARY);

    // Simulate heavy task
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Convert & assign output to the output packet
    CvMat2Packet(thres, out_packet);
  }

  void Tick() {
    WaitForData();

    std::map<std::string, PacketPtr> inputs;
    if (PrepareInputs(inputs) != 0) {
      std::cerr << GetNodeName() << ": Error on preparing inputs." << std::endl;
      return;
    }

    PacketPtr input = inputs["input"];
    PacketPtr output;
    Process(input, output);

    std::map<std::string, PacketPtr> outputs;
    outputs.insert(std::make_pair("output", output));
    Publish(outputs);
  }
};

}  // namespace graphs
}  // namespace daisykit

#endif