#include <daisykitsdk/flows/human_matting_flow.h>

using namespace daisykit::flows;

HumanMattingFlow::HumanMattingFlow(const std::string &config_str,
                                   const cv::Mat &default_background) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  human_matting_model_ =
      new models::HumanMatting(config["human_matting_model"]["model"],
                               config["human_matting_model"]["weights"]);
  background_ = default_background.clone();
}

#ifdef __ANDROID__
HumanMattingFlow::HumanMattingFlow(AAssetManager *mgr,
                                   const std::string &config_str,
                                   const cv::Mat &default_background) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  human_matting_model_ =
      new models::HumanMatting(mgr, config["human_matting_model"]["model"],
                               config["human_matting_model"]["weights"]);
  background_ = default_background.clone();
}
#endif

HumanMattingFlow::~HumanMattingFlow() {
  delete human_matting_model_;
  human_matting_model_ = nullptr;
}

void HumanMattingFlow::Process(cv::Mat &rgb) {
  cv::Mat mask;
  human_matting_model_->Segmentation(rgb, mask);

  {
    const std::lock_guard<std::mutex> lock(mask_lock_);
    mask_ = mask;
  }
}

void HumanMattingFlow::DrawResult(cv::Mat &rgb) {
  {
    const std::lock_guard<std::mutex> lock(mask_lock_);
    human_matting_model_->BindWithBackground(rgb, background_, mask_);
  }
}