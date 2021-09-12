/*
Header for class Base Model, the abstract class for all models.
All models should be inherited from this class to guarantee
the operation of loading model, predicting model and other basics.
*/

#ifndef DAISYKIT_MODELS_BASE_MODEL_H_
#define DAISYKIT_MODELS_BASE_MODEL_H_

#include <net.h>

namespace daisykit {
namespace models {

template <typename TInput, typename TOutput>
class BaseModel {
 public:
  // No abstract the constructor function for arbitary initialization
  virtual ~BaseModel();

  // Only load model from buffer to avoid depending on any format
  // in different platforms. Use module io to convert model to appropriate
  // format.
  virtual int LoadModel(const char* param_buffer,
                        const unsigned char* weight_buffer);

  // Because there are many kinds of data (exp: image, voice,...).
  // Inherited models should decide what data type they process by themselves.
  virtual TOutput Predict(const TInput& input);

 protected:
  ncnn::Net model_;
  ncnn::Mutex lock_;
};

}  // namespace models
}  // namespace daisykit

#endif
