// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.30
// Wrapped Predictor of transducer of mnn runtime.

#ifndef _MNN_S2TRT_MODEL_PREDICTOR_H_
#define _MNN_S2TRT_MODEL_PREDICTOR_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "mnn-s2trt/common/common.h"

namespace s2trt {
namespace models {

// Predictor of Transducer
// inputTensors : [ prev_states, pred_in, ]
// outputTensors: [ next_states, pred_out, ]
class MnnPredictor {
 public:
  explicit MnnPredictor(const char* predictor_model, size_t context_size);

  ~MnnPredictor(){};

  void Init(const int beam_size);

  void StreamingStep(const std::vector<int>& pred_in);

  mnn::Tensor* GetPredOut();

 private:
  // Model resource.
  std::shared_ptr<mnn::Interpreter> model_;

  // Forward session.
  mnn::ScheduleConfig config_;
  mnn::Session* session_;

  // Predictor states.
  int context_size_;
};

}  // namespace models
}  // namespace s2trt

#endif