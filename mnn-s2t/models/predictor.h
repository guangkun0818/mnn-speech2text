// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.30
// Wrapped Predictor of transducer of mnn runtime.

#ifndef _MNN_S2T_MODEL_PREDICTOR_H_
#define _MNN_S2T_MODEL_PREDICTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "mnn-s2t/common/common.h"

namespace s2t {
namespace models {

struct MnnPredictorCfg {
  std::string predictor_model;
  size_t context_size;
};

// Predictor of Transducer
// inputTensors : [ prev_states, pred_in, ]
// outputTensors: [ next_states, pred_out, ]
class MnnPredictor {
 public:
  explicit MnnPredictor(const MnnPredictorCfg& cfg, mnn::ScheduleConfig config);

  ~MnnPredictor();

  mnn::Session* Init(const int beam_size);

  void Reset(mnn::Session* session);

  void StreamingStep(const std::vector<int>& pred_in, mnn::Session* session);

  mnn::Tensor* GetPredOut(mnn::Session* session);

 private:
  // Model resource.
  std::shared_ptr<mnn::Interpreter> model_;

  // Forward session.
  mnn::ScheduleConfig config_;

  // Predictor states.
  int context_size_;
};

}  // namespace models
}  // namespace s2t

#endif