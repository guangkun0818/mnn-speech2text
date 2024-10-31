// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Wrapped Joiner of transducer of mnn runtime.

#ifndef _MNN_S2T_MODEL_JOINER_H_
#define _MNN_S2T_MODEL_JOINER_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "mnn-s2t/common/common.h"

namespace s2t {
namespace models {

// Joiner of Transducer
// inputTensors : [ enc_out, pred_out, ]
// outputTensors: [ logit, ]
class MnnJoiner {
 public:
  explicit MnnJoiner(const char* joiner_model);

  ~MnnJoiner();

  void Init(const int beam_size);

  void Reset();

  void StreamingStep(mnn::Tensor* enc_out, mnn::Tensor* pred_out);

  // Return logits with shape (beam_size, vocab_size)
  std::vector<std::vector<float>> GetJoinerOut() const;

 private:
  // Model resource.
  std::shared_ptr<mnn::Interpreter> model_;

  // Forward session.
  mnn::ScheduleConfig config_;
  mnn::Session* session_;
};

}  // namespace models
}  // namespace s2t

#endif