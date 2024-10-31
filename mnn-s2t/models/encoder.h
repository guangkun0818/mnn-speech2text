// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.23
// Wrapped Encoder of transducer of mnn runtime.

#ifndef _MNN_S2T_MODEL_ENCODER_H_
#define _MNN_S2T_MODEL_ENCODER_H_

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "mnn-s2t/common/common.h"

namespace s2t {
namespace models {

class MnnEncoder {
 public:
  virtual ~MnnEncoder() {}
  virtual void Init(const int num_frames) = 0;
  virtual void Reset() = 0;
  virtual void Inference(const std::vector<std::vector<float>>& feats) = 0;
  virtual void StreamingStep(const std::vector<std::vector<float>>& feats) = 0;
  virtual mnn::Tensor* GetEncOut() = 0;
};

}  // namespace models
}  // namespace s2t

#endif