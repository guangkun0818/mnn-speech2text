// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.23
// Wrapped model session of transducer for threading.

#ifndef _MNN_S2T_MODEL_MODEL_SESSION_H_
#define _MNN_S2T_MODEL_MODEL_SESSION_H_

#include <memory>
#include <vector>

#include "mnn-s2t/common/common.h"

namespace s2t {
namespace models {

static mnn::ScheduleConfig CPU_FORWARD_THREAD_8 = {
    .type = MNNForwardType::MNN_FORWARD_CPU,
    .numThread = 8,
};  // TODO: adaptation for edge device.;

struct RnntModelSession {
  mnn::Session* encoder_session = nullptr;
  mnn::Session* predictor_session = nullptr;
  mnn::Session* joiner_session = nullptr;
};

struct CtcModelSession {};

}  // namespace models
}  // namespace s2t

#endif