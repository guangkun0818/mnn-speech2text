// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.30
// Wrapped Predictor of transducer of mnn runtime.

#include "mnn-s2t/models/predictor.h"

#include <cstring>

namespace s2t {
namespace models {

MnnPredictor::MnnPredictor(const MnnPredictorCfg& cfg,
                           mnn::ScheduleConfig config)
    : context_size_(cfg.context_size), config_(config) {
  this->model_ = std::shared_ptr<mnn::Interpreter>(
      mnn::Interpreter::createFromFile(cfg.predictor_model.c_str()));
  CHECK_NE(this->model_, nullptr);
}

MnnPredictor::~MnnPredictor() {}

mnn::Session* MnnPredictor::Init(const int beam_size) {
  auto session = model_->createSession(config_);
  CHECK_NE(session, nullptr);

  // Predictor state shape: {beam_size, context_size - 1}
  std::vector<int> init_input_shape = {beam_size, this->context_size_ - 1};

  auto prev_states_t = this->model_->getSessionInput(session, "prev_states");
  this->model_->resizeTensor(prev_states_t, init_input_shape);

  // Init predictor state as 0
  std::memset(prev_states_t->host<int>(), 0,
              sizeof(int) * prev_states_t->elementSize());

  std::vector<int> pred_in_shape = {beam_size, 1};  // {beam_size, 1}
  this->model_->resizeTensor(this->model_->getSessionInput(session, "pred_in"),
                             pred_in_shape);

  // Resize session with input beam_size.
  this->model_->resizeSession(session);
  return session;
}

mnn::Session* MnnPredictor::Reset(mnn::Session* session) {
  if (session) {
    CHECK(this->model_->releaseSession(session));
  }
  return nullptr;
}

void MnnPredictor::StreamingStep(const std::vector<int>& pred_in,
                                 mnn::Session* session) {
  CHECK_NE(session, nullptr);
  auto pred_in_t = this->model_->getSessionInput(session, "pred_in");
  CHECK_EQ(pred_in_t->elementSize(), pred_in.size());

  // Copy input into session.
  for (int i = 0; i < pred_in_t->elementSize(); i++) {
    pred_in_t->host<int>()[i] = pred_in[i];
  }
  this->model_->runSession(session);

  // Update predictor states.
  this->model_->getSessionInput(session, "prev_states")
      ->copyFromHostTensor(
          this->model_->getSessionOutput(session, "next_states"));
}

mnn::Tensor* MnnPredictor::GetPredOut(mnn::Session* session) {
  CHECK_NE(session, nullptr);
  return this->model_->getSessionOutput(session, "pred_out");
}

}  // namespace models
}  // namespace s2t