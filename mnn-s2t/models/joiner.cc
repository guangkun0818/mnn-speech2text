// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Wrapped Joiner of transducer of mnn runtime.

#include "mnn-s2t/models/joiner.h"

namespace s2t {
namespace models {

MnnJoiner::MnnJoiner(const MnnJoinerCfg& cfg, mnn::ScheduleConfig config)
    : config_(config) {
  this->model_ = std::shared_ptr<mnn::Interpreter>(
      mnn::Interpreter::createFromFile(cfg.joiner_model.c_str()));
  CHECK_NE(this->model_, nullptr);
}

MnnJoiner::~MnnJoiner() {}

mnn::Session* MnnJoiner::Init(const int beam_size) {
  auto session = model_->createSession(config_);
  CHECK_NE(session, nullptr);

  // Predictor Output shape: {beam_size, 1, pred_out_dim_}
  auto pred_out_shape =
      this->model_->getSessionInput(session, "pred_out")->shape();
  pred_out_shape[0] = beam_size;

  this->model_->resizeTensor(this->model_->getSessionInput(session, "pred_out"),
                             pred_out_shape);

  // Resize session with input beam_size.
  this->model_->resizeSession(session);
  return session;
}

mnn::Session* MnnJoiner::Reset(mnn::Session* session) {
  if (session) {
    CHECK(this->model_->releaseSession(session));
  }
  return nullptr;
}

void MnnJoiner::StreamingStep(mnn::Tensor* enc_out, mnn::Tensor* pred_out,
                              mnn::Session* session) {
  CHECK_NE(session, nullptr);
  this->model_->getSessionInput(session, "enc_out")
      ->copyFromHostTensor(enc_out);
  this->model_->getSessionInput(session, "pred_out")
      ->copyFromHostTensor(pred_out);

  this->model_->runSession(session);
}

std::vector<std::vector<float>> MnnJoiner::GetJoinerOut(
    mnn::Session* session) const {
  CHECK_NE(session, nullptr);
  auto logit_t = this->model_->getSessionOutput(session, "logit");
  auto logit_shape = logit_t->shape();
  std::vector<std::vector<float>> logit_vec(
      logit_shape[0],
      std::vector<float>(logit_shape[1]));  // {beam_size, vocab_size}

  for (int beam = 0; beam < logit_shape[0]; beam++) {
    // Copy logits of each beam from output logit tensor.
    memcpy(logit_vec[beam].data(),
           logit_t->host<float>() + (beam * logit_shape[1]),
           sizeof(float) * logit_shape[1]);
  }
  return logit_vec;
}

}  // namespace models
}  // namespace s2t