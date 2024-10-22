// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Wrapped Joiner of transducer of mnn runtime.

#include "mnn-s2trt/models/joiner.h"

namespace s2trt {
namespace models {

MnnJoiner::MnnJoiner(const char* joiner_model) {
  config_.numThread = 8;
  config_.type = MNNForwardType::MNN_FORWARD_CPU;

  this->model_ = std::shared_ptr<mnn::Interpreter>(
      mnn::Interpreter::createFromFile(joiner_model));
  CHECK_NE(this->model_, nullptr);

  this->session_ = model_->createSession(config_);
  CHECK_NE(this->session_, nullptr);
}

MnnJoiner::~MnnJoiner() { CHECK(this->model_->releaseSession(this->session_)); }

void MnnJoiner::Init(const int beam_size) {
  // Predictor Output shape: {beam_size, 1, pred_out_dim_}
  auto pred_out_shape =
      this->model_->getSessionInput(this->session_, "pred_out")->shape();
  pred_out_shape[0] = beam_size;

  this->model_->resizeTensor(
      this->model_->getSessionInput(this->session_, "pred_out"),
      pred_out_shape);

  // Resize session with input beam_size.
  this->model_->resizeSession(this->session_);
}

void MnnJoiner::StreamingStep(mnn::Tensor* enc_out, mnn::Tensor* pred_out) {
  this->model_->getSessionInput(this->session_, "enc_out")
      ->copyFromHostTensor(enc_out);
  this->model_->getSessionInput(this->session_, "pred_out")
      ->copyFromHostTensor(pred_out);

  this->model_->runSession(this->session_);
}

std::vector<std::vector<float>> MnnJoiner::GetJoinerOut() const {
  auto logit_t = this->model_->getSessionOutput(this->session_, "logit");
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
}  // namespace s2trt