// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.30
// Wrapped Predictor of transducer of mnn runtime.

#include "mnn-s2t/models/predictor.h"

namespace s2t {
namespace models {

MnnPredictor::MnnPredictor(const char* predictor_model, size_t context_size)
    : context_size_(context_size) {
  // TODO(guangkun0818): modify to support more config.
  config_.numThread = 8;
  config_.type = MNNForwardType::MNN_FORWARD_CPU;

  this->model_ = std::shared_ptr<mnn::Interpreter>(
      mnn::Interpreter::createFromFile(predictor_model));
  CHECK_NE(this->model_, nullptr);
  this->session_ = nullptr;
}

MnnPredictor::~MnnPredictor() { this->Reset(); }

void MnnPredictor::Init(const int beam_size) {
  this->session_ = model_->createSession(config_);
  CHECK_NE(this->session_, nullptr);

  // Predictor state shape: {beam_size, context_size - 1}
  std::vector<int> init_input_shape = {beam_size, this->context_size_ - 1};

  auto prev_states_t =
      this->model_->getSessionInput(this->session_, "prev_states");
  this->model_->resizeTensor(prev_states_t, init_input_shape);

  // Init predictor state as 0
  for (int i = 0; i < prev_states_t->elementSize(); i++) {
    prev_states_t->host<int>()[i] = 0;
  }

  std::vector<int> pred_in_shape = {beam_size, 1};  // {beam_size, 1}
  this->model_->resizeTensor(
      this->model_->getSessionInput(this->session_, "pred_in"), pred_in_shape);

  // Resize session with input beam_size.
  this->model_->resizeSession(this->session_);
}

void MnnPredictor::Reset() {
  if (this->session_) {
    CHECK(this->model_->releaseSession(this->session_));
    this->session_ = nullptr;
  }
}

void MnnPredictor::StreamingStep(const std::vector<int>& pred_in) {
  auto pred_in_t = this->model_->getSessionInput(this->session_, "pred_in");
  CHECK_EQ(pred_in_t->elementSize(), pred_in.size());

  // Copy input into session.
  for (int i = 0; i < pred_in_t->elementSize(); i++) {
    pred_in_t->host<int>()[i] = pred_in[i];
  }
  this->model_->runSession(this->session_);

  // Update predictor states.
  this->model_->getSessionInput(this->session_, "prev_states")
      ->copyFromHostTensor(
          this->model_->getSessionOutput(this->session_, "next_states"));
}

mnn::Tensor* MnnPredictor::GetPredOut() {
  return this->model_->getSessionOutput(this->session_, "pred_out");
}

}  // namespace models
}  // namespace s2t