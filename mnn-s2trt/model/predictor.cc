// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.09.30
// Wrapped Predictor of transducer of mnn runtime.

#include "mnn-s2trt/model/predictor.h"

namespace s2trt {
namespace models {

MnnPredictor::MnnPredictor(const char* predictor_model, size_t context_size)
    : context_size_(context_size) {
  // TODO(guangkun0818): modify to support more config.
  config_.numThread = 8;
  config_.type = MNNForwardType::MNN_FORWARD_CPU;

  this->model_ = std::shared_ptr<mnn::Interpreter>(
      mnn::Interpreter::createFromFile(predictor_model));
  CHECK_NE(this->model_, nullptr);

  this->session_ = model_->createSession(config_);
  CHECK_NE(this->session_, nullptr);
}

void MnnPredictor::Init(const int beam_size) {
  // Predictor state shape: {beam_size, context_size - 1}
  std::vector<int> init_input_shape = {beam_size, this->context_size_ - 1};

  pred_state_ = std::shared_ptr<mnn::Tensor>(
      mnn::Tensor::create<float>(init_input_shape, NULL, MNN::Tensor::CAFFE));
  // Init predictor state as 0.0
  for (int i = 0; i < pred_state_->elementSize(); i++) {
    pred_state_->host<float>()[i] = 0.0;
  }
  this->model_->resizeTensor(
      this->model_->getSessionInput(this->session_, "prev_states"),
      init_input_shape);

  std::vector<int> pred_in_shape = {beam_size, 1};  // {beam_size, 1}
  this->model_->resizeTensor(
      this->model_->getSessionInput(this->session_, "pred_in"),
      init_input_shape);

  // Resize session with input beam_size.
  this->model_->resizeSession(this->session_);
}

void MnnPredictor::StreamingStep(const std::vector<int>& pred_in) {}

}  // namespace models
}  // namespace s2trt