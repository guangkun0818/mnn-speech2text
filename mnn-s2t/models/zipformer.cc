// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.23
// Wrapped Encoder of transducer of mnn runtime.

#include "mnn-s2t/models/zipformer.h"

namespace s2t {
namespace models {

MnnZipformer::MnnZipformer(const char* zipformer_model, const int feat_dim,
                           const int chunk_size)
    : feat_dim_(feat_dim), chunk_size_(chunk_size) {
  // TODO(guangkun0818): modify to support more config.
  config_.numThread = 8;
  config_.type = MNNForwardType::MNN_FORWARD_CPU;

  this->model_ = std::shared_ptr<mnn::Interpreter>(
      mnn::Interpreter::createFromFile(zipformer_model));
  CHECK_NE(this->model_, nullptr);
  this->session_ = nullptr;
}

MnnZipformer::~MnnZipformer() { this->Reset(); }

const int MnnZipformer::ChunkSize() const { return this->chunk_size_; }

void MnnZipformer::Init(const int num_frames) {
  this->session_ = model_->createSession(config_);
  CHECK_NE(this->session_, nullptr);

  // Input feats shape: {1, num_frames, feat_dim}
  std::vector<int> feat_shape = {1, /*num_frames=*/chunk_size_,
                                 /*feat_dim=*/feat_dim_};
  auto input_feat = this->model_->getSessionInput(this->session_, "x");
  CHECK_EQ(input_feat->shape()[1], chunk_size_);
  CHECK_EQ(input_feat->shape()[2], feat_dim_);

  this->model_->resizeTensor(input_feat, feat_shape);
  this->processed_lens_ = 0;

  // Resize session with input feat shape.
  this->model_->resizeSession(this->session_);
}

void MnnZipformer::Reset() {
  if (this->session_) {
    CHECK(this->model_->releaseSession(this->session_));
    this->session_ = nullptr;
  }
}

void MnnZipformer::StreamingStep(const std::vector<std::vector<float>>& feats) {
  auto input_feat = this->model_->getSessionInput(this->session_, "x");
  CHECK_EQ(input_feat->shape().size(), 3);
  CHECK_EQ(input_feat->shape()[2], feat_dim_);
  CHECK_EQ(input_feat->shape()[1], feats.size());

  for (int frame_id; frame_id < input_feat->shape()[1]; frame_id++) {
    memcpy(input_feat->host<float>() + (frame_id * feat_dim_),
           feats[frame_id].data(), sizeof(float) * feat_dim_);
  }

  this->model_->runSession(this->session_);
  this->UpdateStates();  // Update model states every streaming step.
}

void MnnZipformer::Inference(const std::vector<std::vector<float>>& feats) {
  LOG(WARNING)
      << "Streaming zipformer does not support non-streaming inference.";
}

void MnnZipformer::UpdateStates() {}

mnn::Tensor* MnnZipformer::GetEncOut() {
  return this->model_->getSessionOutput(this->session_, "encoder_out");
}

}  // namespace models
}  // namespace s2t