// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.23
// Wrapped Encoder of transducer of mnn runtime.

#include "mnn-s2t/models/zipformer.h"

namespace s2t {
namespace models {

MnnZipformer::MnnZipformer(const MnnEncoderCfg& cfg, mnn::ScheduleConfig config)
    : feat_dim_(cfg.feat_dim), chunk_size_(cfg.chunk_size), config_(config) {
  this->model_ = std::shared_ptr<mnn::Interpreter>(
      mnn::Interpreter::createFromFile(cfg.encoder_model.c_str()));
  CHECK_NE(this->model_, nullptr);
}

MnnZipformer::~MnnZipformer() {}

const int MnnZipformer::ChunkSize() const { return this->chunk_size_; }

mnn::Session* MnnZipformer::Init(const int num_frames) {
  auto session = model_->createSession(config_);
  CHECK_NE(session, nullptr);

  // Input feats shape: {1, num_frames, feat_dim}
  std::vector<int> feat_shape = {1, /*num_frames=*/chunk_size_,
                                 /*feat_dim=*/feat_dim_};
  auto input_feat = this->model_->getSessionInput(session, "x");
  CHECK_EQ(input_feat->shape()[1], chunk_size_);
  CHECK_EQ(input_feat->shape()[2], feat_dim_);

  this->model_->resizeTensor(input_feat, feat_shape);
  this->processed_lens_ = 0;

  // Resize session with input feat shape.
  this->model_->resizeSession(session);
  return session;
}

mnn::Session* MnnZipformer::Reset(mnn::Session* session) {
  if (session) {
    CHECK(this->model_->releaseSession(session));
  }
  return nullptr;
}

void MnnZipformer::StreamingStep(const std::vector<std::vector<float>>& feats,
                                 mnn::Session* session) {
  CHECK_NE(session, nullptr);
  auto input_feat = this->model_->getSessionInput(session, "x");
  CHECK_EQ(input_feat->shape().size(), 3);
  CHECK_EQ(input_feat->shape()[2], feat_dim_);
  CHECK_EQ(input_feat->shape()[1], feats.size());

  for (int frame_id; frame_id < input_feat->shape()[1]; frame_id++) {
    memcpy(input_feat->host<float>() + (frame_id * feat_dim_),
           feats[frame_id].data(), sizeof(float) * feat_dim_);
  }

  this->model_->runSession(session);
  this->UpdateStates(session);  // Update model states every streaming step.
}

void MnnZipformer::Inference(const std::vector<std::vector<float>>& feats,
                             mnn::Session* session) {
  LOG(WARNING)
      << "Streaming zipformer does not support non-streaming inference.";
}

void MnnZipformer::UpdateStates(mnn::Session* session) {
  auto inputs = this->model_->getSessionInputAll(session);
  auto outputs = this->model_->getSessionOutputAll(session);
  // Model states are paired with {state, new_state}
  for (auto input : inputs) {
    if (input.first == "x") {
      continue;  // Skip input.
    }
    std::string match_output_name = "new_" + input.first;
    auto found = outputs.find(match_output_name);
    CHECK(found != outputs.end());
    input.second->copyFromHostTensor(found->second);
  }
}

mnn::Tensor* MnnZipformer::GetEncOut(mnn::Session* session) {
  return this->model_->getSessionOutput(session, "encoder_out");
}

}  // namespace models
}  // namespace s2t