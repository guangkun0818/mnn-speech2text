// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Unittest of wrapped Joiner.

#include "mnn-s2t/models/joiner.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mnn-s2t/models/model-session.h"

using namespace s2t;

class TestMnnJoiner : public ::testing::Test {
 protected:
  void SetUp() {
    const char* model = "../sample_data/models/joiner_streaming_step.mnn";
    mnn_joiner_ = std::make_shared<models::MnnJoiner>(
        model, models::CPU_FORWARD_THREAD_8);
    model_sess_ = std::make_shared<models::RnntModelSession>();
  }
  std::shared_ptr<models::MnnJoiner> mnn_joiner_;
  std::shared_ptr<models::RnntModelSession> model_sess_;
};

TEST_F(TestMnnJoiner, TestMnnJoinerSInitRelease) {
  // Unittest of model init/release.
  model_sess_->joiner_session = mnn_joiner_->Init(4);
  mnn_joiner_->Reset(model_sess_->joiner_session);

  model_sess_->joiner_session = mnn_joiner_->Init(1);
  mnn_joiner_->Reset(model_sess_->joiner_session);

  model_sess_->joiner_session = mnn_joiner_->Init(8);
  mnn_joiner_->Reset(model_sess_->joiner_session);
}

TEST_F(TestMnnJoiner, TestMnnJoinerStreamingStep) {
  mnn_joiner_->Reset(model_sess_->joiner_session);
  model_sess_->joiner_session = mnn_joiner_->Init(4);
  std::vector<int> pred_out_shape = {4, 1, 256};
  auto pred_out_data =
      std::vector<int>(4 * 1 * 256, 0.13471);  // Dummy pred_out;

  std::vector<int> enc_out_shape = {1, 1, 256};
  auto enc_out_data = std::vector<int>(1 * 1 * 256, 0.81729);  // Dummy enc_out;

  mnn::Tensor* pred_out = mnn::Tensor::create<float>(
      pred_out_shape, static_cast<void*>(pred_out_data.data()),
      MNN::Tensor::CAFFE);
  mnn::Tensor* enc_out = mnn::Tensor::create<float>(
      enc_out_shape, static_cast<void*>(enc_out_data.data()),
      MNN::Tensor::CAFFE);

  mnn_joiner_->StreamingStep(enc_out, pred_out, model_sess_->joiner_session);
  auto logits = mnn_joiner_->GetJoinerOut(model_sess_->joiner_session);

  mnn::Tensor::destroy(pred_out);
  mnn::Tensor::destroy(enc_out);
}