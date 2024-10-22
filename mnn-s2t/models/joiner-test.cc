// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Unittest of wrapped Joiner.

#include "mnn-s2t/models/joiner.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace s2t;

class TestMnnJoiner : public ::testing::Test {
 protected:
  void SetUp() {
    const char* model = "../sample_data/models/joiner_streaming_step.mnn";
    mnn_joiner_ = std::make_shared<models::MnnJoiner>(model);
  }
  std::shared_ptr<models::MnnJoiner> mnn_joiner_;
};

TEST_F(TestMnnJoiner, TestMnnJoinerStreamingStep) {
  mnn_joiner_->Init(4);
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

  mnn_joiner_->StreamingStep(enc_out, pred_out);
  auto logits = mnn_joiner_->GetJoinerOut();

  mnn::Tensor::destroy(pred_out);
  mnn::Tensor::destroy(enc_out);
}