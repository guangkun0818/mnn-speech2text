// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Unittest of Greedy decoding of Transducer.

#include "mnn-s2t/decoding/rnnt-greedy-decoding.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace s2t;

class TestRnntGreedyDecoding : public ::testing::Test {
 protected:
  void SetUp() {
    const char* predictor =
        "../sample_data/models/predictor_streaming_step.mnn";
    mnn_predictor_ = std::make_shared<models::MnnPredictor>(predictor, 5);

    const char* joiner = "../sample_data/models/joiner_streaming_step.mnn";
    mnn_joiner_ = std::make_shared<models::MnnJoiner>(joiner);

    tokenizer_ = std::make_shared<decoding::SubwordTokenizer>(
        "../sample_data/units.txt");
    max_token_step_ = 1;

    greedy_decoding_ = std::make_shared<decoding::RnntGreedyDecoding>(
        mnn_predictor_, mnn_joiner_, tokenizer_, max_token_step_);
  }

  std::shared_ptr<decoding::RnntGreedyDecoding> greedy_decoding_;
  std::shared_ptr<models::MnnPredictor> mnn_predictor_;
  std::shared_ptr<models::MnnJoiner> mnn_joiner_;
  std::shared_ptr<decoding::SubwordTokenizer> tokenizer_;
  size_t max_token_step_;
};

TEST_F(TestRnntGreedyDecoding, TestDecodingInit) { greedy_decoding_->Init(); }

TEST_F(TestRnntGreedyDecoding, TestDecodingDecode) {
  greedy_decoding_->Init();
  std::vector<int> enc_out_shape = {1, /*tot_time_step=*/200, /*enc_dim=*/256};
  auto enc_out_data =
      std::vector<float>(1 * 200 * 256, 0.81729);  // Dummy enc_out;
  mnn::Tensor* enc_out = mnn::Tensor::create<float>(
      enc_out_shape, static_cast<void*>(enc_out_data.data()),
      MNN::Tensor::CAFFE);

  auto decoded = greedy_decoding_->Decode(enc_out);

  LOG(INFO) << decoded;

  mnn::Tensor::destroy(enc_out);
}