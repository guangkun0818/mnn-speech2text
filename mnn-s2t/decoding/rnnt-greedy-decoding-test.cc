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
    models::MnnPredictorCfg predictor_cfg;
    predictor_cfg.predictor_model = "../sample_data/models/predictor-int8.mnn";
    predictor_cfg.context_size = 5;
    mnn_predictor_ = std::make_shared<models::MnnPredictor>(
        predictor_cfg, models::CPU_FORWARD_THREAD_8);

    models::MnnJoinerCfg joiner_cfg;
    joiner_cfg.joiner_model = "../sample_data/models/joiner-int8.mnn";
    mnn_joiner_ = std::make_shared<models::MnnJoiner>(
        joiner_cfg, models::CPU_FORWARD_THREAD_8);

    model_sess_ = std::make_shared<models::RnntModelSession>();

    tokenizer_ = std::make_shared<decoding::SubwordTokenizer>(
        "../sample_data/units.txt");

    decoding::DecodingCfg decoding_cfg;
    decoding_cfg.decoding_type = decoding::DecodingType::kRnntGreedyDecoding;
    decoding_cfg.max_token_step = 1;

    greedy_decoding_ = std::make_shared<decoding::RnntGreedyDecoding>(
        mnn_predictor_, mnn_joiner_, model_sess_, tokenizer_, decoding_cfg);
  }

  std::shared_ptr<decoding::DecodingMethod> greedy_decoding_;
  std::shared_ptr<models::MnnPredictor> mnn_predictor_;
  std::shared_ptr<models::MnnJoiner> mnn_joiner_;
  std::shared_ptr<models::RnntModelSession> model_sess_;
  std::shared_ptr<decoding::SubwordTokenizer> tokenizer_;
  size_t max_token_step_;
};

TEST_F(TestRnntGreedyDecoding, TestDecodingInit) {
  greedy_decoding_->Init();
  greedy_decoding_->Reset();
  greedy_decoding_->Init();
  greedy_decoding_->Reset();
  greedy_decoding_->Init();
  greedy_decoding_->Reset();
  greedy_decoding_->Init();
  greedy_decoding_->Reset();
}

TEST_F(TestRnntGreedyDecoding, TestDecodingDecode) {
  greedy_decoding_->Init();
  std::vector<int> enc_out_shape = {1, /*tot_time_step=*/200, /*enc_dim=*/256};
  auto enc_out_data =
      std::vector<float>(1 * 200 * 256, 0.81729);  // Dummy enc_out;
  mnn::Tensor* enc_out = mnn::Tensor::create<float>(
      enc_out_shape, static_cast<void*>(enc_out_data.data()),
      MNN::Tensor::CAFFE);

  greedy_decoding_->Decode(enc_out);

  LOG(INFO) << greedy_decoding_->GetResults();

  mnn::Tensor::destroy(enc_out);
  greedy_decoding_->Reset();
}