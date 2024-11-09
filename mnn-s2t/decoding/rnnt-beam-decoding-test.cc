// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Unittest of Beam decoding of Transducer.

#include "mnn-s2t/decoding/rnnt-beam-decoding.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace s2t;

class TestRnntBeamDecoding : public ::testing::Test {
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
    decoding_cfg.decoding_type = decoding::DecodingType::kRnntBeamDecoding;
    decoding_cfg.cutoff_top_k = 4;  // cutoff_top_k
    decoding_cfg.beam_size = 4;     // beam_size

    beam_decoding_ = std::make_shared<decoding::RnntBeamDecoding>(
        mnn_predictor_, mnn_joiner_, model_sess_, tokenizer_, decoding_cfg);
  }

  std::shared_ptr<decoding::RnntBeamDecoding> beam_decoding_;
  std::shared_ptr<models::MnnPredictor> mnn_predictor_;
  std::shared_ptr<models::MnnJoiner> mnn_joiner_;
  std::shared_ptr<models::RnntModelSession> model_sess_;
  std::shared_ptr<decoding::SubwordTokenizer> tokenizer_;
};

TEST_F(TestRnntBeamDecoding, TestDecodingInit) {
  beam_decoding_->Init();
  beam_decoding_->Reset();
  beam_decoding_->Init();
  beam_decoding_->Reset();
  beam_decoding_->Init();
  beam_decoding_->Reset();
  beam_decoding_->Init();
  beam_decoding_->Reset();
}

TEST_F(TestRnntBeamDecoding, TestDecodingArgSort) {
  std::vector<float> log_probs = {-4.0, -2.0, -1.0, 2.0, -30.0, 40.0};
  std::vector<size_t> token_idxs(log_probs.size());
  beam_decoding_->ArgSort(log_probs, token_idxs);

  ASSERT_EQ(token_idxs[0], 5);  // 40.0
  ASSERT_EQ(token_idxs[1], 3);  // 2.0
  ASSERT_EQ(token_idxs[2], 2);  // -1.0
  ASSERT_EQ(token_idxs[3], 1);  // -2.0
  ASSERT_EQ(token_idxs[4], 0);  // -4.0
  ASSERT_EQ(token_idxs[5], 4);  // -30.0
}

TEST_F(TestRnntBeamDecoding, TestDecodingDecode) {
  beam_decoding_->Init();
  std::vector<int> enc_out_shape = {1, /*tot_time_step=*/200, /*enc_dim=*/256};
  auto enc_out_data =
      std::vector<float>(1 * 200 * 256, 0.81729);  // Dummy enc_out;
  mnn::Tensor* enc_out = mnn::Tensor::create<float>(
      enc_out_shape, reinterpret_cast<void*>(enc_out_data.data()),
      MNN::Tensor::CAFFE);

  beam_decoding_->Decode(enc_out);

  LOG(INFO) << beam_decoding_->GetResults();

  mnn::Tensor::destroy(enc_out);
  beam_decoding_->Reset();
}
