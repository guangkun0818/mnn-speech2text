// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.16
// Unittest of Predictor

#include "mnn-s2t/models/predictor.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mnn-s2t/models/model-session.h"

using namespace s2t::models;

class TestMnnPredictor : public ::testing::Test {
 protected:
  void SetUp() {
    MnnPredictorCfg cfg;
    cfg.predictor_model = "../sample_data/models/predictor-int8.mnn";
    cfg.context_size = 5;

    mnn_predictor_ = std::make_shared<MnnPredictor>(cfg, CPU_FORWARD_THREAD_8);
    model_sess_ = std::make_shared<RnntModelSession>();
  }
  std::shared_ptr<MnnPredictor> mnn_predictor_;
  std::shared_ptr<RnntModelSession> model_sess_;
};

TEST_F(TestMnnPredictor, TestPredictorInit) {
  // Unittest of model init/release.
  model_sess_->predictor_session = mnn_predictor_->Init(4);
  model_sess_->predictor_session =
      mnn_predictor_->Reset(model_sess_->predictor_session);
  ASSERT_EQ(model_sess_->predictor_session, nullptr);

  model_sess_->predictor_session = mnn_predictor_->Init(8);
  model_sess_->predictor_session =
      mnn_predictor_->Reset(model_sess_->predictor_session);
  ASSERT_EQ(model_sess_->predictor_session, nullptr);

  model_sess_->predictor_session = mnn_predictor_->Init(1);
  model_sess_->predictor_session =
      mnn_predictor_->Reset(model_sess_->predictor_session);
  ASSERT_EQ(model_sess_->predictor_session, nullptr);
}

TEST_F(TestMnnPredictor, TestPredictorStreamingStep) {
  model_sess_->predictor_session =
      mnn_predictor_->Reset(model_sess_->predictor_session);
  model_sess_->predictor_session = mnn_predictor_->Init(4);

  std::vector<int> pred_in = {1, 2, 3, 4};
  mnn_predictor_->StreamingStep(pred_in, model_sess_->predictor_session);
  mnn_predictor_->GetPredOut(model_sess_->predictor_session)->print();
}