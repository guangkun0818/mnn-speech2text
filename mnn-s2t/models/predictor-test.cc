// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.16
// Unittest of Predictor

#include "mnn-s2t/models/predictor.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace s2t::models;

class TestMnnPredictor : public ::testing::Test {
 protected:
  void SetUp() {
    const char* model = "../sample_data/models/predictor_streaming_step.mnn";
    mnn_predictor_ = std::make_shared<MnnPredictor>(model, 5);
  }
  std::shared_ptr<MnnPredictor> mnn_predictor_;
};

TEST_F(TestMnnPredictor, TestPredictorInit) {
  // Unittest of model init/release.
  mnn_predictor_->Init(4);
  mnn_predictor_->Reset();
  
  mnn_predictor_->Init(8);
  mnn_predictor_->Reset();
  
  mnn_predictor_->Init(1);
  mnn_predictor_->Reset();
}

TEST_F(TestMnnPredictor, TestPredictorStreamingStep) {
  mnn_predictor_->Init(4);

  std::vector<int> pred_in = {1, 2, 3, 4};
  mnn_predictor_->StreamingStep(pred_in);
  mnn_predictor_->GetPredOut()->print();
}