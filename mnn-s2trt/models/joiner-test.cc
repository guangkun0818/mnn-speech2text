// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.17
// Unittest of wrapped Joiner.

#include "mnn-s2trt/models/joiner.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace s2trt;

class TestMnnJoiner : public ::testing::Test {
 protected:
  void SetUp() {
    const char* model = "sample_data/models/joiner_streaming_step.mnn";
    mnn_joiner_ = std::make_shared<models::MnnJoiner>(model);
  }
  std::shared_ptr<models::MnnJoiner> mnn_joiner_;
};

TEST_F(TestMnnJoiner, TestMnnJoinerStreamingStep) { mnn_joiner_->Init(4); }