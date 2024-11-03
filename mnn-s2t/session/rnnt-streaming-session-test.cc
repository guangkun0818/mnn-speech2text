// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.23
// Unittest of transducer Session.

#include "mnn-s2t/session/rnnt-streaming-session.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mnn-s2t/frontend/wav.h"

using namespace s2t;

class TestRnntStreamingSession : public ::testing::Test {
 protected:
  void SetUp() {}

  std::shared_ptr<session::RnntStreamingSession> session_;
};