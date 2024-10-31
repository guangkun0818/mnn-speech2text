// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.10.30
// Speech2text main with transducer system.

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "mnn-s2t/decoding/decoding.h"
#include "mnn-s2t/decoding/tokenizer.h"
#include "mnn-s2t/frontend/frontend-pipeline.h"
#include "mnn-s2t/models/encoder.h"
#include "mnn-s2t/models/joiner.h"
#include "mnn-s2t/models/predictor.h"

int main(int argc, char* argv[]) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "原神";
  std::this_thread::sleep_for(std::chrono::seconds(3));
  LOG(INFO) << "启动!!!!!";

  return 0;
}