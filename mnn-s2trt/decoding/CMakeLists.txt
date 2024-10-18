# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.09.06
enable_testing()

# predictor lib
add_library(decoding_shared SHARED rnnt-greedy-decoding.cc tokenizer.cc
)# shared
add_library(decoding_static STATIC rnnt-greedy-decoding.cc tokenizer.cc
)# static

target_link_libraries(decoding_shared glog MNN)
target_link_libraries(decoding_static glog MNN)

# Unittest of kaldi-fbank-feature.
add_executable(tokenizer-test tokenizer-test.cc)
target_link_libraries(tokenizer-test decoding_shared gtest_main)
add_test(
  NAME tokenizer-test
  COMMAND tokenizer-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..)
