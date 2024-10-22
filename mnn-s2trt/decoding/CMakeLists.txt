# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.09.06
enable_testing()

# predictor lib
add_library(decoding_shared SHARED rnnt-greedy-decoding.cc tokenizer.cc
)# shared
add_library(decoding_static STATIC rnnt-greedy-decoding.cc tokenizer.cc
)# static

target_link_libraries(decoding_shared glog MNN predictor_shared joiner_shared)
target_link_libraries(decoding_static glog MNN predictor_static joiner_static)

# Unittest of tokenizer
add_executable(tokenizer-test tokenizer-test.cc)
target_link_libraries(tokenizer-test decoding_shared gtest_main)
add_test(
  NAME tokenizer-test
  COMMAND tokenizer-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..)

# Unittest of rnnt-greedy-decoding.
add_executable(rnnt-greedy-decoding-test rnnt-greedy-decoding-test.cc)
target_link_libraries(rnnt-greedy-decoding-test decoding_shared gtest_main)
add_test(
  NAME rnnt-greedy-decoding-test
  COMMAND tokenizer-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..)
