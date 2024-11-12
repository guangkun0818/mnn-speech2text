# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.09.06
enable_testing()

# decoding lib
add_library(
  decoding_shared SHARED rnnt-beam-decoding.cc rnnt-greedy-decoding.cc
                         tokenizer.cc) # shared

target_link_libraries(decoding_shared glog MNN model_shared)

# Unittest of tokenizer
add_executable(tokenizer-test tokenizer-test.cc)
target_link_libraries(tokenizer-test decoding_shared gtest_main)
add_test(
  NAME tokenizer-test
  COMMAND tokenizer-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# Unittest of rnnt-greedy-decoding.
add_executable(rnnt-greedy-decoding-test rnnt-greedy-decoding-test.cc)
target_link_libraries(rnnt-greedy-decoding-test decoding_shared gtest_main)
add_test(
  NAME rnnt-greedy-decoding-test
  COMMAND rnnt-greedy-decoding-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# Unittest of rnnt-beam-decoding.
add_executable(rnnt-beam-decoding-test rnnt-beam-decoding-test.cc)
target_link_libraries(rnnt-beam-decoding-test decoding_shared gtest_main)
add_test(
  NAME rnnt-beam-decoding-test
  COMMAND rnnt-beam-decoding-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
