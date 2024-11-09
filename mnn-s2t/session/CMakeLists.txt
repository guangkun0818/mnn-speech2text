# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.11.02

enable_testing()

# Rnnt session lib
add_library(rnnt_session SHARED rnnt-streaming-session.cc) # shared

target_link_libraries(rnnt_session model_shared frontend_shared decoding_shared)

# Unittest of Rnnt session.
add_executable(rnnt-streaming-session-test rnnt-streaming-session-test.cc)
target_link_libraries(rnnt-streaming-session-test rnnt_session gtest_main)
add_test(
  NAME rnnt-streaming-session-test
  COMMAND rnnt-streaming-session-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
