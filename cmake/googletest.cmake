# Author: guangkun0818 Email: 609946862@qq.com Created on 2024.08.28

# GoogleTest v1.13.0
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.0)
FetchContent_MakeAvailable(googletest)

include_directories(
  ${googletest_SOURCE_DIR}/googletest/include
  ${googletest_SOURCE_DIR}/googlemock/include ${googletest_BINARY_BIN})
