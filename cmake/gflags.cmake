# Author: guangkun0818 Email: 609946862@qq.com Created on 2024.10.31

FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG v2.2.1)
FetchContent_MakeAvailable(gflags)

include_directories(${glog_BINARY_BIN}/include)
