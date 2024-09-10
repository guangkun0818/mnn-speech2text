# Author: guangkun0818 Email: 609946862@qq.com Created on 2024.08.28

# Mnn runtime.
FetchContent_Declare(
  mnn
  GIT_REPOSITORY https://github.com/alibaba/MNN.git
  GIT_TAG 2.9.0)
FetchContent_MakeAvailable(mnn)

include_directories(${mnn_SOURCE_DIR}/include ${mnn_SOURCE_DIR}/source)
