# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.09.06

enable_testing()

set(frontend_src
    kaldi-fbank-feature.cc
    feature-function.cc
    feature-window.cc
    fftsg.cc
    kaldi-math.cc
    mel-computations.cc
    frontend-pipeline.cc
    rfft.cc)

# frontend lib
add_library(frontend_shared SHARED ${frontend_src}) # shared

target_link_libraries(frontend_shared glog)

# Unittest of kaldi-fbank-feature.
add_executable(kaldi-fbank-test kaldi-fbank-feature-test.cc)
target_link_libraries(kaldi-fbank-test frontend_shared gtest_main)
add_test(
  NAME kaldi-fbank-test
  COMMAND kaldi-fbank-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# Unittest of vad_session
add_executable(frontend-pipeline-test frontend-pipeline-test.cc)
target_link_libraries(frontend-pipeline-test frontend_shared gtest_main)
add_test(
  NAME frontend-pipeline-test
  COMMAND frontend-pipeline-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
