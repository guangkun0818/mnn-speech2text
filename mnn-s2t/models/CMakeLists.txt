# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.09.06
enable_testing()

# Model lib
set(model_src zipformer.cc predictor.cc joiner.cc model-session.cc)
add_library(model_shared SHARED ${model_src}) # shared

target_link_libraries(model_shared glog MNN)

# Unittest of predictor.
add_executable(predictor-test predictor-test.cc)
target_link_libraries(predictor-test model_shared gtest_main)
add_test(
  NAME predictor-test
  COMMAND predictor-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# Unittest of joiner.
add_executable(joiner-test joiner-test.cc)
target_link_libraries(joiner-test model_shared gtest_main)
add_test(
  NAME joiner-test
  COMMAND joiner-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# Unittest of joiner.
add_executable(zipformer-test zipformer-test.cc)
target_link_libraries(zipformer-test model_shared frontend_shared gtest_main)
add_test(
  NAME zipformer-test
  COMMAND zipformer-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
