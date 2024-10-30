# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.09.06
enable_testing()

# predictor lib
add_library(predictor_shared SHARED predictor.cc) # shared
add_library(predictor_static STATIC predictor.cc) # static

target_link_libraries(predictor_shared glog MNN)
target_link_libraries(predictor_static glog MNN)

# Unittest of predictor.
add_executable(predictor-test predictor-test.cc)
target_link_libraries(predictor-test predictor_shared gtest_main)
add_test(
  NAME predictor-test
  COMMAND predictor-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# Joiner lib
add_library(joiner_shared SHARED joiner.cc) # shared
add_library(joiner_static STATIC joiner.cc) # static

target_link_libraries(joiner_shared glog MNN)
target_link_libraries(joiner_static glog MNN)

# Unittest of joiner.
add_executable(joiner-test joiner-test.cc)
target_link_libraries(joiner-test joiner_shared gtest_main)
add_test(
  NAME joiner-test
  COMMAND joiner-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# Encoder lib
set(encoder_src zipformer.cc)

add_library(encoder_shared SHARED ${encoder_src}) # shared
add_library(encoder_static STATIC ${encoder_src}) # static

target_link_libraries(encoder_shared glog MNN)
target_link_libraries(encoder_static glog MNN)

# Unittest of joiner.
add_executable(zipformer-test zipformer-test.cc)
target_link_libraries(zipformer-test encoder_shared frontend_shared gtest_main)
add_test(
  NAME zipformer-test
  COMMAND zipformer-test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
