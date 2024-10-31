# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.08.17

cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(
  mnn-s2trt
  LANGUAGES CXX C
  VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)

include(CTest)

include(FetchContent)
set(FETCHCONTENT_QUIET off)
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/3rd_party")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(mnn)
include(gflags)
include(glog)
include(googletest)

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(mnn-s2t)
