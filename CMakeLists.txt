cmake_minimum_required(VERSION 3.2)
project(xgboost_test)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/main.cpp src/XGBoost.cpp src/XGBoost.h)
include_directories(include)
include_directories(rabit/include/)
link_directories(lib)
add_executable(xgboost_test ${SOURCE_FILES})
target_link_libraries(xgboost_test
        xgboost)
