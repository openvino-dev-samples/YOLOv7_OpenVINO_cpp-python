cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(TARGET_NAME yolov7)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(${TARGET_NAME} main_preprocessing.cpp)
# uncomment the code below to enable preprocessing api for optimization
# add_executable(${TARGET_NAME} main_preprocessing.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime ${OpenCV_LIBS})