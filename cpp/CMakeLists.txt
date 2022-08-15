cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(TARGET_NAME yolov7)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(${TARGET_NAME} main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime ${OpenCV_LIBS})