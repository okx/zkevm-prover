cmake_minimum_required(VERSION 3.10)
project(test_aggregator_mock)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加源代码根目录到包含路径
get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)
message(STATUS "Project root: ${PROJECT_ROOT}")
message(STATUS "Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")

# 查找必要的包
find_package(GTest REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
find_package(nlohmann_json REQUIRED)

# MacOS 特定設置
if(APPLE)
    find_library(UUID_LIBRARY System) # MacOS 上使用系統的 UUID 庫
else()
    find_library(UUID_LIBRARY uuid)   # 其他系統上查找 libuuid
endif()

if(NOT UUID_LIBRARY)
    message(FATAL_ERROR "UUID library not found")
endif()

# 源文件
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_aggregator_mock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aggregator_client_mock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aggregator_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aggregator_service.cpp
    ${PROJECT_ROOT}/src/grpc/gen/aggregator.grpc.pb.cc
    ${PROJECT_ROOT}/src/grpc/gen/aggregator.pb.cc
)

message(STATUS "Source files:")
foreach(SOURCE ${SOURCES})
    message(STATUS "  ${SOURCE}")
endforeach()

# 添加可执行文件
add_executable(test_aggregator_mock ${SOURCES})

# 包含目录
target_include_directories(test_aggregator_mock PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_ROOT}/src/grpc/gen
    ${PROJECT_ROOT}/src/config 
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(test_aggregator_mock PRIVATE
    GTest::GTest
    GTest::Main
    gRPC::grpc++
    gRPC::grpc++_reflection
    protobuf::libprotobuf
    nlohmann_json::nlohmann_json
    pthread
    ${UUID_LIBRARY}
)

# MacOS 特定链接
if(APPLE)
    target_link_libraries(test_aggregator_mock PRIVATE "-framework CoreFoundation")
endif()

# 打印一些调试信息
message(STATUS "Include directories:")
get_target_property(INCLUDES test_aggregator_mock INCLUDE_DIRECTORIES)
foreach(DIR ${INCLUDES})
    message(STATUS "  ${DIR}")
endforeach()