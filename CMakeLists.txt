cmake_minimum_required(VERSION 3.20)
project(AuroraTerrian LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_executable(auroraterrian
    src/main.cpp
)

target_link_libraries(auroraterrian PRIVATE Vulkan::Vulkan)