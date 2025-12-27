cmake_minimum_required(VERSION 3.20)

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.10
)
FetchContent_MakeAvailable(glfw)


project(AuroraTerrian LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_executable(auroraterrian
  src/main.cpp
  src/build_command.cpp
  src/vk_util.cpp
  src/stb_impl.cpp
  src/export_mesh_command.cpp
)

target_link_libraries(auroraterrian PRIVATE Vulkan::Vulkan)
target_link_libraries(auroraterrian PRIVATE Vulkan::Vulkan glfw)

