# 空岛行星生成器的 C++ 核心库（docs/PLAN-CORE.md）：不含 Python、不含 Godot。
#   静态库 skyisle_core ——游戏（Zhouzhu）以子模块只编它；
#   Python 扩展 _core   —— SKYISLE_PYTHON=ON 且找得到 nanobind 时（输出到 ../skyisle_gen/）；
#   自检 core_selftest   —— ctest。
# 一键构建：python core/build.py（Windows 自动进 VS 的 x64 环境）。
cmake_minimum_required(VERSION 3.20)
project(skyisle_core LANGUAGES CXX)

option(SKYISLE_PYTHON "构建 Python 扩展 _core（nanobind）" ON)
option(SKYISLE_TESTS "构建 C++ 自检 core_selftest" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(skyisle_core STATIC
  src/rng.cpp
  src/grid.cpp
  src/flow.cpp
  src/config.cpp
)
target_include_directories(skyisle_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(skyisle_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
find_package(Threads REQUIRED)
target_link_libraries(skyisle_core PUBLIC Threads::Threads)
if(MSVC)
  # 默认 /fp:precise、不开 /arch:AVX2：乘加不合并成 FMA，浮点次序尽量与 numpy 同
  target_compile_options(skyisle_core PUBLIC /utf-8 /EHsc /permissive- /W3 /wd4244 /wd4267)
else()
  target_compile_options(skyisle_core PUBLIC -ffp-contract=off -Wall -Wno-sign-compare)
endif()

if(SKYISLE_TESTS)
  enable_testing()
  add_executable(core_selftest tests/selftest.cpp)
  target_link_libraries(core_selftest PRIVATE skyisle_core)
  add_test(NAME core_selftest COMMAND core_selftest)
endif()

if(SKYISLE_PYTHON)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  if(NOT nanobind_DIR)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
  endif()
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core NB_STATIC python/module.cpp python/bind_island.cpp)
  target_link_libraries(_core PRIVATE skyisle_core)
  set(_pkg ${CMAKE_CURRENT_SOURCE_DIR}/../skyisle_gen)
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${_pkg}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${_pkg}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${_pkg}
    RUNTIME_OUTPUT_DIRECTORY ${_pkg}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${_pkg}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${_pkg})
endif()
