cmake_minimum_required(VERSION 3.14)
project(neural_network_cpp VERSION 1.0.0 LANGUAGES CXX
        DESCRIPTION "Dependency-free feedforward neural network in modern C++17")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Examples and tests are built by default only when this is the top-level project,
# so consuming the library through add_subdirectory / FetchContent stays lightweight.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(NN_IS_TOP_LEVEL ON)
else()
  set(NN_IS_TOP_LEVEL OFF)
endif()

option(NN_BUILD_EXAMPLES "Build the example programs" ${NN_IS_TOP_LEVEL})
option(NN_BUILD_TESTS "Build the test suite" ${NN_IS_TOP_LEVEL})

# Header-only library target: link with  target_link_libraries(app PRIVATE nn::nn)
add_library(nn INTERFACE)
add_library(nn::nn ALIAS nn)
target_include_directories(nn INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(nn INTERFACE cxx_std_17)

function(nn_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

if(NN_BUILD_EXAMPLES)
  foreach(example binary_classification spiral_multiclass mnist_example)
    add_executable(${example} examples/${example}.cpp)
    target_link_libraries(${example} PRIVATE nn::nn)
    nn_enable_warnings(${example})
  endforeach()
endif()

if(NN_BUILD_TESTS)
  enable_testing()
  add_executable(nn_tests tests/tests.cpp)
  target_link_libraries(nn_tests PRIVATE nn::nn)
  nn_enable_warnings(nn_tests)
  add_test(NAME nn_tests COMMAND nn_tests)
endif()

install(DIRECTORY include/ DESTINATION include)
