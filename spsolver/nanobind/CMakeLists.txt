cmake_minimum_required(VERSION 3.18)
project(ce_superlu_nanobind LANGUAGES C CXX)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SUPERLU_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../superlu")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(enable_internal_blaslib ON CACHE BOOL "" FORCE)
set(enable_examples OFF CACHE BOOL "" FORCE)
set(enable_tests OFF CACHE BOOL "" FORCE)
set(enable_fortran OFF CACHE BOOL "" FORCE)
set(enable_doc OFF CACHE BOOL "" FORCE)
set(enable_single OFF CACHE BOOL "" FORCE)
set(enable_double ON CACHE BOOL "" FORCE)
set(enable_complex OFF CACHE BOOL "" FORCE)
set(enable_complex16 ON CACHE BOOL "" FORCE)

add_subdirectory(${SUPERLU_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/superlu_build EXCLUDE_FROM_ALL)

nanobind_add_module(_superlu_nb NOMINSIZE src/superlu_nb.cpp)

target_link_libraries(_superlu_nb PRIVATE superlu)

target_include_directories(_superlu_nb PRIVATE
  ${SUPERLU_SOURCE_DIR}/SRC
  ${CMAKE_CURRENT_BINARY_DIR}/superlu_build/SRC
)

set_target_properties(_superlu_nb PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
