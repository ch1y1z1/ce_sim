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

option(
  CE_SUPERLU_USE_INTERNAL_BLAS
  "Build and use SuperLU internal CBLAS instead of system BLAS"
  OFF
)
set(
  CE_SUPERLU_BLA_VENDOR
  "OpenBLAS"
  CACHE STRING
  "Preferred BLAS vendor passed to FindBLAS (e.g. OpenBLAS, Generic, Intel10_64lp)"
)
set(
  CE_SUPERLU_BLAS_LIBRARIES
  ""
  CACHE STRING
  "Optional semicolon-separated BLAS library list (overrides FindBLAS), e.g. /usr/lib/libopenblas.so"
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(enable_examples OFF CACHE BOOL "" FORCE)
set(enable_tests OFF CACHE BOOL "" FORCE)
set(enable_fortran OFF CACHE BOOL "" FORCE)
set(enable_doc OFF CACHE BOOL "" FORCE)
set(enable_single OFF CACHE BOOL "" FORCE)
set(enable_double ON CACHE BOOL "" FORCE)
set(enable_complex OFF CACHE BOOL "" FORCE)
set(enable_complex16 ON CACHE BOOL "" FORCE)

if (CE_SUPERLU_USE_INTERNAL_BLAS)
  set(enable_internal_blaslib ON CACHE BOOL "" FORCE)
  set(TPL_ENABLE_INTERNAL_BLASLIB ON CACHE BOOL "" FORCE)
  message(STATUS "ce_superlu_nanobind: using SuperLU internal CBLAS")
else()
  set(enable_internal_blaslib OFF CACHE BOOL "" FORCE)
  set(TPL_ENABLE_INTERNAL_BLASLIB OFF CACHE BOOL "" FORCE)

  if (CE_SUPERLU_BLAS_LIBRARIES)
    set(TPL_BLAS_LIBRARIES "${CE_SUPERLU_BLAS_LIBRARIES}" CACHE STRING "" FORCE)
    message(STATUS "ce_superlu_nanobind: using user-specified BLAS libs: ${TPL_BLAS_LIBRARIES}")
  else()
    set(BLA_VENDOR "${CE_SUPERLU_BLA_VENDOR}" CACHE STRING "" FORCE)
    find_package(BLAS REQUIRED)
    set(TPL_BLAS_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "" FORCE)
    message(STATUS "ce_superlu_nanobind: using system BLAS (${BLA_VENDOR}): ${TPL_BLAS_LIBRARIES}")
  endif()
endif()

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
