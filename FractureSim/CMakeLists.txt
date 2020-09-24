
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(FractureSim CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
set(CMAKE_CXX_FLAGS "-Wall")

# libigl
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          ON)
option(LIBIGL_WITH_PNG               "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML               "Use XML"            OFF)
option(USE_CUDA                      "Enable CUDA support" OFF)

if (NOT LIBIGL_FOUND)
    find_package(LIBIGL REQUIRED QUIET)
endif()

# Add default project files
file(GLOB LIBFILES ${PROJECT_SOURCE_DIR}/../include/*.*)
source_group("Library Files" FILES ${LIBFILES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

if(USE_CUDA)
	message(STATUS "CUDA SUPPORT ENABLED")
	add_library(cudaSupport STATIC
		CudaSupport.cu
		CudaSupport.h)
	set_target_properties(cudaSupport PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_property(TARGET cudaSupport PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	target_compile_features(cudaSupport PUBLIC cxx_std_11)
endif(USE_CUDA)

add_definitions(-DIGL_VIEWER_VIEWER_QUIET)
add_executable(${PROJECT_NAME} 
	main.cpp
	FractureSim.cpp
	helpers.cpp
	FractureSim.h
	helpers.h
	KDTreeVectorOfVectorsAdaptor.h
	nanoflann.hpp
	utils.h
	parameters.h
	${LIBFILES} )


if(USE_CUDA)
	target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif(USE_CUDA)

target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw igl::opengl_glfw_imgui)

if(USE_CUDA)
	target_link_libraries(${PROJECT_NAME} cudaSupport)
endif(USE_CUDA)	
