
project(Utils)

cmake_minimum_required(VERSION 2.8)

	find_package(Eigen3 REQUIRED)
	MESSAGE(STATUS "Using EIGEN")
        ADD_DEFINITIONS(-DUSE_EIGEN)
        INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

        FIND_PACKAGE(OpenCV 3.1 REQUIRED COMPONENTS opencv_core opencv_highgui opencv_imgproc opencv_ximgproc opencv_imgcodecs opencv_xfeatures2d tracking )
        MESSAGE(STATUS "Using OpenCV 3.1")
        ADD_DEFINITIONS(-DBUILD_TIFF=ON )
        INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})

        INCLUDE_DIRECTORIES(
            include
            ${Logger_SOURCE_DIR}/include
        )
        find_package(OpenMP)
        if(OPENMP_FOUND)
          ADD_DEFINITIONS(-DUSE_OPENMP)
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()

        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -m3dnow -mmmx -msse -msse2 -mfpmath=sse  -Wextra -std=c++11")

        file(GLOB_RECURSE source_files ./*.cc ./*.h)
        ADD_LIBRARY(Utils STATIC ${source_files} ${UTILS_SRC} ${UTILS_INCLUDE})
        TARGET_LINK_LIBRARIES(Utils Logger ${Boost_LIBRARIES} ${OpenCV_LIBS}) #Daisy
	
        INSTALL(TARGETS Utils ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/install)
