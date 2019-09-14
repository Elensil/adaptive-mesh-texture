project(4DCVT)
cmake_minimum_required(VERSION 2.8.11)


# C++ version flags

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}  -O3 -fopenmp -std=c++11 -std=gnu++11")#-std=gnu++11 to be added in some cases TBD: investigate
SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -m3dnow -mmmx -msse -msse2 -mfpmath=sse  -Wextra -std=c++11") #-std=gnu++11 to be added in some cases TBD: investigate
# Build type
set(CMAKE_BUILD_TYPE Release)

# Add specific FindXXX.cmake modules
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

find_package(Boost COMPONENTS program_options system filesystem regex random iostreams serialization )
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

# Eigen3
find_package(Eigen3 3.1.91 REQUIRED)
        MESSAGE(STATUS "Using EIGEN")
        ADD_DEFINITIONS(-DUSE_EIGEN)
        INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

# OpenMP
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
    message(STATUS "Finding OpenMP -- found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else(OPENMP_FOUND)
    message(STATUS "Finding OpenMP -- not found")
endif(OPENMP_FOUND)


# -------------------------
# THE SOURCES

set( SRC_COMMON
        ./main.cpp
        ./mesh.cpp
     )

set( INCLUDE_COMMON
./mesh.h
)




INCLUDE_DIRECTORIES( .
                     ${Logger_SOURCE_DIR}/include
                     ${Utils_SOURCE_DIR}/Include
)

ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC_COMMON} ${INCLUDE_COMMON})

TARGET_LINK_LIBRARIES(	${PROJECT_NAME}
                        Utils
                        ${Boost_LIBRARIES}
                        ${LBFGS_LIBRARIES}
)


#file(GLOB_RECURSE SRC_LIST ./*.cpp ./*.cxx ./*.h)

#add_executable(${PROJECT_NAME} ${SRC_LIST})
#target_link_libraries(${PROJECT_NAME} ${LBFGS_LIBRARIES})

