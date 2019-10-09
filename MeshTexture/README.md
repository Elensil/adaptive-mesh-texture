## Introduction

Here is the core of the project.

## Documentation

### Prerequisites

- CMake
- OpenCV version 3.1
- Eigen3

Detailed instructions:


	#get and install opencv
	mkdir tempWorkDir
	cd tempWorkDir
	git clone https://github.com/opencv/opencv.git
	cd opencv 
	RUN git checkout 3.3.0
	cd ..
	git clone https://github.com/opencv/opencv_contrib.git
	cd opencv_contrib
	git checkout 3.3.0
	cd ..
	mkdir opencv-build
	cd opencv-build
	cmake -D CMAKE_BUILD_TYPE=DEBUG -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D WITH_TBB=OFF -D WITH_V4L=OFF -D WITH_QT=OFF -D WITH_OPENGL=OFF -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -D BUILD_EXAMPLES=OFF -D WITH_CUDA=OFF -D BUILD_opencv_gpu=OFF -D BUILD_opencv_cvv=OFF -D BUILD_opencv_ovis=OFF -D BUILD_opencv_viz=OFF ../opencv
	make -j
	make install

### Installation


	# Clone and compile
	git clone https://gitlab.inria.fr/marmando/adaptive-mesh-texture.git
	cd adaptive-mesh-texture
	mkdir build
	cd build
	cmake . ../MeshTexture
	make -j




### Execution




#### Parameters settings:



#### Possible modes:
- 'C': Coloring: Takes a .obj mesh, some input views, along with calibration matrices, and generates a textured mesh.
Parameters:
- 'Z': Compression:
- 'X': Extraction:
