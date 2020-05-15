#ifndef MATRIX_H
#define MATRIX_H

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseCore>
#include <vector>


typedef Eigen::Matrix<float, 2, 1> Vector2f;
typedef Eigen::Matrix<double, 2, 1> Vector2d;
typedef Eigen::Matrix<unsigned long int, 2, 1> Vector2uli;
typedef Eigen::Matrix<long int, 2, 1> Vector2li;
typedef Eigen::Matrix<float, 3, 1> Vector3f;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<unsigned long int, 3, 1> Vector3uli;
typedef Eigen::Matrix<unsigned short int, 3, 1> Vector3ui;
typedef Eigen::Matrix<long int, 3, 1> Vector3li;
typedef Eigen::Matrix<unsigned char, 3, 1> Vector3uchar;
typedef Eigen::Matrix<char, 3, 1> Vector3char;
typedef Eigen::Matrix<float, 4, 1> Vector4f;
typedef Eigen::Matrix<float, 3, 3> Matrix3f;
typedef Eigen::Matrix<float, 4, 4> Matrix4f;
typedef Eigen::Matrix<float, 3, 4> Matrix34f;
typedef Eigen::Matrix<float, 4, 3> Matrix43f;

typedef struct MyTriangle{
    unsigned long long int ref;
    unsigned long long int edge1;
    unsigned long long int edge2;
}MyTriangle;

/// TBD: Remove the GLM dependency (used to compute the back-projection matrix in Camera)


#include "glm/glm.hpp"

typedef glm::dmat3x4 GLM_Mat3x4;
typedef glm::dmat4x3 GLM_Mat4x3;
typedef glm::dmat4x4 GLM_Mat4;
typedef glm::dmat3x3 GLM_Mat3;
typedef glm::dvec3   GLM_Vec3;
typedef glm::dvec4   GLM_Vec4;


#define IMG_WIDTH 2048
#define IMG_HEIGHT 2048
#define CLEANING_FACTOR 2.0

// --- Coloring parameters ---

#define OUT_OF_CAMERA_NUMBER 1 					// First, select this number of camera for each vertex. Then, remove less consensual camera one by one, until you're left with CAMERA_NUMBER cameras.
#define CAMERA_NUMBER 1 						// number of cameras voting for each vertex
#define VOTE_PIXEL_RADIUS 0
#define PROJ_MARGIN_RADIUS 10 					// Margin around sudden depth change in images, while reprojecting, to avoid visibility errors due to imprecise geometry/calibration
#define PROJ_MARGIN_DEPTH_TH 0.03;

// You might want to tweak these:
#define MAX_FACE_RES 16 						// Maximum face resolution.
#define DOWNSAMPLE_THRESHOLD 30 				// Parameter th in the paper
#define FACE_RES_RATIO 3 						// Parameter ? in the paper

// --- Compression parameters ---

#define QUANT_BITS 16 							// Number of bits used to encode the coefficients of the eigen vectors in the PCA decomposition. Can be tuned down, but for a limited gain, and it can become a limiting quality factor...
												// It is advised to leave it at 16. Bigger than 16 is not supported.

#define QUANT_FACTOR 4096 						// Quantization factor. Main parameter to tweak, for varying compression ratio.
#define QUANT_MAT_L1 1.0 						// Parameters for building quantization matrix
# define QUANT_MAT_L2 0.01 						// For each coefficient i (from 0 to space_dim), quant(i) = QUANT_FACTOR * (1 + QUANT_MAT_L1 * i + QUANT_MAT_L2 * i*i)
												// Not very useful to change.
												// WARNING!!! For the time being, these 3 parameters (QUNAT_FACTOR, QUANT_MAT_L1, QUANT_MAT_L2) are NOT encoded with the data. They must be known before decoding.
												// (Contrary to QUANT_BITS, which is read from the file).

#endif // MATRIX_H
