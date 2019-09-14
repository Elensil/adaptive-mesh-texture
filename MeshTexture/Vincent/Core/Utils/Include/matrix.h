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

#define EROSION_SIZE 15 //8 ; 20 doorway
#define CAMERA_REMOVAL_SIZE 2.0


#define IMG_WIDTH 2048
#define IMG_HEIGHT 2048
#define CLEANING_FACTOR 3.0


#endif // MATRIX_H
