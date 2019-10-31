#ifndef MY_MESH_ENCODER
#define MY_MESH_ENCODER

#include "matrix.h"
#include "optionmanager.h"
#include "bit_array.h"
#include "camera.h"

#include "stdlib.h"
#include <list>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <map>
#include <iostream>


class MeshEncoder{

private:
	std::vector<Vector3f> points;
	std::vector<Vector3ui> colors;
	std::vector<size_t> in_edge_indices;
    std::vector<unsigned short> in_face_res;
    std::vector<Vector3li> in_edge_color_ind;
    std::vector<unsigned long> in_face_color_ind;
    int max_face_res;
    std::vector<BitArray> out_bit_array;
    BitArray* test_bit_array;
    BitArray actual_bit_array;

    const std::vector<unsigned char> m_SOS = {255,218};         //start of scan     				FFDA
    const std::vector<unsigned char> m_SOF0 = {255,192};        //Start of Frame 0, Baseline DCT
    const std::vector<unsigned char> m_DQT = {255,219};         //Define quantization table 		FFDB
    const std::vector<unsigned char> m_DHT = {255,196};         //Define Huffman table 				FFC4
    const std::vector<unsigned char> m_EOI = {255,217};         //End of Image 						FFD9
    const std::vector<unsigned char> m_SOI = {255,216};         //Start of Image    				FFD8
    const std::vector<unsigned char> m_APP4 = {255,228};         //Application segment 4. 			FFE4 	Used for resolution change for now
    const std::vector<unsigned char> m_APP5 = {255,229};		//Application segment 5. 			FFE5	Used for PCA components(?)
    


public:
	MeshEncoder(){
	};

	void writeJPEGMeshColor(std::map<int,cv::Mat> &dctTrianglesList, std::map<int,std::vector<float>> &quantizationTables, std::map<int,cv::Mat> &eigenVectors, std::map<int,int> &quantMultipliers, std::string filePath, std::string fileName);

    // void writeJPEGMeshColor(std::map<int,std::vector<cv::Mat>> &dctTrianglesList, std::string outPath)const;

    std::vector<BitArray> getMeshBinaryColor(std::map<int,std::vector<cv::Mat>> &dctTrianglesList,int trianglesNumber)const;

    std::vector<BitArray> getMeshBinaryColor(std::map<int,cv::Mat> &dctTrianglesList, int trianglesNumber)const;

    void write2BytesNum(std::vector<char> &byteArray, unsigned short myNum)const;

    void addDCCoef(int myCoef, HuffTree &myHT)const;

    void writeCodedDCCoef(BitArray &myBitArray, int myCoef, HuffTree &myHT)const;

    void addACCoef(int myCoef, int trailingZeros, HuffTree &myHT)const;

    void writeCodedACCoef(BitArray &myBitArray, int myCoef, int trailingZeros, HuffTree &myHT)const;

    /**
    * Returns the triangular DCT matrix for the specified triangle resolution (including edges and vertices),
    * as specified in the paper 'Two-dimensional orthogonal DCT expansion in trapezoid and triangular blocks and modified JPEG image compression' by J. Ding
    * @param N triangle length (Res-2 or Res+1 depending on whether you include edges/vertices or not)
    * @return orthonormal DCT matrix
    */
    cv::Mat getTriangularDCTMatrix(const int N, std::vector<int> &quantizationMat)const;

    //inline std::vector<MyTriangle> getTriangles()const{return faces;}

    inline std::vector<Vector3f> getPoints()const{return points;}

    std::map<int,cv::Mat> decodeCompressedData(std::map<int,cv::Mat> &resPCAEigenVectors, std::string filePath);

    unsigned char readCodedByte(BitArray &bitStream, HuffTree &dcTree);

    int readDCCoef(BitArray &bitStream, HuffTree &dcTree, int previousDCValue);

    int readACCoef(BitArray &bitStream, HuffTree &acTree, int &out_trailingZeros);

    HuffTree readHuffTree(BitArray &bitStream);

    void writeHuffTree(BitArray &myBitArray, HuffTree &myHT)const;

    void encodeQuantizationMatrix(BitArray &bitStream, std::vector<float> quantMat, int qtNumber);

    void encodeEigenVectors(BitArray &bitStream, cv::Mat &eigenVectors, int vectorsNumber, int qtNumber);

    //returns QT number, fills matrix with quantization parameters
    int readQuantizationMatrix(BitArray &bitStream, std::vector<float> &out_QT);

    //returns decomposition number, fills matrix with pca vectors
    int readPCAEigenVectors(BitArray &bitStream, cv::Mat &eigenVectors);

};

class ColoredEdge{

public:
	ColoredEdge(){
	}

private:
	unsigned long long int vert1;
	unsigned long long int vert2;
	unsigned short edgeRes;
	std::vector<Vector3ui> colors;
};

class ColoredTriangle{

private:
	unsigned long long int vert1;
    unsigned long long int vert2;
    unsigned long long int vert3;
    unsigned short faceRes;
    ColoredEdge* edge1;
    ColoredEdge* edge2;
    ColoredEdge* edge3;
    bool edge1Oriented;
    bool edge2Oriented;
    bool edge3Oriented;
    std::vector<Vector3ui> colors;

public:

	ColoredTriangle(){
	}

	inline long long int getV1()const{return vert1;}

	inline long long int getV2()const{return vert2;}

	inline long long int getV3()const{return vert3;}	

	inline unsigned short getFaceRes()const{return faceRes;}

	//return color sample at the right position. Can be interpolated if edge sample.
	//bo, b1 in [0, faceRes]
	Vector3ui getColor(int b0, int b1)const;

	//return color value at exact fragment. Interpolate between 3 neighboring samples
	//b0, b1 in [0,1]
	Vector3ui getColor(float b0, float b1)const;

};

#endif	// MY_MESH_ENCODER
