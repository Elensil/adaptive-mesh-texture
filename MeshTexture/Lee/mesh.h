#ifndef MESH_H
#define MESH_H


#include "stdio.h"
#include <vector>
#include <map>
#include <list>
#include <iostream>
#include <boost/container/flat_set.hpp>


#include "matrix.h"
#include "Logger.h"
#include "space_time_sampler.h"


class MySpecialMesh{

public:
    MySpecialMesh(long int frame, std::string filename = "");
    MySpecialMesh(const MySpecialMesh &in);
    ~MySpecialMesh();

    /**  Loading */

    bool loadOBJ(const std::string objFile, bool clear = false);

    bool loadMOFF(const std::string moffFile, bool clear = false);

    /**  Cleaning */

    template<class PhotoUtils>
    void cleanAndColor(const PhotoUtils *hyper_volume, int in_faceResParam = 8, int in_downsamplingThreshold = 0, int droppedCamNum = -1);

    template<class PhotoUtils>
    void compressColoredMesh(const PhotoUtils *hyper_volume, int quantFactor, float quantMatCoefs[]);

    /**  Export */

    void exportAsOBJ(std::string filename = "");

    void exportAsCOFF(std::string filename)const;

    void exportAsMOFF(std::string filename)const;	//added by Matt

    void exportAsNMOFF(std::string filename)const;   //added by Matt

    void exportAsMPLY(std::string filename)const;   //added by Matt

    void exportAsFullPLY(std::string filename)const;   //added by Matt

    void exportAsMinPLY(std::string filename) const;

    inline unsigned int getPointSize(){return v_points_.size();}

    /**
     *      Getters
     */
    long int getActiveFrame()const{return active_frame_;}
    std::string getFileName()const{return s_file_name_;}
    void getPointsVector(std::vector<Vector3f> &out_vec)const{out_vec = v_points_;}
    void getFacesVector(std::vector<MyTriangle> &out_vec)const{out_vec = v_faces_;}
    void getTexCoords(std::vector<Vector2f> &out_vec)const{out_vec = tex_coords_;}
    void getTexIndices(std::vector<Vector3uli> &out_vec)const{out_vec = tex_indices_;}
    void getColorsVector(std::vector<Vector3ui> &out_vec)const{out_vec = v_colors_;}
    void getFacesResVector(std::vector<unsigned short> &out_vec)const{out_vec = v_face_res_;}
    void getEdgesIndVector(std::vector<Vector3li> &out_vec)const{out_vec = v_edge_color_ind_;}
    void getFacesIndVector(std::vector<unsigned long> &out_vec)const{out_vec = v_face_color_ind_;}

    // std::vector<MyTriangle> & getRealFacesVector(){return v_faces_;}

    inline void getPoint(int32_t index, Vector3f &out)const{
        if(index >= 0 && index < v_points_.size())
            out = v_points_[index];
        else
            std::cout<<"wrong index, please investigate"<<std::endl;
    }

    /**
     *      Setters
     */
    void setFacesVector(std::vector<MyTriangle> &out_vec){v_faces_ = out_vec;}
    void setPointsVector(std::vector<Vector3f> &out_vec){v_points_ = out_vec;}
    void setColorsVector(std::vector<Vector3ui> &out_vec){v_colors_ = out_vec;}
    void setFacesResVector(std::vector<unsigned short> &out_vec){v_face_res_ = out_vec;}
    void setEdgesIndVector(std::vector<Vector3li> &out_vec){v_edge_color_ind_ = out_vec;}
    void setFacesIndVector(std::vector<unsigned long> &out_vec){v_face_color_ind_ = out_vec;}
    void setColorsBitArray(std::vector<BitArray> &out_vec){v_colors_bitarray = out_vec;}
    
    private:

        long int active_frame_;

        //Mesh components
        std::string s_file_name_;
        std::vector<Vector3f> v_points_;
        std::vector<Vector3ui> v_colors_;
        std::vector<MyTriangle> v_faces_;

        //added by Matt - test
        std::vector<Vector3li> v_edge_color_ind_;
        std::vector<size_t> v_edge_real_color_ind_;   //long story
        std::vector<unsigned short> v_face_res_;
        std::vector<unsigned long> v_face_color_ind_;
        std::vector<Vector2f> tex_coords_;
        std::vector<Vector3uli> tex_indices_;
        std::vector<BitArray> v_colors_bitarray;

        std::vector<int32_t> v_points_separator_;
        std::vector<int32_t> v_faces_separator_;
};

template<class PhotoUtils>
void MySpecialMesh::cleanAndColor(const PhotoUtils *hyper_volume, int in_faceResParam, int in_downsamplingThreshold, int droppedCamNum){
    hyper_volume->colorPointCloud(this, in_faceResParam, in_downsamplingThreshold, droppedCamNum);
}

template<class PhotoUtils>
void MySpecialMesh::compressColoredMesh(const PhotoUtils *hyper_volume, int quantFactor, float quantMatCoefs[]){

    log(ALWAYS)<<"Starting compression."<<endLog();
    
    // hyper_volume->compressColorFull(this, v_edge_real_color_ind_, 32, quantFactor, quantMatCoefs, 0);
    hyper_volume->reIndexColors(this, v_edge_real_color_ind_, 32, quantFactor, quantMatCoefs, 0);
    hyper_volume->compressColor(this, v_edge_real_color_ind_, 32, quantFactor, quantMatCoefs, 0);
    hyper_volume->decodeCompressedColor(this, 32, quantFactor);
    // hyper_volume->reIndexColors(this, v_edge_real_color_ind_, 32, quantFactor, quantMatCoefs, 0);

    
}


#endif //MESH_H
