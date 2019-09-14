#ifndef CAMERA_H
#define CAMERA_H

#include "matrix.h"
#include "Logger.h"

#include "stdio.h"
#include <iostream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/ximgproc.hpp>



class Camera
{

public:
    Camera(unsigned int _u8_camera_id_ , std::string _s_images_sequence_ = "");
    ~Camera();

    inline bool isProjectionMatrixDefined() const {return b_is_projection_matrix_defined_;}

    inline bool isOrientationDefined() const {return b_is_orientation_defined_;}

    inline bool isPositionDefined() const { return b_is_position_defined_;}

    inline bool isImageSequenceDefined() const {return !s_images_sequence_.empty();}

    boost::optional<bool> isInsideSilhouette(const int &y, const int &x) const;

    template<class InVec2>
    inline boost::optional<bool> isInsideSilhouette(const InVec2 &in) const{return isInsideSilhouette(in(0),in(1));}

    boost::optional<bool> isInsideShrunkSilhouette(const int &y, const int &x) const;

    template<class InVec2>
    inline boost::optional<bool> isInsideShrunkSilhouette(const InVec2 &in) const{return isInsideShrunkSilhouette(in(0),in(1));}

    ///Setters
    inline void setIndex(unsigned int index){u8_camera_index_ = index;}

    template<class InPath> inline void setSilhouettesSequence(const InPath &s_silhouettes_sequence){s_silhouettes_sequence_ = s_silhouettes_sequence;}

    template<class InPath>  void setProjectionMatrix(const InPath &projection_matrix_file){loadMatFile(getProjectionMatrixFile(projection_matrix_file));}

    void loadFrame(const long int frame);

    void setPositionFromProjectionMatrix();

    void setInverseProjectionMatrixFromProjectionMatrix();

    ///Getters
    inline const unsigned int& getID()const{return u8_camera_id_;}

    inline const unsigned int& getIndex()const{return u8_camera_index_;}

    inline unsigned int getWidth()const{return loaded_image_.size().width;}

    inline unsigned int getHeight()const{return loaded_image_.size().height;}

    inline const Vector3f &getPosition()const{return vec3_position_;}

    template <class InFrameNumber>
    inline std::string getImageFile(const InFrameNumber &frame_number) const {return (s_images_sequence_.empty())?"":(boost::format(s_images_sequence_) % u8_camera_id_ % frame_number).str();}

    template <class InFrameNumber>
    inline std::string getSilhouetteFile(const InFrameNumber &frame_number) const {return (s_silhouettes_sequence_.empty())?"":(boost::format(s_silhouettes_sequence_) % u8_camera_id_ % frame_number).str();}

    template <class InPath>
    inline InPath getProjectionMatrixFile(const InPath &path) const {return (path.empty())?"":(boost::format(path) % u8_camera_id_ ).str();}

    template <class InVec3, class OutVec2>
    inline void getTextureCoords(const InVec3 &pos, OutVec2 &out) const { // OutVec2 -> 0:rows, 1:cols
        Vector4f v4f_pos;
        v4f_pos << pos,1.0;
        InVec3 tCoord3 = mat34_projection_matrix_ * v4f_pos;
        tCoord3(0) /=  tCoord3(2);
        tCoord3(1) /=  tCoord3(2);
        out(1) = tCoord3(0)  ; //Invert row/col with respect to opencv convention
        out(0) = tCoord3(1)  ;
    }

    template<class InVec2, class OutColor>
    inline bool getPixelColor(const InVec2 &tex_coords, OutColor &out_color)const { //row first : (y,x)
        bool out = true;
        if(tex_coords(0) > 0 && tex_coords(0) < loaded_image_.size().height && tex_coords(1) > 0.0 && tex_coords(1) < loaded_image_.size().width  )
            out_color = loaded_image_.at<OutColor>(tex_coords(0),tex_coords(1));
        else
            out = false;
        return out;
    }

    template<class OutColor>
    inline bool getPixelColor(const Vector2f &tex_coords, OutColor &out_color)const { //row first : (y,x)
        bool out = true;

        Vector2uli intTexCoords;
        intTexCoords(0) = (int)(tex_coords(0)+0.5);
        intTexCoords(1) = (int)(tex_coords(1)+0.5);
        return getPixelColor(intTexCoords, out_color);
    }

    template<class OutputArray>
    inline int getLabels(OutputArray &out_labels)const{
        int out = 0;
        if(is_slic_defined_){
            slic_clustering_->getLabels(out_labels);
            out = slic_clustering_->getNumberOfSuperpixels();
        }
        return out;
    }

    template<class InVec2, class Depth, class OutVec3> //InVec2 as (y,x)
    inline void backProject(const InVec2 &tex_coords, const Depth depth, OutVec3 &point)const{ //get 3D position corresponding to in tex_coords(in pixels) + depth
        if(!b_is_inverse_projection_matrix_defined_){
            log(ERROR)<<"[Camera " << u8_camera_id_ <<"]: Error, you should call setInverseProjectionMatrixFromProjectionMatrix() method before trying to backproject points... "<<endLog();
            //abort();
        }
        else{
            //First backproject texcoords
            OutVec3 v3f_tex_coords;
            v3f_tex_coords << (float)tex_coords(1), (float)tex_coords(0),1.0;
            Vector4f v4f_temp = mat43_inverse_projection_matrix_ * v3f_tex_coords;
            v4f_temp /= v4f_temp(3);
            OutVec3 ray = {v4f_temp(0),v4f_temp(1),v4f_temp(2)};
            ray -= vec3_position_;//Ray going out of camera
            ray.normalize();

            point = (OutVec3) vec3_position_ + ray*depth;
        }
    }
    
    
private:


    //! Camera ID
    unsigned int u8_camera_id_;

    //! Camera Index in vector
    unsigned int u8_camera_index_;

    //! The format of the camera image sequence.
    /*! Warning: IN ANY CASE!
      The first boost %0% format to appear is the camera ID, then frame number
     */
    std::string s_images_sequence_;

    //! The format of the camera silhouettes sequence
    /*! Warning: IN ANY CASE!
      The first boost %0% format to appear is the camera ID, then frame number
     */
    std::string s_silhouettes_sequence_;

    //! Camera Position
    Vector3f vec3_position_;
    bool b_is_position_defined_;

    //! Camera Viewing Direction
    Vector3f vec3_orientation_;
    bool b_is_orientation_defined_;
    float f_barrel_rotation_;

    //! Camera Projection Matrix
    Matrix34f mat34_projection_matrix_;
    bool b_is_projection_matrix_defined_;

    //! Camera Inverse Projection Matrix
    Matrix43f mat43_inverse_projection_matrix_;
    bool b_is_inverse_projection_matrix_defined_;

    //! Loaded Frame
    long int loaded_frame_;

    //! Loaded image
    cv::Mat loaded_image_;

    //! Loaded Silhouette
    cv::Mat loaded_silhouette_;
    cv::Mat loaded_shrunk_silhouette_;

    //! Super pixel image decomposition
    bool is_slic_defined_;
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic_clustering_;

    //! Load Projection Matrix File
    void loadMatFile(const std::string &projection_matrix_file);

    //! Set silhouette from Black Pixels
    void set_silhouette_from_black_pixels();

};


#endif
