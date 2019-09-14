#include "../Include/camera.h"
#include <boost/filesystem.hpp>

/**
 * @brief Camera::Camera
 * @param _u8_camera_id_
 * @param _s_images_sequence_
 */
Camera::Camera(unsigned int _u8_camera_id_ , std::string _s_images_sequence_ ):u8_camera_id_(_u8_camera_id_),s_images_sequence_(_s_images_sequence_){
    b_is_projection_matrix_defined_ = false;
    b_is_inverse_projection_matrix_defined_ = false;
    b_is_orientation_defined_ = false;
    b_is_position_defined_ = false;
    loaded_frame_= -1;
    u8_camera_index_ = -1; // error if used un-initialized
    is_slic_defined_ = false;
    f_barrel_rotation_ = -1.0;
}


/**
 * @brief Camera::~Camera
 */
Camera::~Camera(){

}


/**
 * @brief Camera::loadFrame
 * @param frame
 */
void Camera::loadFrame(const long int frame){
    if(!s_images_sequence_.empty()){
        loaded_image_ = cv::imread(getImageFile(frame), CV_LOAD_IMAGE_COLOR);   // Read the file
        if(! loaded_image_.data ){                              // Check for invalid input
#pragma omp critical
            log(ERROR)<< "[ Camera "<< u8_camera_id_ <<" ] : Error, Could not open or find the image "<< getImageFile(frame) << endLog();
            return ;
        }
        else{
            loaded_frame_ = frame;

            //TBD: test other parameters
            //Apply Bilateral filter
//            cv::Mat temp;
//            cv::bilateralFilter( loaded_image_, temp, 10, 20.0, 8.0 );
//            loaded_image_ = temp;
//            //displayLoadedImage();

            if(!s_silhouettes_sequence_.empty()){
                //cv::Mat gray_scale_silhouette;
                loaded_silhouette_ = cv::imread(getSilhouetteFile(frame), CV_LOAD_IMAGE_UNCHANGED);

                std::vector<cv::Mat> channels_split;
                cv::split(loaded_silhouette_,channels_split); //Separate channels for silhouette (useful if silhouette stored in alpha channel)
                if (loaded_silhouette_.channels()>1){
                    cv::cvtColor(loaded_silhouette_,loaded_silhouette_,CV_BGRA2GRAY);
                }
                if(channels_split.size() == 4) //Alpha value available
                    cv::threshold(channels_split[3],loaded_silhouette_, 1, 255, cv::THRESH_BINARY);
                else
                    //Apply thresholding when silhouette is RGB or blended into image
                    cv::threshold(loaded_silhouette_,loaded_silhouette_, 1, 255, cv::THRESH_BINARY);



                if(! loaded_silhouette_.data){
                    #pragma omp critical
                    log(ERROR)<< "[ Camera "<< u8_camera_id_ <<" ] : Error, Could not open or find silhouette file." << endLog();
                    return ;
                }
                if(loaded_silhouette_.size().width != loaded_image_.size().width || loaded_silhouette_.size().height != loaded_image_.size().height){
                    #pragma omp critical
                    log(ERROR)<< "[ Camera "<< u8_camera_id_ <<" ] : Error, Image and Silhouette are not the same size." << endLog();
                    return ;
                }
            }

            //Init shrunk silhouette
            int morph_type = cv::MORPH_ELLIPSE;
            int erosion_size = EROSION_SIZE ;
            cv::Mat kernel = getStructuringElement( morph_type, cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ), cv::Point( erosion_size, erosion_size ) );
            cv::erode(loaded_silhouette_,loaded_shrunk_silhouette_,kernel);
            //cv::threshold(loaded_shrunk_silhouette_, loaded_shrunk_silhouette_, 1, 255, cv::THRESH_BINARY);

        }
    }
    else
    {
        #pragma omp critical
        log(ERROR)<< "[ Camera "<< u8_camera_id_ <<"] : Error, images sequence undefined."<<endLog();
        return;
    }
}


/**
 * @brief Camera::loadMatFile
 * @param projection_matrix_file
 */
void Camera::loadMatFile(const std::string &projection_matrix_file){

    std::string format = boost::filesystem::extension(projection_matrix_file);
    format = format.substr(1);
    if(!format.compare( "bat" ) && !format.compare( "txt" ) && !format.compare("P")){
        #pragma omp critical
        log(ERROR)<<"[Camera] Error: Wrong Matrix format for file "<<projection_matrix_file<<endLog();
        return;
    }

    std::ifstream fp (projection_matrix_file.c_str());

    if (!fp.is_open())
    {
        #pragma omp critical
        log(ERROR)<<"[ Camera "<< u8_camera_id_ <<" ] Error: Could not open mat file"<<std::endl;
        return;
    }
    else{
        int j = 0;
        for(std::string line; std::getline(fp, line); j++)   //read stream line by line
        {
            std::istringstream in(line);      //make a stream for the line itself
            if(std::count(line.begin(), line.end(), ' ') == 4){
                if (j < 4){///Projection matrix is composed of 3 lines
                    for(int i = 0; i < 4; i++)
                        in >> mat34_projection_matrix_(j , i);       //now read the whitespace-separated floats
                }
            }
            else if(std::count(line.begin(), line.end(), ' ') == 3){
                if (j < 4){///Projection matrix is composed of 3 lines
                    for(int i = 0; i < 4; i++)
                        in >> mat34_projection_matrix_(j , i);       //now read the whitespace-separated floats
                }
            }
            else if(std::count(line.begin(), line.end(), ' ') + 1 == 12){
                ///Projection matrix is composed of 1 line
                if (j == 0){
                for(int i = 0; i < 12; i++)
                    in >> mat34_projection_matrix_(i/4,i%4);       //now read the whitespace-separated floats
                }
                else
                {
                    #pragma omp critical
                    log(ERROR)<<"[ Camera "<< u8_camera_id_ <<" Unkown mat file format..."<<std::endl;
                    return;
                }
            }
            else{
                #pragma omp critical
                log(ERROR)<<"[ Camera "<< u8_camera_id_ <<" Error: Could not read mat file format... "<<std::count(line.begin(), line.end(), ' ') <<" words in line "<< line <<std::endl;
                return;
            }
        }
    }
    log(DEBUG)<<"Matrix (cam "<< u8_camera_id_ <<" ): "<<std::endl<< mat34_projection_matrix_ <<endLog();

    b_is_projection_matrix_defined_ = true;

    setInverseProjectionMatrixFromProjectionMatrix();

}


/**
 * @brief Camera::setPositionFromProjectionMatrix
 */
void Camera::setPositionFromProjectionMatrix()
{
    if(b_is_projection_matrix_defined_)
    {
        Vector3f col1 = {mat34_projection_matrix_(0,0), mat34_projection_matrix_(1,0), mat34_projection_matrix_(2,0)};
        Vector3f col2 = {mat34_projection_matrix_(0,1), mat34_projection_matrix_(1,1), mat34_projection_matrix_(2,1)};
        Vector3f col3 = {mat34_projection_matrix_(0,2), mat34_projection_matrix_(1,2), mat34_projection_matrix_(2,2)};
        Vector3f col4 = {mat34_projection_matrix_(0,3), mat34_projection_matrix_(1,3), mat34_projection_matrix_(2,3)};

        /*
         * % w = -det([P1,P2,P3])
            % x = det([P2,P3,P4])/w
            % y = -det([P1,P3,P4])/w
            % z = det([P1,P2,P4])/w
         * */

        Matrix3f determinant_temp;
        determinant_temp << col1,col2,col3;
        double w = -determinant_temp.determinant();
        determinant_temp << col2,col3,col4;
        vec3_position_(0) = determinant_temp.determinant()/w;
        determinant_temp << col1,col3,col4;
        vec3_position_(1) = -determinant_temp.determinant()/w;
        determinant_temp << col1,col2,col4;
        vec3_position_(2) = determinant_temp.determinant()/w;

        b_is_position_defined_ = true;

        log(DEBUG)<<"Camera "<<u8_camera_id_<<" position: "<<vec3_position_.transpose()<<endLog();
    }
    else
        #pragma omp critical
        log(ERROR)<< "[Camera "<<u8_camera_id_<<" ] in setPositionFromProjectionMatrix() : Error, projection matrix undefined"<<endLog();
}


/**
 * @brief Camera::setInverseProjectionMatrixFromProjectionMatrix
 */
void Camera::setInverseProjectionMatrixFromProjectionMatrix(){

    if(!b_is_projection_matrix_defined_)
        #pragma omp critical
        log(ERROR) << "[ Camera "<<u8_camera_id_<<" ] in function BackProject() : Projection Matrix Undefined"<< endLog();
    else{
        GLM_Mat3x4 InvProj;
        GLM_Mat4 TransP(    GLM_Vec4(mat34_projection_matrix_(0),mat34_projection_matrix_(3),mat34_projection_matrix_(6),mat34_projection_matrix_(9)),
                            GLM_Vec4(mat34_projection_matrix_(1),mat34_projection_matrix_(4),mat34_projection_matrix_(7),mat34_projection_matrix_(10)),
                            GLM_Vec4(mat34_projection_matrix_(2),mat34_projection_matrix_(5),mat34_projection_matrix_(8),mat34_projection_matrix_(11)),
                            GLM_Vec4(0,0,0,0));//Filled it as transpose, just lazy to change this
        GLM_Mat4 P = glm::transpose(TransP);
        //get P*PTranspose
        GLM_Mat4 PPTranspose =  P * TransP ; //Don't know why mat3x4 * mat4x3 is not working here... Also, the order is inverted because i am lazy
        //Inverse it
        GLM_Mat3 InvPPTranspose = glm::inverse(GLM_Mat3(PPTranspose[0][0],PPTranspose[0][1],PPTranspose[0][2],PPTranspose[1][0],PPTranspose[1][1],PPTranspose[1][2],PPTranspose[2][0],PPTranspose[2][1],PPTranspose[2][2]));

        //P+ = PTranspose*inv(P*PTranspose)
        InvProj = GLM_Mat3x4(TransP[0],TransP[1],TransP[2]) * InvPPTranspose;

        mat43_inverse_projection_matrix_<<  (float)InvProj[0][0] , (float)InvProj[1][0] , (float)InvProj[2][0],
                                            (float)InvProj[0][1] , (float)InvProj[1][1] , (float)InvProj[2][1],
                                            (float)InvProj[0][2] , (float)InvProj[1][2] , (float)InvProj[2][2],
                                            (float)InvProj[0][3] , (float)InvProj[1][3] , (float)InvProj[2][3];
    }
    b_is_inverse_projection_matrix_defined_ = true;

//    std::cout<<"Projection Matrix : "<<std::endl;
//    std::cout<<mat34_projection_matrix_<<std::endl;
//    std::cout<<"Inverse Projection Matrix : "<<std::endl;
//    std::cout<<mat43_inverse_projection_matrix_<<std::endl;
}


/**
 * @brief Camera::isInsideSilhouette
 * @param y
 * @param x
 * @return
 */
boost::optional<bool> Camera::isInsideSilhouette(const int &y, const int &x) const {
    if(! loaded_silhouette_.data)
        //No silhouette loaded
        return boost::optional<bool>(true);

    if(y < 0 || y >= loaded_silhouette_.size().height ||x < 0 || x >= loaded_silhouette_.size().width)
        return boost::none; //Outside image

    else
        if(loaded_silhouette_.at<unsigned char>(y,loaded_silhouette_.channels()*x) > 0)
            return boost::optional<bool>(true);
        else
            return boost::optional<bool>(false) ;
}

/**
 * @brief Camera::isInsideShrunkSilhouette
 * @param y
 * @param x
 * @return
 */
boost::optional<bool> Camera::isInsideShrunkSilhouette(const int &y, const int &x) const {
    if(! loaded_shrunk_silhouette_.data)
        //No silhouette loaded
        return boost::optional<bool>(true);

    if(y < 0 || y >= loaded_shrunk_silhouette_.size().height ||x < 0 || x >= loaded_shrunk_silhouette_.size().width)
        return boost::none;

    else
        if(loaded_shrunk_silhouette_.at<unsigned char>(y,loaded_shrunk_silhouette_.channels()*x) > 0)
            return boost::optional<bool>(true);
        else
            return boost::optional<bool>(false) ;
}


/**
 * @brief Camera::set_silhouette_from_black_pixels
 */
void Camera::set_silhouette_from_black_pixels(){
    // get black pixels as mask, then n*dilatation+n*erosion to fill holes (n to be defined in different cases)
    cv::Mat gray_scale_image, silhouette_to_compute;
    cv::cvtColor(loaded_image_,gray_scale_image, CV_BGRA2GRAY);
    cv::threshold(gray_scale_image, silhouette_to_compute, 15.0, 255.0, cv::THRESH_BINARY);//30.0 for temple

    int morph_type = cv::MORPH_RECT;
    int dilatation_size = 30, erosion_size = 27 ;
    cv::Mat kernel = getStructuringElement( morph_type, cv::Size( 2*dilatation_size + 1, 2*dilatation_size+1 ), cv::Point( dilatation_size, dilatation_size ) );
    cv::dilate(silhouette_to_compute,silhouette_to_compute,kernel);
    kernel = getStructuringElement( morph_type, cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ), cv::Point( erosion_size, erosion_size ) );
    cv::erode(silhouette_to_compute,silhouette_to_compute,kernel);

    loaded_silhouette_ = silhouette_to_compute;


//#pragma omp critical
//    {

//        std::cout<<"cam "<<u8_camera_id_<<std::endl;
//        cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
//        cv::imshow( "Display window", loaded_silhouette_ );
//        cv::waitKey(0);
//        displayLoadedImageMultipliedBySilhouette();

//    }
}
