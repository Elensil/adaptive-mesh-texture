#include "../Include/space_time_sampler.h"


/**
 * @brief SpaceTimeSampler::SpaceTimeSampler
 * @param om option manager
 */
SpaceTimeSampler::SpaceTimeSampler(const OptionManager& om){

    frame_manager_ = FrameManager(om.first_frame_,om.last_frame_,om.backward_processing_,om.first_frame_);
    std::vector<size_t> v_cameras_ids;
    //Get camera indices
    if(boost::optional< std::vector<size_t> > o_vector = om.getCamIds())
         v_cameras_ids = *o_vector;
    else
        log(ERROR) << "[SpaceTimeSampler] : no camera detected in folder" << endLog();
    if(v_cameras_ids.empty())
        return;
    log(ALWAYS)<<"[SpaceTimeSampler] : "<< v_cameras_ids.size() <<" cameras found. "<< endLog();

    //Camera Init
    v_cameras_.clear();
    std::vector<std::pair<unsigned int,Vector3f>> cam_positions;
    for(std::vector<size_t>::iterator it_camera = v_cameras_ids.begin() ; it_camera != v_cameras_ids.end() ; ++it_camera)
    {
        Camera temp_cam(*it_camera,om.images_sequences_);
        if(!om.silhouettes_sequences_.empty())  temp_cam.setSilhouettesSequence(om.silhouettes_sequences_);
        if(!om.projection_matrices_.empty()){
            temp_cam.setProjectionMatrix(om.projection_matrices_);
            temp_cam.setPositionFromProjectionMatrix();
            temp_cam.setInverseProjectionMatrixFromProjectionMatrix();
            cam_positions.push_back(std::make_pair(*it_camera,temp_cam.getPosition()));
        }
        temp_cam.setIndex(v_cameras_.size());
        v_cameras_.push_back(temp_cam);
        log(DEBUG)<<"[SpaceTimeSampler] Added camera "<<*it_camera<< " (index "<<v_cameras_.size() - 1<<")"<<endLog();
    }

    if(false)   //Export camera positions
    {
        
        std::string filename = output_folder_ + "/camera_positions.obj";
        std::ofstream outFile;
        outFile.open(filename);
        if(outFile.is_open())
        {
            outFile << "o camera_positions"<<std::endl;
            for(unsigned int i = 0; i < cam_positions.size(); ++i)
            {
                outFile<<"v "<< cam_positions[i].second.transpose()<<std::endl;
            }
            outFile.close();
        }
        else
            log(ERROR)<<"Could not export camera positions..."<<endLog();
    }

    output_folder_ = om.get_output_folder();

}


/**
 * @brief SpaceTimeSampler::~SpaceTimeSampler
 */
SpaceTimeSampler::~SpaceTimeSampler(){

}

/**
 * @brief loadImages creates an elementary PhotoUtils in map_photometric_implicit_functions_ storing basic info such as
 * camera colors and silhouettes and computes interest points,
 * @param frame
 */
void SpaceTimeSampler::loadImages(unsigned int frame){
    if(frame < frame_manager_.num_frames()){
        boost::posix_time::ptime begin;

        begin  = boost::posix_time::second_clock::local_time();
        const int& t = frame_manager_.get_frame_number(frame);
        // frame_manager_.setActiveFrameNumber(t);
        setActiveFrame(t);

        int loading_count = 0;
        #pragma omp parallel for
        for(size_t camera_index = 0 ; camera_index < v_cameras_.size() ; camera_index++)
        {
            v_cameras_[camera_index].loadFrame(t);

            #pragma omp critical
            {
                log(ALWAYS) << "\e[A"<< "[SpaceTimeSampler] : Loaded Frame "<<t<<" in Camera "<<v_cameras_[camera_index].getID()
                            <<" <index "<< camera_index <<"> ("<< ++loading_count <<"/"<< v_cameras_.size() << ")         "<<endLog();
            }
        }

        boost::posix_time::time_duration diff = boost::posix_time::second_clock::local_time() - begin;
        log(ALWAYS) << "\e[A"<< "[SpaceTimeSampler] : Took " <<diff.total_milliseconds()<<" ms to load images for frame "<< t <<"          "<<endLog();

    }
    else
        log(ERROR)<<"[SpaceTimeSampler] : Error, overflow for frame : "<<frame<<". Max Frame index is "<<frame_manager_.num_frames()<<endLog();
}


//Returns index triplet (in pseudo-barycentric coordinates) of 3 closest samples of point in triangle, along with their respective weights
void SpaceTimeSampler::get3ClosestSamples(const unsigned short faceRes, const float lambda1, const float lambda2, Vector3ui &out_S1, Vector3ui &out_S2, Vector3ui &out_S3, Vector3f &weights)const{
    Vector3ui E_bary = {int(lambda1*faceRes),int(lambda2*faceRes),int((1-lambda1-lambda2)*faceRes)};
    //int E_bary[3] = {(int)lambda1,(int)lambda2,(int)lambda3};
    weights = {(lambda1*faceRes)-E_bary[0],(lambda2*faceRes)-E_bary[1],(1-lambda1-lambda2)*faceRes-E_bary[2]};
    //float weights[3] = {lambda1-E_bary[0],lambda2-E_bary[1],lambda3-E_bary[2]};
    float sumWeights = weights[0]+weights[1]+weights[2];
    Vector3ui CP1, CP2, CP3;
    //std::vector<int> CP1(3);
    //std::vector<int> CP2(3);
    //std::vector<int> CP3(3);
    //int CP1[3], CP2[3], CP3[3]; //for the 3 closest samples (barycentric coordinates)
    for(int k=0;k<3;++k)
    {
        out_S1[k]=E_bary[k];
        out_S2[k]=E_bary[k];
        out_S3[k]=E_bary[k];
    }
    if(sumWeights<0.00001f)     //very unlikely... right on a color sample!
    {
        weights[0]=1;
        weights[1]=0;
        weights[2]=0;
    }
    else if(sumWeights-1<0.00001f)   //sum of the weights = 1
    {
        out_S1[0]+=1;
        out_S2[1]+=1;
        out_S3[2]+=1;
    }
    else                            //sum of the weights = 2
    {
        out_S1[1]+=1;
        out_S1[2]+=1;
        out_S2[0]+=1;
        out_S2[2]+=1;
        out_S3[0]+=1;
        out_S3[1]+=1;
    }
}


/**
* @briefSpaceTimeSampler::reorderTrianglesVertices Change vertices order for each triangle, to make 2D DCT directions as orthogonal as possible (down the line)
* @param in_faces vector of triangles (with 3 vertex indices)
* @param in_point vector of vertices positions
*/
template<class InTriangle, class InPoint>
void SpaceTimeSampler::reorderTrianglesVertices(std::vector<InTriangle> &in_faces ,std::vector<InPoint> &in_points)const
{
    for(int i=0;i<in_faces.size();++i)  //for every triangle
    {
        InTriangle &tri = in_faces[i];
        InPoint v1, v2, v3;
        v1 = in_points[tri.ref];
        v2 = in_points[tri.edge1];
        v3 = in_points[tri.edge2];
        Vector3f v12, v13, v23;
        v12 = v2-v1;
        v13 = v3-v1;
        v23 = v3-v2;
        //measure all three angles
        float cosv1, cosv2, cosv3;
        cosv1 = std::abs(v13.dot(v12));
        cosv2 = std::abs(v12.dot(v23));
        cosv3 = std::abs(v13.dot(v23));
        if((cosv2<cosv1)&&(cosv2<cosv3))    //most orthogonal angle is on v2
        {
            size_t tempInd = tri.edge2;
            tri.edge2 = tri.edge1;
            tri.edge1 = tri.ref;
            tri.ref = tempInd;
        }
        else if((cosv1<cosv2)&&(cosv1<cosv3))   //most orthogonal angle is on v1
        {
            size_t tempInd = tri.edge2;
            tri.edge2 = tri.ref;
            tri.ref = tri.edge1;
            tri.edge1 = tempInd;
        }
        //If v3, leave it as it is

    }
}


Vector3f SpaceTimeSampler::vectorAvgColor(const std::vector<Vector3f> &in_vec)const{

    float avgR = 0.0f;
    float avgG = 0.0f;
    float avgB = 0.0f;

    int pointsNum = in_vec.size();

    for(int i=0; i<pointsNum;++i)
    {
        avgR += in_vec[i](0);
        avgG += in_vec[i](1);
        avgB += in_vec[i](2);
    }
    avgR /= pointsNum;
    avgG /= pointsNum;
    avgB /= pointsNum;

    return Vector3f(avgR,avgG,avgB);

}


void SpaceTimeSampler::vectorMeanAndStd(const std::vector<Vector3f> &in_vec, Vector3f &out_mean, Vector3f &out_std)const{

    float avgR = 0.0f;
    float avgG = 0.0f;
    float avgB = 0.0f;

    int pointsNum = in_vec.size();

    for(int i=0; i<pointsNum;++i)
    {
        avgR += in_vec[i](0);
        avgG += in_vec[i](1);
        avgB += in_vec[i](2);
    }
    avgR /= pointsNum;
    avgG /= pointsNum;
    avgB /= pointsNum;

    out_mean = Vector3f(avgR,avgG,avgB);

    float stdR = 0.0f;
    float stdG = 0.0f;
    float stdB = 0.0f;

    for(int i=0; i<pointsNum;++i)
    {
        stdR += (in_vec[i](0)-avgR)*(in_vec[i](0)-avgR);
        stdG += (in_vec[i](1)-avgG)*(in_vec[i](1)-avgG);
        stdB += (in_vec[i](2)-avgB)*(in_vec[i](2)-avgB);
    }
    stdR /= pointsNum;
    stdG /= pointsNum;
    stdB /= pointsNum;

    stdR = sqrt(stdR);
    stdG = sqrt(stdG);
    stdB = sqrt(stdB);

    out_std = Vector3f(stdR,stdG,stdB);

    return;

}


float SpaceTimeSampler::normalizedCorrelation(const std::vector<Vector3f> &x, const std::vector<Vector3f> &y)const{

    Vector3f xmean, xstd, ymean, ystd;

    vectorMeanAndStd(x, xmean, xstd);
    vectorMeanAndStd(y, ymean, ystd);

    float ncR = 0.0f;
    float ncG = 0.0f;
    float ncB = 0.0f;

    for(int i=0;i<x.size();++i)
    {
        ncR += (x[i](0)-xmean(0))*(y[i](0) - ymean(0));
        ncG += (x[i](1)-xmean(1))*(y[i](1) - ymean(1));
        ncB += (x[i](2)-xmean(2))*(y[i](2) - ymean(2));
    }
    ncR /= (xstd(0)*ystd(0));
    ncG /= (xstd(1)*ystd(1));
    ncB /= (xstd(2)*ystd(2));

    return ncR+ncG+ncB;

}


// template void SpaceTimeSampler::reorderTrianglesVertices(std::vector<MyTriangle> &in_faces ,std::vector<Vector3f> &in_points)const;

