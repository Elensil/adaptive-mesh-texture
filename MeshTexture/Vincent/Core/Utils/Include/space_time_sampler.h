#ifndef SPACE_TIME_SAMPLER
#define SPACE_TIME_SAMPLER



#include "matrix.h"
#include "camera.h"
#include "framemanager.h"
#include "optionmanager.h"
#include "bit_array.h"
#include "my_mesh_encoder.h"

// #include "mesh.h"


#include <boost/optional.hpp>
#include <boost/foreach.hpp>
#include <boost/tuple/tuple.hpp>

#include "stdlib.h"
#include <list>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <map>
#include <iostream>

enum SamplerMode{UNDEFINED,StaticMode,DynamicMode,CleaningMode};
enum ImplicitFunctionMode{VisualHullMode,DepthMapMode};

class SpaceTimeSampler{

    /* This class is quite the core of the whole 4D Reconstruction pipeline
     * It contains all the data needed to build the 4D Manifold lying underneath the 2D projections on every camera at each time step:
     - a set of sparse 4D points
     {- a set of regular grids at every time step} UNUSED
     - the set of calibrated cameras
     - a photometry based implicit function per timestep

     * */

public:
    SpaceTimeSampler(){}
    SpaceTimeSampler(const OptionManager &om);     // Parameters for the constructor contained in Option Manager
    ~SpaceTimeSampler();

    inline int getNumberOfFrames()const{return frame_manager_.num_frames();}

    inline int getFrameNumber(int frame)const{return frame_manager_.get_frame_number(frame);}

    inline int getActiveFrame()const{return frame_manager_.getActiveFrameNumber();}

    inline void setActiveFrame(int frame){frame_manager_.setActiveFrameNumber(frame);}

    unsigned int getNumCams()const{return v_cameras_.size();}

    //Only load images
    void loadImages(unsigned int frame = 0);
    
    template<class InTriangle = MyTriangle, class InPoint = Vector3f, class OutColor = Vector3ui, class MySpecialMesh>
    void colorPointCloud(MySpecialMesh *in_mesh,
                            int in_faceResParam,
                            int in_downsamplingThreshold)const;

    template<class InColor>
    float getColorDistance(const InColor &c1, const InColor &c2)const;

    template<class InTriangle, class InPoint>
    std::vector<Vector3f> getVerticesNormals(const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points)const;

    template<class InTriangle, class OutColor>
    long getSampleColorIndex(const InTriangle &triangle, const int tri_ind, const int faceRes, const int b0, const int b1, const std::vector<Vector3li> &in_edge_color_ind, const std::vector<unsigned long> &in_face_color_ind, const std::vector<OutColor> &in_colors)const;

    template<class InTriangle, class OutColor>
    OutColor getSampleColor(const InTriangle &triangle, const int tri_ind, const int faceRes, const int b0, const int b1, const std::vector<Vector3li> &in_edge_color_ind, const std::vector<unsigned long> &in_face_color_ind, const std::vector<OutColor> &in_colors)const;

    template<class InColor, class InTriangle, class InPoint>
    void filterColoredMeshBilateral(const float sigma_s, const float sigma_c, const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points, std::vector<InColor> &in_colors, std::vector<unsigned short> &in_face_res, const std::vector<Vector3li> &in_edge_color_ind, const std::vector<unsigned long> &in_face_color_ind)const;

    template<class InColor, class InTriangle, class InPoint>
    void filterColoredMeshLoG(const float sigma, const float lambda, const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points, std::vector<InColor> &in_colors, std::vector<unsigned short> &in_face_res, const std::vector<Vector3li> &in_edge_color_ind, const std::vector<unsigned long> &in_face_color_ind)const;


    template<class InColor, typename T>
    void sharpenColor(const float lambda, std::vector<InColor> &in_colors, std::vector<std::list<T> > &adjList)const;

    template<class InColor, class InTriangle, class InPoint>
    void bleedColor(float sigma_s, const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points, std::vector<InColor> &in_colors, std::vector<unsigned short> &in_face_res, const std::vector<Vector3li> &in_edge_color_ind, const std::vector<unsigned long> &in_face_color_ind)const;

    template<class InColor>
    void colorWeightedMedianVote(const std::vector<std::vector<InColor> > &in_color_votes, const std::vector<std::vector<float> > &in_votes_weight, std::vector<InColor> &out_colors)const;

    template<class InColor>
    void colorMedianVote(const std::vector<std::vector<InColor> > &in_color_votes, std::vector<InColor> &out_colors)const;

    template<class InColor>
    void colorWeightedMeanVote(const std::vector<std::vector<InColor> > &in_color_votes, const std::vector<std::vector<float> > &in_votes_weight, std::vector<InColor> &out_colors)const;

    template<class InColor>
    void colorTVNormVote(const std::vector<std::vector<InColor> > &in_color_votes, const std::vector<std::vector<float> > &in_votes_weight, /*const cv::Mat adj_mat,*/ const std::vector<std::list<int32_t> > &samples_adj, std::vector<InColor> &out_colors)const;

    template<class InColor, class InTriangle, class InPoint>
    void SRTest(const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points, std::vector<InColor> &in_colors,
                                std::vector<unsigned short> &in_face_res, const std::vector<Vector3li> &in_edge_color_ind,
                                const std::vector<unsigned long> &in_face_color_ind, const std::vector<std::vector<size_t> > &triangles_cam, const std::vector<float> &camera_K)const;

    
    void addAdjacencyEdge(const int32_t i1, const int32_t i2, cv::Mat &adjMat, std::vector<int> &adjMatInd)const;

    void removeAdjacencyVertex(const int32_t i1, cv::Mat &adjMat)const;

    void removeAdjacencyEdge(const int32_t i1, const int32_t i2, cv::Mat &adjMat, std::vector<int> &adjMatInd)const;

    template<class InColor>
    void filterCameraVotes(std::vector<std::vector<InColor> > &in_color_votes, std::vector<std::vector<float> > &in_votes_weight, const int kept_votes_number)const;

    template<class InTriangle, class InPoint>
    std::vector<unsigned long> getNeighbouringTriangles(const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points, const int centerTriangle, const float radius)const;

    template<class InTriangle, class InPoint>
    cv::Mat generateTextureMap(const unsigned int width, const unsigned int height, const std::vector<InTriangle> &in_faces, const std::vector<Vector2f> &in_tex_coords, const std::vector<Vector3uli> &in_tex_indices, const std::vector<InPoint> &in_points, const std::vector<std::vector<size_t> > &vertices_cam, const std::vector<float> &camera_K, std::vector<cv::Mat> &cam_tri_ind, std::string outputPath)const;


    void get3ClosestSamples(const unsigned short faceRes, const float lambda1, const float lambda2, Vector3ui &out_S1, Vector3ui &out_S2, Vector3ui &out_S3, Vector3f &weights)const;

    template<class InTriangle, class InPoint>
    void setPixelsWhereTrianglesProjectCloser(const std::vector<InTriangle> &triangles, const std::vector<InPoint> &points , cv::Mat &out_image, const Camera &cam, const float cleaning_factor = CLEANING_FACTOR)const;


    template<class InTriangle = MyTriangle, class InPoint = Vector3f, class InColor = Vector3ui, class MySpecialMesh>
    void reIndexColors(MySpecialMesh *in_mesh, int default_face_res, float quantMatCoefs[], int downsamplingThreshold)const;

    template<class InTriangle = MyTriangle, class InPoint = Vector3f, class InColor = Vector3ui, class MySpecialMesh>
    void compressColor(MySpecialMesh *in_mesh, std::vector<size_t> &in_edge_indices, int default_face_res, int quantFactor, float quantMatCoefs[], int downsamplingThreshold)const;

    template<class InTriangle = MyTriangle, class InPoint = Vector3f, class InColor = Vector3ui, class MySpecialMesh>
    void decodeCompressedColor(MySpecialMesh *in_mesh, int default_face_res, int quantFactor)const;

    template<class InTriangle, class InPoint>
    void reorderTrianglesVertices(std::vector<InTriangle> &in_faces ,std::vector<InPoint> &in_points)const;

    template<class InColor, class InTriangle>
    int downsampleTriangle(unsigned long tri, const InTriangle &myTri, int triRes, const std::vector<unsigned long>&in_face_color_ind, const std::vector<Vector3li> &in_edge_color_ind, std::vector<InColor> &in_colors, float maxIPThreshold)const;

    template<class InColor, class InTriangle>
    int downsampleTriangleMean(unsigned long tri, const InTriangle &myTri, int triRes, const std::vector<unsigned long>&in_face_color_ind, const std::vector<Vector3li> &in_edge_color_ind, std::vector<InColor> &in_colors, float maxIPThreshold)const;

    template<class InColor>
    void downsampleEdge(long edgeInd, long v1Ind, long v2Ind, std::vector<InColor> &in_colors, float maxIPThreshold)const;

    template<class InColor>
    void downsampleEdgeMean(long edgeInd, long v1Ind, long v2Ind, std::vector<InColor> &in_colors, float maxIPThreshold)const;

    template <class InTriangle, class InColor>
    void downsampleMeshColor( std::vector<InTriangle> &in_faces, std::vector<InColor> &in_colors, std::vector<unsigned short> &in_face_res, std::vector<Vector3li> &in_edge_color_ind, std::vector<unsigned long> &in_face_color_ind, int downsamplingThreshold)const;

    template<class InColor, class InTriangle>
    void downsampleTriangleChroma(unsigned long tri, const InTriangle &myTri, int triRes, const std::vector<unsigned long>&in_face_color_ind, const std::vector<Vector3li> &in_edge_color_ind, std::vector<InColor> &in_colors)const;

    template<class InColor>
    void downsampleEdgeChroma(long edgeInd, long v1Ind, long v2Ind, std::vector<InColor> &in_colors, bool writeLog=false)const;

    template<class InPoint, class InTriangle>
    bool getSurfacePointColor(InTriangle &myTri, const std::vector<InPoint> &in_points, Vector3f baryCoords, int cameraNumber, Vector3ui &out_color, bool writeLog=false, bool downsample=false)const;

    template<class InPoint, class InTriangle>
    bool getSurfacePointColorWVis(InTriangle &myTri, int triangle_idx, const std::vector<InPoint> &in_points, Vector3f baryCoords, int cameraNumber, Vector3ui &out_color, std::vector<cv::Mat> &cam_tri_ind, bool writeLog=false)const;

    template<class InPoint, class InTriangle>
    bool getSurfacePointColorNN(InTriangle &myTri, const std::vector<InPoint> &in_points, Vector3f baryCoords, int cameraNumber, Vector3ui &out_color, bool writeLog=false)const;

    template<class InColor>
    void rgbToYcc(InColor &myColor)const;

    template<class InColor>
    void yccToRgb(InColor &myColor)const;

    Vector3f vectorAvgColor(const std::vector<Vector3f> &in_vec)const;

    void vectorMeanAndStd(const std::vector<Vector3f> &in_vec, Vector3f &out_mean, Vector3f &out_std)const;

    float normalizedCorrelation(const std::vector<Vector3f> &x, const std::vector<Vector3f> &y)const;

    template<class InColor>
    void consistencyTest(std::vector<InColor> &in_colors, std::vector<unsigned short> &in_face_res, std::vector<Vector3li> &in_edge_color_ind, std::vector<unsigned long> &in_face_color_ind)const;

    template<class InTriangle, class InPoint>
    void setPixelsWhereTrianglesProjectCloser2(const std::vector<InTriangle> &triangles, const std::vector<InPoint> &points , cv::Mat &out_image, cv::Mat &depth_map, const Camera &cam, const int window_rad, const double depth_threshold, const float cleaning_factor = CLEANING_FACTOR)const;

    template<class InTriangle, class InPoint>
    void setPixelsWhereTrianglesProjectCloserWConfidence(const std::vector<InTriangle> &triangles, const std::vector<InPoint> &points, cv::Mat &out_image, cv::Mat &depth_map, cv::Mat &confidence_map, const Camera &cam, const int window_rad, const double depth_threshold, const float cleaning_factor = CLEANING_FACTOR)const;

    void writeIntMatToFile(cv::Mat& m, const char* filename)const;

    void exportOFF(std::vector<Vector3f> points, std::string filename) const;

    void exportCOFF(std::vector<Vector3f> points, std::vector<Vector3ui> colors, std::string filename) const;

    template <typename T>
    std::vector<size_t> sort_indexes(const std::vector<T> &v) const;

private:

    std::string output_folder_;

    //Frame Manager
    /* Generate the frame sequence order depending on the options.
    */
    FrameManager frame_manager_;

    // Set of calibrated cameras
    std::vector<Camera> v_cameras_;
};


template <typename T>
std::vector<size_t> SpaceTimeSampler::sort_indexes(const std::vector<T> &v) const
{

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}


/**
 * @brief colorPointCloud If frame is undefined, use active frame
 * @param in_faces
 * @param in_points
 * @param out_colors
 */
template<class InTriangle, class InPoint, class OutColor, class MySpecialMesh>
void SpaceTimeSampler::colorPointCloud( MySpecialMesh *in_mesh,
                                        int in_faceResParam,
                                        int in_downsamplingThreshold)const
{

    log(ALWAYS)<<"[SpaceTimeSampler] : Starting Point Cloud cleaning and coloring..."<<endLog();

    // std::vector<InTriangle> &in_faces = in_mesh->getRealFacesVector();
    // in_faces = in_mesh->getFacesVector();
    std::vector<InTriangle> in_faces;
    std::vector<InPoint> in_points;
    std::vector<OutColor> out_colors;
    std::vector<unsigned short> out_face_res;
    std::vector<Vector3li> out_edge_color_ind;
    std::vector<unsigned long> out_face_color_ind;
    std::vector<Vector2f> in_tex_coords;
    std::vector<Vector3uli> in_tex_indices;


    in_mesh->getFacesVector(in_faces);
    in_mesh->getPointsVector(in_points);
    in_mesh->getTexCoords(in_tex_coords);
    in_mesh->getTexIndices(in_tex_indices);

    std::string outputPath = output_folder_;
    
    // --- CONSTANTS definitions ---
    unsigned short default_face_res=MAX_FACE_RES;           //maximum face resolution
    int vote_pixel_radius = VOTE_PIXEL_RADIUS;              //When projecting votes, project on neighbouring triangles as well, with this radius.
                                                            //TODO: replace this by saving spatial coordinates (or ray) of votes, and looping through votes of neighbours (for SR)
    
    bool bInputDownsampled = false;
    int projection_margin_radius = PROJ_MARGIN_RADIUS;               
    double projection_margin_depth_threshold = PROJ_MARGIN_DEPTH_TH;

    int faceResParam = in_faceResParam;                      //used in the criterion for choosing face resolution per triangle. Choose resolution so that (#votes <= faceResParam * #samples)
                                                //(Then, pick nearest inferior power of two resolution)

    int downsamplingThreshold = in_downsamplingThreshold;             //Max error (color euclidian distance) tolerated /pixel when downsampling faces or edges.
                                                    //Set to -1 to deactivate downsampling entirely 
    
    boost::posix_time::ptime time_begin;     //to measure computation time of different algorithmic blocks.
    boost::posix_time::time_duration time_diff;

    //Make a copy of faces and points vectors
    std::vector<InTriangle> triangles(in_faces);        
    std::vector<InPoint> points(in_points);

    std::vector< std::vector<OutColor> > colors(triangles.size()); //store it in float to compute mean. Changed by Matt: vote associated with triangle
    std::vector< std::vector<Vector3f> > barycoords(triangles.size()); //created by Matt. Used to store barycentric coordinates of votes
    std::vector< std::vector<float> > incidence_weight(triangles.size()); //used to store the cos of the incidence angle for each vote, to be used as a weight
    std::vector< std::vector<short> > sample_camera_number(triangles.size());
    std::vector<int32_t> voted_color_count(triangles.size(),0);
    std::vector<InTriangle> kept_triangles;
    kept_triangles.reserve(triangles.size());

    std::vector<int32_t> new_triangle_index(triangles.size(),-1);
    
    std::vector<Vector3f> vertices_normals = getVerticesNormals(triangles,points);

    
    //for every cam
    //TODO: second version of coloring: project samples on cameras (rather than projecting pixels on triangles)
    std::vector<std::vector<float> > voting_cameras_weight(triangles.size(), std::vector<float>(v_cameras_.size(),0));  //for each triangle/camera, 0 if not seen, (surface seen)*cos(incidence angle) if seen

    for(int tri=0;tri<triangles.size();++tri)
    {
        new_triangle_index[tri] = tri;
    }

    //change vertices order for each triangle, in order to make DCT more efficiently at the end of the pipeline
    //reorderTrianglesVertices(triangles,points);

    kept_triangles=triangles;

    time_begin  = boost::posix_time::microsec_clock::local_time();

    std::vector<cv::Mat> cam_tri_ind(v_cameras_.size());
    // ---------------------------------------------------------------------------
    //      1. Project pixels (votes) on triangles)
    // ---------------------------------------------------------------------------
    
    #pragma omp parallel for schedule(dynamic)
    for(unsigned int cam = 0; cam < v_cameras_.size(); ++cam)
    {
        //TBD: make this image size a variable
        cv::Mat cam_image(IMG_HEIGHT*CLEANING_FACTOR,IMG_WIDTH*CLEANING_FACTOR,CV_32SC1,cv::Scalar(0));//Contains index of the nearest projected triangle +1! (0 is non affected)
        cv::Mat depth_map(IMG_HEIGHT*CLEANING_FACTOR,IMG_WIDTH*CLEANING_FACTOR,CV_64FC1,cv::Scalar(0.0f));     //used to store depth, so that we can grow a safety margin around foreground triangles
        cv::Mat confidence_map(IMG_HEIGHT*CLEANING_FACTOR,IMG_WIDTH*CLEANING_FACTOR,CV_64FC1,cv::Scalar(1.0f));     //used to store depth, so that we can grow a safety margin around foreground triangles
    
        const Camera &temp_cam = v_cameras_[cam];
        Vector3f cam_pos = temp_cam.getPosition();

        //First, backproject every point and fill pixels of image with (triangle index +1) if closer than previous registered value
        setPixelsWhereTrianglesProjectCloser2(triangles,points,cam_image,depth_map,temp_cam, projection_margin_radius, projection_margin_depth_threshold);
        // setPixelsWhereTrianglesProjectCloserWConfidence(triangles,points,cam_image,depth_map, confidence_map, temp_cam, projection_margin_radius, projection_margin_depth_threshold);
        
        cam_tri_ind[cam] = cam_image;

        for(unsigned int y = 0; y < IMG_HEIGHT*CLEANING_FACTOR; ++y)
        {
            for(unsigned int x = 0; x < IMG_WIDTH*CLEANING_FACTOR; ++x)
            {
                //bleed votes on neighbouring triangles as well
                std::vector<int32_t>triangles_indices;
                
                triangles_indices.reserve(1+vote_pixel_radius*vote_pixel_radius); //theoretical maximum is a bit more than 4 times this
                //int32_t triangle_idx = cam_image.at<int32_t>(y,x);
                int32_t triangle_idx;
                for(int yn = std::max(0,int(y)-vote_pixel_radius);yn<std::min(IMG_HEIGHT*CLEANING_FACTOR,double(int(y)+vote_pixel_radius)+0.1f);++yn)
                    for(int xn = std::max(0,int(x)-vote_pixel_radius);xn<std::min(IMG_WIDTH*CLEANING_FACTOR,double(int(x)+vote_pixel_radius)+0.1f);++xn)
                    {
                        triangle_idx = cam_image.at<int32_t>(yn,xn);
                        if((triangle_idx > 0) && std::find(triangles_indices.begin(),triangles_indices.end(),triangle_idx)==triangles_indices.end())
                            triangles_indices.push_back(triangle_idx);
                    }

                for(int index_ind=0;index_ind < triangles_indices.size();++index_ind)
                {
                    triangle_idx = triangles_indices[index_ind];
                    if(triangle_idx > 0)
                    {
                        triangle_idx -= (int32_t)1; //Adjust index according to offset
                        const InTriangle &tri = triangles[triangle_idx];
                        /*
                        if(new_triangle_index[triangle_idx] < 0)
                        {
                            #pragma omp critical
                            {
                                //new_triangle_index[triangle_idx] = kept_triangles.size();
                                //kept_triangles.push_back(tri);
                            }
                        }
                        */
                        cv::Vec3b col;//BGR order
                        Vector2f tex_coords;
                        bool is_safe = true;
                        
                        
                        Vector2f ref_coords,edge1_coords,edge2_coords;
                        temp_cam.getTextureCoords(points[tri.ref],ref_coords);
                        temp_cam.getTextureCoords(points[tri.edge1],edge1_coords);
                        temp_cam.getTextureCoords(points[tri.edge2],edge2_coords);

                        //Compute barycentric coordinates
                        double x1,x2,x3,y1,y2,y3,lambda1,lambda2, xd, yd;
                        x1 = ref_coords(1);     y1 = ref_coords(0);
                        x2 = edge1_coords(1);   y2 = edge1_coords(0);
                        x3 = edge2_coords(1);   y3 = edge2_coords(0);
                        xd = double(x)/(CLEANING_FACTOR);
                        yd = double(y)/(CLEANING_FACTOR);

                        lambda1 = ((y2-y3)*(xd-x3) + (x3-x2)*(yd-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3));
                        lambda2 = ((y3-y1)*(xd-x3) + (x1-x3)*(yd-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3));

                        //we want the color value of our current subpixel
                        tex_coords[1]=xd;
                        tex_coords[0]=yd;


                        //get position of point
                        Vector3f fragPos = lambda1*points[tri.ref]+lambda2*points[tri.edge1]+(1-lambda1-lambda2)*points[tri.edge2];
                        Vector3f camRay = cam_pos-fragPos;
                        Vector3f fragNormal = lambda1*vertices_normals[tri.ref]+lambda2*vertices_normals[tri.edge1]+(1-lambda1-lambda2)*vertices_normals[tri.edge2];
                        camRay.normalize();
                        fragNormal.normalize();

                        is_safe = temp_cam.getPixelColor(tex_coords,col);

                        if(is_safe)
                        {
                            #pragma omp critical
                            {
                                //store color value and barycentric coordinates
                                colors[triangle_idx].push_back(OutColor(col[2],col[1],col[0]));
                                barycoords[triangle_idx].push_back(Vector3f(lambda1,lambda2,1-lambda1-lambda2));
                                //incidence_weight[triangle_idx].push_back(std::abs(fragNormal.dot(camRay)));
                                incidence_weight[triangle_idx].push_back(1);                                    //test: incidence angle is kind of already included in visible area. No need to count it twice
                                // incidence_weight[triangle_idx].push_back(confidence_map.at<double>(y,x));           //test: linearly vary weight of pixel near border, to account for confidence in reprojection

                                sample_camera_number[triangle_idx].push_back(cam);
                                ++voted_color_count[triangle_idx];
                            }
                        }
                        if(lambda1>=0 && lambda2>=0 && (1-lambda1-lambda2)>=0)
                        {
                            voting_cameras_weight[triangle_idx][cam]+= std::abs(fragNormal.dot(camRay));    //increment surface by cos of incidence angle for (camera,triangle) pair
                            // voting_cameras_weight[triangle_idx][cam]+= std::abs(fragNormal.dot(camRay))*confidence_map.at<double>(y,x);    //increment surface by cos of incidence angle for (camera,triangle) pair
                        }
                    }
                }//new loop
            }
        }
    }

    time_diff = boost::posix_time::microsec_clock::local_time() - time_begin;
    log(ALWAYS)<<"[SpaceTimeSampler] : Get triangle per pixel + voting: "<<int(float(time_diff.total_milliseconds())/1000)<<" s"<<endLog();

    log(ALWAYS)<<"[SpaceTimeSampler] : Coloring Done, Cleaning..."<<endLog();

    in_faces.clear();
    in_faces.reserve(triangles.size());
    in_points.clear();
    in_points.reserve(points.size());
    out_colors.clear();
    
    
    std::vector<int32_t> point_indices(points.size(),-1);
    
    // long max_sample_size = points.size()+triangles.size()*(3*default_face_res/2+(default_face_res-1)*(default_face_res-2)/2);       //Assimptotical max number, if we ignore edge triangles (or with a watertight mesh)
    long max_sample_size = points.size()+triangles.size()*(3*default_face_res+(default_face_res-1)*(default_face_res-2)/2);       //Max number if every triangle is disconnected (edges not shared)
    
    int max_votes_number = 30;       //Empirical value!

    // std::vector< std::vector<OutColor> > color_map_votes(max_sample_size, std::vector<OutColor>(max_votes_number));    //to store the color map (duh). Should reserve memory space?
    // std::vector< std::vector<float> > color_map_votes_weight(max_sample_size, std::vector<float>(max_votes_number)); //each vote is weighted by distance

    std::vector< std::vector<OutColor> > color_map_votes(max_sample_size);    //to store the color map (duh). Should reserve memory space?
    std::vector< std::vector<float> > color_map_votes_weight(max_sample_size); //each vote is weighted by distance


    std::vector<std::list<int32_t> > samples_adj(max_sample_size);
    log(ALWAYS)<<"Points Size = "<<points.size()<<endLog();
    log(ALWAYS)<<"triangles Size = "<<triangles.size()<<endLog();
    log(ALWAYS)<<"samples_adj Size = "<<samples_adj.size()<<endLog();
    //int dims[] = {points.size()+triangles.size()*(3*default_face_res/2+(default_face_res-1)*(default_face_res-2)/2),points.size()+triangles.size()*(3*default_face_res/2+(default_face_res-1)*(default_face_res-2)/2)};
    // int dims[] = {1,1};
    // cv::SparseMat adj_mat(2,dims,CV_8SC1);
    int K = 20;     //max number of neighbours. Arbitrary/empirical
    cv::Mat adjListMat(max_sample_size, K, CV_32SC1, cv::Scalar(-1));
    std::vector<int> adjMatInd(max_sample_size, 1);

    for(int i=0;i<adjListMat.rows;++i)
    {
        adjListMat.at<int32_t>(i,0)=i;
    }

    std::map< std::pair<int,int>,std::pair<int,int> > edge_map;     //key is (1st vertex index, 2nd vertex index), value is (color index, edge resolution)

    std::map< std::pair<int,int>,std::pair<int,int> >::iterator it;

    //define face resolution for each triangle, and deal with vertices
    out_face_res.clear();
    out_face_res.reserve(triangles.size());
    out_edge_color_ind.clear();
    out_edge_color_ind.reserve(triangles.size());
    out_face_color_ind.clear();
    out_face_color_ind.reserve(triangles.size());

    unsigned long color_index_pointer = points.size();            //used as an incremented pointer into color_map for edges and faces values
    std::vector<int32_t> triverts(3);
    int vI, vI2;
    std::vector<float> total_cam_weight(v_cameras_.size(),0.0f);

    /* -----------------------------------------------------------------------------
    /
    /                       Filtering Cameras
    /
    / ------------------------------------------------------------------------------ */
    log(ALWAYS)<<"[SpaceTimeSampler] : Keeping best cameras only..."<<endLog();
    
    time_begin  = boost::posix_time::microsec_clock::local_time();
    
    //Vertices version
    /* ------------------------------------------------------------------------------------
    / Determine intensity multipliers per cameras
    / ------------------------------------------------------------------------------------ */
    std::vector<std::vector<size_t> > vertices_cam(points.size());
    std::vector<std::vector<size_t> > triangles_cam(triangles.size());  //keep best cams per triangle as well, for later use in SR

    //For each triangle, get best cam, and add it to vertices
    for(int32_t tri = 0; tri <triangles.size(); ++tri)
    {
        std::vector<size_t> sorted_cameras = sort_indexes(voting_cameras_weight[tri]);          //sort camera numbers by weight
        std::vector<size_t> best_cam(sorted_cameras.end()-OUT_OF_CAMERA_NUMBER, sorted_cameras.end()); //keep only the best one(s)
        triangles_cam[tri] = best_cam;
    }

    std::vector<std::vector<float> > triangle_cam_red(triangles.size(), std::vector<float> (v_cameras_.size(),0.0f));
    std::vector<std::vector<float> > triangle_cam_green(triangles.size(), std::vector<float> (v_cameras_.size(),0.0f));
    std::vector<std::vector<float> > triangle_cam_blue(triangles.size(), std::vector<float> (v_cameras_.size(),0.0f));

    std::vector<std::vector<float> > triangle_cam_weight(triangles.size(), std::vector<float> (v_cameras_.size(),0.0f));

    
    //constitute a matrix of intensity value per (triangle, camera) pairs. Will be used to get rid of less consensual cameras

    for(int32_t tri = 0; tri <triangles.size(); ++tri)
    {

        for(int i=0; i<colors[tri].size();++i)      //for each vote
        {
            if(barycoords[tri][i](0)>=0 && barycoords[tri][i](1)>=0 && barycoords[tri][i](2)>=0)
            {
                //triangle
                if(std::find(triangles_cam[tri].begin(), triangles_cam[tri].end(), sample_camera_number[tri][i]) != triangles_cam[tri].end())
                {
                    triangle_cam_red[tri][sample_camera_number[tri][i]] += float(colors[tri][i](0))*incidence_weight[tri][i]/255;
                    triangle_cam_green[tri][sample_camera_number[tri][i]] += float(colors[tri][i](1))*incidence_weight[tri][i]/255;
                    triangle_cam_blue[tri][sample_camera_number[tri][i]] += float(colors[tri][i](2))*incidence_weight[tri][i]/255;

                    triangle_cam_weight[tri][sample_camera_number[tri][i]] += incidence_weight[tri][i]; 
                }
            }
        }
    }
    
    //normalize value for each (triangle,cam)
    for(int tri=0;tri<triangles.size();++tri)
        for(int cam=0;cam<v_cameras_.size();++cam)
        {
            if(triangle_cam_weight[tri][cam]>0.0f)
            {
                triangle_cam_red[tri][cam] = triangle_cam_red[tri][cam]/triangle_cam_weight[tri][cam];
                triangle_cam_green[tri][cam] = triangle_cam_green[tri][cam]/triangle_cam_weight[tri][cam];
                triangle_cam_blue[tri][cam] = triangle_cam_blue[tri][cam]/triangle_cam_weight[tri][cam];
            }
        }
    
    //now, measure consensus of each camera WRT the others, for each vertex
    
    int iterNumber = OUT_OF_CAMERA_NUMBER-CAMERA_NUMBER;
    log(ALWAYS)<<"Iter number = "<<iterNumber<<endLog();                           
    //We compute color value per triangle, and use this to discard or select appropriate cameras. Then, we end up with a list of cameras per triangle, just as before, and we can project them back on vertices.
    //Start with OUT_OF_CAMERA_NUMBER per triangle (selected based on number of votes), and remove least consensual ones until there are CAMERA_NUMBER cameras left per triangle.
    //Now, when projected back on vertices, we might end up with cameras from nearby triangles voting on a triangle even though they were discarded.
    //To rpevent this from happening, we remove discarded cameras of triangle from the 3 vertices after the lists of cameras are projected on vertices.
    //Current implementation: Only remove camera from vertices if discrepancy is greater than a given threshold.
    std::vector<std::vector<size_t> > discarded_cameras (triangles.size());
    if (iterNumber>0)
    {
        
        for(int tri=0; tri<points.size();++tri)
        {
            discarded_cameras[tri].clear();
            discarded_cameras[tri].reserve(iterNumber);
        }
        log(ALWAYS)<<"Iter number = "<<iterNumber<<endLog();
        int totalDiscardedCams = 0;
        float max_avg_discrepancy = 0.2;
        int currentIter=0;
        //reduce list of cameras per triangle, and save list of discarded cameras per triangle (those with a big discrepancy score: intuitively, those that see something different)
        while(currentIter<iterNumber)
        {
            int discardedCamInd = 0;
            //log(ALWAYS)<<"currentCameraNumber: "<<currentCameraNumber<<endLog();
            for(int tri=0; tri<points.size();++tri)   //loop over triangles
            {
                //log(ALWAYS)<<"triangle: "<<tri<<endLog();
                int currentCameraNumber = triangles_cam[tri].size();
                std::vector<float> tri_cam_discrepancy (currentCameraNumber, 0.0f);
                for(int camI1=0;camI1<currentCameraNumber-1;++camI1)     //loop over pairs of (non-equal) camera indices.
                    for(int camI2=camI1+1;camI2<currentCameraNumber;++camI2)
                    {
                        //discrepancy is the sum on all other cameras, of the color distance between camera and current camera. (color distance = euclidian distance in color space)
                        float color_discrepancy = std::pow(triangle_cam_red[tri][triangles_cam[tri][camI1]]-triangle_cam_red[tri][triangles_cam[tri][camI2]],2);
                        color_discrepancy += std::pow(triangle_cam_green[tri][triangles_cam[tri][camI1]]-triangle_cam_green[tri][triangles_cam[tri][camI2]],2);
                        color_discrepancy += std::pow(triangle_cam_blue[tri][triangles_cam[tri][camI1]]-triangle_cam_blue[tri][triangles_cam[tri][camI2]],2);
                        color_discrepancy = std::sqrt(color_discrepancy);
                        
                        //float color_discrepancy = std::abs(triangle_cam_red[tri][triangles_cam[tri][camI1]]-triangle_cam_red[tri][triangles_cam[tri][camI2]]);
                        //color_discrepancy += std::abs(triangle_cam_green[tri][triangles_cam[tri][camI1]]-triangle_cam_green[tri][triangles_cam[tri][camI2]]);
                        //color_discrepancy += std::abs(triangle_cam_blue[tri][triangles_cam[tri][camI1]]-triangle_cam_blue[tri][triangles_cam[tri][camI2]]);

                        tri_cam_discrepancy[camI1] += color_discrepancy;
                        tri_cam_discrepancy[camI2] += color_discrepancy;
                    }

                for(int ind=0;ind<currentCameraNumber;++ind)
                {
                    tri_cam_discrepancy[ind]/=(currentCameraNumber-1);
                }

                
                //order camera indices based on discrepancy
                std::vector<size_t> sorted_discrepancy = sort_indexes(tri_cam_discrepancy);

                //remove the 'top' camera, and add it to the list of discarded cameras for this triangle, to be reapplied on vertices
                triangles_cam[tri].erase(triangles_cam[tri].begin()+sorted_discrepancy[currentCameraNumber-1]);
                if(tri_cam_discrepancy[sorted_discrepancy[currentCameraNumber-1]]>max_avg_discrepancy)
                {
                    ++totalDiscardedCams;
                    discarded_cameras[tri].push_back(triangles_cam[tri][sorted_discrepancy[currentCameraNumber-1]]);
                }
            }
            ++currentIter;
            //--currentCameraNumber;
            ++discardedCamInd;
        }

        log(ALWAYS)<<"Total discarded cameras: "<<totalDiscardedCams<<endLog();
    }

    //Project triangles' cameras on vertices
    for(int tri=0; tri<triangles.size();++tri)
    {
        vertices_cam[triangles[tri].ref].insert(vertices_cam[triangles[tri].ref].end(),triangles_cam[tri].begin(),triangles_cam[tri].end());        //add camera(s) to all 3 vertices
        vertices_cam[triangles[tri].edge2].insert(vertices_cam[triangles[tri].edge2].end(),triangles_cam[tri].begin(),triangles_cam[tri].end());
        vertices_cam[triangles[tri].edge1].insert(vertices_cam[triangles[tri].edge1].end(),triangles_cam[tri].begin(),triangles_cam[tri].end());
    }
    //now, we have a list of best cams for each vertex.

    if(iterNumber>0)
    {
        //Now, remove discarded cameras that might have been added by neighbouring triangle
        for(int tri=0; tri<triangles.size();++tri)
        {
            for(int dc=0; dc<discarded_cameras[tri].size();++dc)
            {
                std::vector<size_t> &camVec1 = vertices_cam[triangles[tri].ref];
                std::vector<size_t> &camVec2 = vertices_cam[triangles[tri].edge1];
                std::vector<size_t> &camVec3 = vertices_cam[triangles[tri].edge2];
                size_t disCam = discarded_cameras[tri][dc];
                
                while(std::find(camVec1.begin(), camVec1.end(), disCam)!=camVec1.end())
                {
                    camVec1.erase(std::remove(camVec1.begin(), camVec1.end(), disCam),camVec1.end());
                }
                while(std::find(camVec2.begin(), camVec2.end(), disCam)!=camVec2.end())
                {
                    camVec2.erase(std::remove(camVec2.begin(), camVec2.end(), disCam),camVec2.end());
                }
                while(std::find(camVec3.begin(), camVec3.end(), disCam)!=camVec3.end())
                {
                    camVec3.erase(std::remove(camVec3.begin(), camVec3.end(), disCam),camVec3.end());
                }
            }
        }
    }

    //constitute a matrix of intensity value per (vertex,camera)
    std::vector<std::vector<float> > vertex_cam_intensity(points.size(), std::vector<float> (v_cameras_.size(),0.0f));
    std::vector<std::vector<float> > vertex_cam_red(points.size(), std::vector<float> (v_cameras_.size(),0.0f));
    std::vector<std::vector<float> > vertex_cam_green(points.size(), std::vector<float> (v_cameras_.size(),0.0f));
    std::vector<std::vector<float> > vertex_cam_blue(points.size(), std::vector<float> (v_cameras_.size(),0.0f));
    std::vector<std::vector<float> > vertex_cam_weight(points.size(), std::vector<float> (v_cameras_.size(),0.0f));
    
    //compute a color per (vertex,camera).
    for(int32_t tri = 0; tri <triangles.size(); ++tri)
    {
        for(int i=0; i<colors[tri].size();++i)      //for each vote
        {
            if(barycoords[tri][i](0)>=0 && barycoords[tri][i](1)>=0 && barycoords[tri][i](2)>=0)
            {
                //vertices
                float myWeight=0.0f;
                if(std::find(vertices_cam[triangles[tri].ref].begin(), vertices_cam[triangles[tri].ref].end(), sample_camera_number[tri][i]) != vertices_cam[triangles[tri].ref].end())
                {
                    vertex_cam_intensity[triangles[tri].ref][sample_camera_number[tri][i]] += float(colors[tri][i].sum())*incidence_weight[tri][i]*barycoords[tri][i](0)/765;   //765 = 255 *3
                    vertex_cam_red[triangles[tri].ref][sample_camera_number[tri][i]] += float(colors[tri][i](0))*incidence_weight[tri][i]*barycoords[tri][i](0)/255;
                    vertex_cam_green[triangles[tri].ref][sample_camera_number[tri][i]] += float(colors[tri][i](1))*incidence_weight[tri][i]*barycoords[tri][i](0)/255;
                    vertex_cam_blue[triangles[tri].ref][sample_camera_number[tri][i]] += float(colors[tri][i](2))*incidence_weight[tri][i]*barycoords[tri][i](0)/255;
                    
                    vertex_cam_weight[triangles[tri].ref][sample_camera_number[tri][i]] += incidence_weight[tri][i]*barycoords[tri][i](0);
                }
                if(std::find(vertices_cam[triangles[tri].edge1].begin(), vertices_cam[triangles[tri].edge1].end(), sample_camera_number[tri][i]) != vertices_cam[triangles[tri].edge1].end())
                {
                    vertex_cam_intensity[triangles[tri].edge1][sample_camera_number[tri][i]] += float(colors[tri][i].sum())*incidence_weight[tri][i]*barycoords[tri][i](1)/765;
                    vertex_cam_red[triangles[tri].ref][sample_camera_number[tri][i]] += float(colors[tri][i](0))*incidence_weight[tri][i]*barycoords[tri][i](1)/255;
                    vertex_cam_green[triangles[tri].ref][sample_camera_number[tri][i]] += float(colors[tri][i](1))*incidence_weight[tri][i]*barycoords[tri][i](1)/255;
                    vertex_cam_blue[triangles[tri].ref][sample_camera_number[tri][i]] += float(colors[tri][i](2))*incidence_weight[tri][i]*barycoords[tri][i](1)/255;
                    vertex_cam_weight[triangles[tri].edge1][sample_camera_number[tri][i]] += incidence_weight[tri][i]*barycoords[tri][i](1);
                }
                if(std::find(vertices_cam[triangles[tri].edge2].begin(), vertices_cam[triangles[tri].edge2].end(), sample_camera_number[tri][i]) != vertices_cam[triangles[tri].edge2].end())
                {
                    vertex_cam_intensity[triangles[tri].edge2][sample_camera_number[tri][i]] += float(colors[tri][i].sum())*incidence_weight[tri][i]*barycoords[tri][i](2)/765;
                    vertex_cam_red[triangles[tri].ref][sample_camera_number[tri][i]] += float(colors[tri][i](0))*incidence_weight[tri][i]*barycoords[tri][i](2)/255;
                    vertex_cam_green[triangles[tri].ref][sample_camera_number[tri][i]] += float(colors[tri][i](1))*incidence_weight[tri][i]*barycoords[tri][i](2)/255;
                    vertex_cam_blue[triangles[tri].ref][sample_camera_number[tri][i]] += float(colors[tri][i](2))*incidence_weight[tri][i]*barycoords[tri][i](2)/255;
                    vertex_cam_weight[triangles[tri].edge2][sample_camera_number[tri][i]] += incidence_weight[tri][i]*barycoords[tri][i](2);
                }
            }
        }
    }
    //normalize value for each (vertex,cam)
    for(int v=0;v<points.size();++v)
        for(int cam=0;cam<v_cameras_.size();++cam)
        {
            if(vertex_cam_weight[v][cam]>0.0f)
            {
                vertex_cam_intensity[v][cam] = vertex_cam_intensity[v][cam]/vertex_cam_weight[v][cam];
                vertex_cam_red[v][cam] = vertex_cam_red[v][cam]/vertex_cam_weight[v][cam];
                vertex_cam_green[v][cam] = vertex_cam_green[v][cam]/vertex_cam_weight[v][cam];
                vertex_cam_blue[v][cam] = vertex_cam_blue[v][cam]/vertex_cam_weight[v][cam];
            }
        }

    // --------------------
    // gradient descent
    // --------------------
    std::vector<float> hidden_vertex_intensity(points.size(),0);
    std::vector<float> hidden_vertex_red(points.size(),0);
    std::vector<float> hidden_vertex_green(points.size(),0);
    std::vector<float> hidden_vertex_blue(points.size(),0);
    std::vector<float> camera_K(v_cameras_.size(),1);               //camera coeficients initialized
    std::vector<float> camera_K_r(v_cameras_.size(),1);
    std::vector<float> camera_K_g(v_cameras_.size(),1);
    std::vector<float> camera_K_b(v_cameras_.size(),1);
    //intialize vertices intensity
    for(int v=0;v<points.size();++v)
    {
        int cam=0;
        while(vertex_cam_weight[v][cam]==0.0f)
            ++cam;
        hidden_vertex_intensity[v]=vertex_cam_intensity[v][cam];
        hidden_vertex_red[v]=vertex_cam_red[v][cam];
        hidden_vertex_green[v]=vertex_cam_green[v][cam];
        hidden_vertex_blue[v]=vertex_cam_blue[v][cam];
    }
    float gamma = 0.1;
    for (int it=0;it<1000;++it) //used to be 2000. TODO: add a data term
    {       //one iteration
        #pragma omp parallel for schedule(dynamic)
        for (int v=0;v<points.size();++v)   //first, update each hidden intensity value in turn
        {
            float df=0.0f;
            float df_r=0.0f;
            float df_g=0.0f;
            float df_b=0.0f;
            float samp_n=0;
            for(int cam=0;cam<v_cameras_.size();++cam)  //compute derivative / Iv (i.e. intensity at vertex v)
            {
                if(vertex_cam_weight[v][cam]>0.0f)
                {
                    df += (camera_K[cam]*hidden_vertex_intensity[v]-vertex_cam_intensity[v][cam])*camera_K[cam]*vertex_cam_weight[v][cam];
                    df_r += (camera_K_r[cam]*hidden_vertex_red[v]-vertex_cam_red[v][cam])*camera_K_r[cam]*vertex_cam_weight[v][cam];
                    df_g += (camera_K_g[cam]*hidden_vertex_green[v]-vertex_cam_green[v][cam])*camera_K_g[cam]*vertex_cam_weight[v][cam];
                    df_b += (camera_K_b[cam]*hidden_vertex_blue[v]-vertex_cam_blue[v][cam])*camera_K_b[cam]*vertex_cam_weight[v][cam];
                    
                    samp_n+= vertex_cam_weight[v][cam];

                }
            }
            if(samp_n>0)
                df=df/samp_n;
                df_r=df_r/samp_n;
                df_g=df_g/samp_n;
                df_b=df_b/samp_n;
            hidden_vertex_intensity[v] += -gamma*df;
            hidden_vertex_red[v] += -gamma*df_r;
            hidden_vertex_green[v] += -gamma*df_g;
            hidden_vertex_blue[v] += -gamma*df_b;

        }
        
        #pragma omp parallel for schedule(dynamic)
        for (int cam=0;cam<v_cameras_.size();++cam)     //then, do the same with camera factors
        {
            float df=0.0f;
            float df_r=0.0f;
            float df_g=0.0f;
            float df_b=0.0f;
            float samp_n=0;
            for(int v=0;v<points.size();++v)
            {
                if(vertex_cam_weight[v][cam]>0.0f)
                {
                    df += (camera_K[cam]*hidden_vertex_intensity[v]-vertex_cam_intensity[v][cam])*hidden_vertex_intensity[v]*vertex_cam_weight[v][cam];
                    df_r += (camera_K_r[cam]*hidden_vertex_red[v]-vertex_cam_red[v][cam])*hidden_vertex_red[v]*vertex_cam_weight[v][cam];
                    df_g += (camera_K_g[cam]*hidden_vertex_green[v]-vertex_cam_green[v][cam])*hidden_vertex_green[v]*vertex_cam_weight[v][cam];
                    df_b += (camera_K_b[cam]*hidden_vertex_blue[v]-vertex_cam_blue[v][cam])*hidden_vertex_blue[v]*vertex_cam_weight[v][cam];
                    samp_n+= vertex_cam_weight[v][cam];
                }
            }
            if(samp_n>0)
                df=df/samp_n;
                df_r=df_r/samp_n;
                df_g=df_g/samp_n;
                df_b=df_b/samp_n;
            camera_K[cam] += -gamma*df;
            camera_K_r[cam] += -gamma*df_r;
            camera_K_g[cam] += -gamma*df_g;
            camera_K_b[cam] += -gamma*df_b;
        }
        if(it==800)
        {
            gamma=0.01;
            
        }
    }
    
    time_diff = boost::posix_time::microsec_clock::local_time() - time_begin;
    log(ALWAYS)<<"[SpaceTimeSampler] : Get best cameras per vertex, and intensity factors: "<<(float(time_diff.total_milliseconds())/1000)<<" s"<<endLog();
    


    // //generate texture map
    // if((in_tex_coords.size()==0)||(in_tex_indices.size()==0))
    // {
    //     log(ALWAYS)<<"No texture atlas found. Skipping texture generation."<<endLog();
    // }
    // else
    // {
    //     int tr;
    //     // std::vector<int> trValues = {512,1024, 1536, 2048,3072,4096,5120,6144,7168,8192};
    //     std::vector<int> trValues = {512};
    //     for(int trInd = 0;trInd<trValues.size();++trInd)
    //     {
    //         tr = trValues[trInd];
    //         if (!boost::filesystem::exists(outputPath + "texture_" + std::to_string(tr) + ".png"))
    //         {
    //             log(ALWAYS)<<"Generating texture map..."<<endLog();
    //             log(ALWAYS)<<"In tex coords size: "<<in_tex_coords.size()<<endLog();
    //             log(ALWAYS)<<"In tex indices size: "<<in_tex_indices.size()<<endLog();
    //             int texRes=tr;
    //             cv::Mat texMat = generateTextureMap(texRes, texRes, triangles, in_tex_coords, in_tex_indices, points, vertices_cam, camera_K, cam_tri_ind, outputPath);
    //             cv::imwrite(outputPath + "texture_" + std::to_string(texRes) + ".png", texMat);
    //             log(ALWAYS)<<"texture written ("<<texRes<<")"<<endLog();
    //         }

    //     }
    // }

    time_begin  = boost::posix_time::microsec_clock::local_time();
    //Filter and weight votes based on this
    for(int32_t tri = 0; tri <triangles.size(); ++tri)
    {
        std::vector<float> new_weights;
        std::vector<OutColor> new_colors;
        std::vector<Vector3f> new_bary_coords;
        std::vector<short> new_sample_camera_number;
        for(int i=0; i<colors[tri].size();++i)      //for each vote
        {
            float myWeight=0.0f;
            //1st version here takes the number of camera occurences per vertex to compute weights.
            //*
            int currentCam = 0;
            while(currentCam<vertices_cam[triangles[tri].ref].size())
            {
                if(vertices_cam[triangles[tri].ref][currentCam] == sample_camera_number[tri][i]){
                    if(barycoords[tri][i](0)>0)
                        myWeight += incidence_weight[tri][i]*barycoords[tri][i](0);
                }
                ++currentCam;
            }
            currentCam = 0;
            while(currentCam<vertices_cam[triangles[tri].edge1].size())
            {
                if(vertices_cam[triangles[tri].edge1][currentCam] == sample_camera_number[tri][i]){
                    if(barycoords[tri][i](1)>0)
                        myWeight += incidence_weight[tri][i]*barycoords[tri][i](1);
                }
                ++currentCam;
            }
            currentCam = 0;
            while(currentCam<vertices_cam[triangles[tri].edge2].size())
            {
                if(vertices_cam[triangles[tri].edge2][currentCam] == sample_camera_number[tri][i]){
                    if(barycoords[tri][i](2)>0)
                        myWeight += incidence_weight[tri][i]*barycoords[tri][i](2);
                }
                ++currentCam;
            }

            if(myWeight>0.0f)
            {
                new_weights.push_back(myWeight);
                OutColor testSmth;
                //can go beyond 255 because of camera intensity corrections (right?)
                testSmth(0)= int(float(colors[tri][i](0))/camera_K[sample_camera_number[tri][i]]);
                testSmth(0)=std::min(255,std::max(0,int(testSmth(0))));
                testSmth(1)= int(float(colors[tri][i](1))/camera_K[sample_camera_number[tri][i]]);
                testSmth(1)=std::min(255,std::max(0,int(testSmth(1))));
                testSmth(2)= int(float(colors[tri][i](2))/camera_K[sample_camera_number[tri][i]]);
                testSmth(2)=std::min(255,std::max(0,int(testSmth(2))));
                new_colors.push_back(testSmth);

                new_bary_coords.push_back(barycoords[tri][i]);
                new_sample_camera_number.push_back(sample_camera_number[tri][i]);
                total_cam_weight[sample_camera_number[tri][i]] += myWeight;
            }
        }
        incidence_weight[tri]=new_weights;
        colors[tri]=new_colors;
        barycoords[tri]=new_bary_coords;
        sample_camera_number[tri] = new_sample_camera_number;
    }

    time_diff = boost::posix_time::microsec_clock::local_time() - time_begin;
    log(ALWAYS)<<"[SpaceTimeSampler] : Filter and correct votes: "<<int(float(time_diff.total_milliseconds())/1000)<<" s"<<endLog();

    // Filtering cameras complete
    // -----------------------------------------------------------


    log(ALWAYS)<<"[SpaceTimeSampler] : Starting voting process..."<<endLog();

    time_begin  = boost::posix_time::microsec_clock::local_time();

    std::vector<size_t> edge_indices;               //temp vector to store a list of edges with their color indices
    edge_indices.clear();                           //makes color reindexing easier down the road
    edge_indices.reserve(triangles.size()*3/2);
    edge_indices.push_back(0);


    int uselessPixels=0;    //counter for pixels being freed when an edge changes resolution

    std::vector<unsigned int> votes_num(colors.size());
    for(int32_t tri = 0; tri <triangles.size(); ++tri) //For every kept triangle
    {
        votes_num[tri] = colors[tri].size();
    }
    colors.clear();
    barycoords.clear();
    sample_camera_number.clear();
    incidence_weight.clear();

    for(int32_t tri = 0; tri <triangles.size(); ++tri) //For every kept triangle
    {
        int idx[2];
        int32_t i1, i2;
        //define some local loop variable, to make the code clearer
        if(new_triangle_index[tri] >= -1)
        {
            long current_color_index;

            unsigned short faceRes=1;
            
            if(faceResParam==0)
            {
                faceRes=default_face_res;
            }
            else
            {
                float tri_score = *max_element(voting_cameras_weight[tri].begin(), voting_cameras_weight[tri].end());       //See paper for justification
                // while (2*votes_num[tri]>faceRes*faceRes*faceResParam)                   
                while(2*tri_score>faceRes*faceRes*faceResParam)
                {       
                    ++faceRes;  
                }
                if(faceRes>1)
                {
                    --faceRes;
                }
                while (not ((faceRes & (faceRes-1)) == 0))       //we want face resolutions to be powers of 2
                {
                    faceRes+=1; //WARNING: changed from truncating to taking upper value!
                }
            }

            faceRes=std::min(faceRes,default_face_res);

            unsigned long face_color_index=color_index_pointer;   //reserve space to write this face's colors
            if(faceRes==1||faceRes==2)
                face_color_index=0;
            color_index_pointer+=(faceRes-1)*(faceRes-2)/2;
            
            //fill adjacency matrix for face samples
            for(int b0=1; b0<=faceRes-1;++b0)
                for(int b1=1; b1<=faceRes-1-b0;++b1)
                {
                    if(faceRes-b0-b1==0)    //make sure we're not on the last edge
                    {
                        continue;
                    }
                    current_color_index = face_color_index + (b0-1)*faceRes - (b0+1)*b0/2 +1 + (b1-1);
                    if(b1<faceRes-1-b0)    //not last sample on its 'line'
                    {
                        i1 = current_color_index;
                        i2 = current_color_index+1;
                        addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                    }
                    if(b0<faceRes-1)
                    {
                        if(b0+b1<faceRes-1)
                        {

                            i1 = current_color_index;
                            i2 = current_color_index+faceRes-(b0-1)-2;
                            addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                        }
                        if(b1>1)
                        {

                            i1 = current_color_index;
                            i2 = current_color_index+faceRes-(b0-1)-3;
                            addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                        }
                    }
                }

            
            InTriangle &triangle = kept_triangles[new_triangle_index[tri]];
            triverts[0] = triangle.ref;
            triverts[1] = triangle.edge1;
            triverts[2] = triangle.edge2;
            Vector3li edge_ind = Vector3li(0,0,0);
            int edge_num, edge_res;
            for(int j=0;j<3;++j)
            {
                if(point_indices[triverts[j]] == -1)         //vertex not saved in new vector vector yet. Add it
                {
                    //Save index and vertex in new vectors
                    point_indices[triverts[j]] = in_points.size();
                    in_points.push_back(points[triverts[j]]);
                }
            }

            //Go get color in the input images, rather than projecting votes on the surface
            //*
            for(int b0=0;b0<=faceRes;++b0)
            {
                for(int b1=0;b1<=faceRes-b0;++b1)
                {
                    current_color_index=-1;
                    vI = -1;    //dirty trick to save code... Useful for now
                    vI2 = -1;
                    if(b0==faceRes)     //1st vertex
                    {
                        current_color_index=point_indices[triverts[0]];
                    }
                    else if(b1==faceRes)    //2nd vertex
                    {
                        current_color_index=point_indices[triverts[1]];
                    }
                    else if((b0==0)&&(b1==0))   //3rd vertex
                    {
                        current_color_index=point_indices[triverts[2]];
                    }
                    else
                    {
                        int edgeSampInd=-1;
                        if(b0==0)   //2nd edge (v3,v2)
                        {
                            vI=2;
                            vI2=1;
                            edge_num=1;
                            edgeSampInd=b1;
                        }
                        else if (b1==0)     //3rd edge (v1,v3)
                        {
                            vI=0;
                            vI2=2;
                            edge_num=2;
                            edgeSampInd=faceRes-b0;
                        }
                        else if (b0+b1==faceRes)     //1st edge, (v2,v1)
                        {
                            vI=1;
                            vI2=0;
                            edge_num=0;
                            edgeSampInd=b0;
                        }

                        if(vI>=0)
                        {
                            it = edge_map.find(std::pair<int,int>(triverts[vI],triverts[vI2]));
                            if(it!=edge_map.end())          //existing edge case
                            {
                                current_color_index = edge_indices[(it->second).first];    //get color index from edge map
                                edge_ind(edge_num,0)=(it->second).first;
                            }
                            else
                            {
                                it = edge_map.find(std::make_pair(triverts[vI2],triverts[vI]));
                                if(it!=edge_map.end())      //inverted edge case
                                {
                                    int temp = vI;      //does this work as intended?
                                    vI = vI2;
                                    vI2=temp;
                                    current_color_index = edge_indices[(it->second).first];    //get color index from edge map
                                    edge_ind(edge_num,0)=-(it->second).first;
                                }
                                else    //not recorded yet
                                {
                                    current_color_index = color_index_pointer;
                                    edge_ind(edge_num,0)=edge_indices.size();
                                    
                                    edge_map.insert(std::pair<std::pair<int,int>,std::pair<int,int> >(std::pair<int,int>(triverts[vI],triverts[vI2]),std::pair<int,int>(edge_indices.size(),faceRes)));
                                    edge_indices.push_back(color_index_pointer);
                                    color_index_pointer+=faceRes;   //faceRes-1 color samples + 1 value for edge Res
                                    //Display edge res in cyan for debugging
                                    color_map_votes[current_color_index].push_back(OutColor(faceRes,255,255));      //record edgeRes
                                    color_map_votes_weight[current_color_index].push_back(1);
                                    //here, we put blue and green canal to the value of an edge vote, to minimize diskspace (useful?)
                                     //fill adjacency matrix for edge samples:
                                    // edge-edge
                                    for(int adj=1;adj<faceRes-1;++adj)  //watch out for 'edge res' index
                                    {
                                        i1 = current_color_index+adj;
                                        i2 = current_color_index+adj+1;
                                        addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                                    }
                                    //vertex-edge
                                    i1 = point_indices[triverts[vI]];
                                    i2 = current_color_index+1;
                                    addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                                    i1 = point_indices[triverts[vI2]];
                                    i2 = current_color_index+faceRes-1;
                                    addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                                    
                                    //face-edge: dealt with later
                                }
                            }
                            if(color_map_votes[current_color_index][0](0)>faceRes){        //this triangle has lower resolution than the one sharing its edge.
                                uselessPixels+=color_map_votes[current_color_index][0](0)-faceRes;
                                edge_res=color_map_votes[current_color_index][0](0);    //In this case, we choose the lower resolution for this edge, for a more consistent looking image
                                color_map_votes[current_color_index][0](0)=faceRes;

                                for(int edge_i=1;edge_i<=faceRes-1;edge_i++){            //Drop extra samples and move the "good" ones to their new place
                                    color_map_votes[current_color_index+edge_i] = color_map_votes[current_color_index+(edge_res*edge_i/faceRes)];
                                    color_map_votes_weight[current_color_index+edge_i] = color_map_votes_weight[current_color_index+(edge_res*edge_i/faceRes)];
                                }
                                
                                for(int edge_i=faceRes;edge_i<edge_res;++edge_i)
                                {
                                    removeAdjacencyVertex(edge_i,adjListMat);
                                }
                                removeAdjacencyEdge(point_indices[triverts[vI2]],current_color_index+edge_res-1, adjListMat,adjMatInd);
                                removeAdjacencyEdge(current_color_index+faceRes-1,current_color_index+faceRes, adjListMat,adjMatInd);
                                addAdjacencyEdge(point_indices[triverts[vI2]],current_color_index+faceRes-1, adjListMat,adjMatInd);
                                
                            }
                            edge_res=color_map_votes[current_color_index][0](0);
                            if((edgeSampInd*edge_res)%faceRes==0)          //if the edge has a lower resolution than the face, only add vote if the current edge point is a sample
                            {
                                current_color_index+=edgeSampInd*edge_res/faceRes;     // (-1 +1 because of 1st value used for the edgeRes)
                            }
                            else
                            {
                                current_color_index=-1;     //don't do anything
                            }
                            if(edge_ind(edge_num,0)<0)
                            {
                                current_color_index=-1;
                            }
                        }
                        //edge case taken care of. Only the face to go!!
                        else
                        {
                            current_color_index = face_color_index + (b0-1)*faceRes - (b0*(b0+1))/2 +1 + (b1-1);
                        }
                    }
                    if(current_color_index>=0)
                    {
                        float lambda1 = float(b0)/float(faceRes);
                        float lambda2 = float(b1)/float(faceRes);
                        float lambda3 = 1-lambda1-lambda2;
                        Vector3f myBarycoords = Vector3f(lambda1,lambda2,lambda3);
                        for(int verNum=0;verNum<3;++verNum)     //for each vertex
                        {
                            for(int myCam=0;myCam<vertices_cam[triverts[verNum]].size();++myCam)
                            {
                                int camNum = vertices_cam[triverts[verNum]][myCam];
                                OutColor inputColor=OutColor(0,0,0);
                                bool is_safe=true;
                                if(tri==-1)
                                {

                                }
                                else
                                {
                                    is_safe = getSurfacePointColor(triangle, points, myBarycoords,camNum,inputColor, false, bInputDownsampled);

                                }
                                if(is_safe)
                                {
                                    inputColor(0) = (unsigned int)(std::min(255.0f,std::max(0.0f,float(inputColor(0))/camera_K[camNum])));
                                    inputColor(1) = (unsigned int)(std::min(255.0f,std::max(0.0f,float(inputColor(1))/camera_K[camNum])));
                                    inputColor(2) = (unsigned int)(std::min(255.0f,std::max(0.0f,float(inputColor(2))/camera_K[camNum])));
                                    color_map_votes[current_color_index].push_back(inputColor);
                                    color_map_votes_weight[current_color_index].push_back(myBarycoords(verNum));
                                }
                            }
                        }
                    }
                }
            }
            //Update triangle
            triangle.ref = point_indices[triangle.ref];
            triangle.edge1 = point_indices[triangle.edge1];
            triangle.edge2 = point_indices[triangle.edge2];
            in_faces.push_back(triangle);
            out_face_res.push_back(faceRes);
            out_face_color_ind.push_back(face_color_index);
            out_edge_color_ind.push_back(edge_ind);
        }
    }

    log(ALWAYS)<<"Dead pixels remaining: "<<uselessPixels<<endLog();
    log(ALWAYS)<<"Filling adjacency matrix"<<endLog();
    //loop over triangles again to fill adjacency matrix for edge-face pairs
    //#pragma omp parallel for schedule(dynamic)
    for(int32_t tri = 0; tri <triangles.size(); ++tri) //For every kept triangle
    {
        unsigned short faceRes = out_face_res[tri];
        unsigned long faceInd = out_face_color_ind[tri];
        //get edgeRes
        int s0=1;
        int s1=1;
        int s2=1;
        int idx[2];
        int i1, i2;
        if(out_edge_color_ind[tri](0)<0)
            s0=-1;
        if(out_edge_color_ind[tri](1)<0)
            s1=-1;
        if(out_edge_color_ind[tri](2)<0)
            s2=-1;
        int32_t eI0 = edge_indices[std::abs(out_edge_color_ind[tri](0))];
        int edgeRes0 = color_map_votes[eI0][0](0);
        int32_t eI1 = edge_indices[std::abs(out_edge_color_ind[tri](1))];
        int edgeRes1 = color_map_votes[eI1][0](0);
        int32_t eI2 = edge_indices[std::abs(out_edge_color_ind[tri](2))];
        int edgeRes2 = color_map_votes[eI2][0](0);
        //First, add edge-to-edge links next to vertices
        if(edgeRes0>=2)
        {
            if(edgeRes2>=2)
            {
                i1 = eI0+(edgeRes0+s0*edgeRes0)/2-s0;           //the trick with s_k allows us to write (edgeRes-1 or 1 depending on the sign)
                i2 = eI2+(edgeRes2-s2*edgeRes2)/2+s2;
                addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
            }
            if(edgeRes1>=2)
            {
                i1 = eI1+(edgeRes1+s1*edgeRes1)/2-s1;
                i2 = eI0+(edgeRes0-s0*edgeRes0)/2+s0;
                addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
            }
        }
        if(edgeRes1>=2 && edgeRes2>=2)
        {
            i1 = eI2+(edgeRes2+s2*edgeRes2)/2-s2;
            i2 = eI1+(edgeRes1-s1*edgeRes1)/2+s1;
            addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
        }

        //Then, add links between face and edge samples
        
        if(faceRes>2)   //most tedious part...
        {
            //3rd edge
            for(int ei=1;ei<edgeRes2;++ei)
            {
                //attach edge to two face samples (and not the other way around)
                int b0 = ei*(faceRes/edgeRes2);
                if(b0>1)
                {
                    i1 = eI2+(edgeRes2+s2*edgeRes2)/2-s2*ei;                //couple samples (b0,0) & (b0-1,1)
                    i2 = faceInd+ (b0-2)*faceRes - (b0-1)*b0/2 +1;
                    addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                }
                if(b0<faceRes-1)
                {
                    i1 = eI2+(edgeRes2+s2*edgeRes2)/2-s2*ei;                //couple samples (b0,0) & (b0,1)
                    i2 = faceInd + (b0-1)*faceRes - b0*(b0+1)/2 + 1;
                    addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                }
            }
            
            //2nd edge
            for(int ei=1;ei<edgeRes1;++ei)
            {
                int b1 = ei*(faceRes/edgeRes1);
                if(b1>1)
                {
                    i1 = eI1+(edgeRes1-s1*edgeRes1)/2+s1*ei;                                    //couple samples (0,b1) & (1,b1-1)
                    i2 = faceInd + b1-2;
                    addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                }
                if(b1<faceRes-1)
                {
                    i1 = eI1+(edgeRes1-s1*edgeRes1)/2+s1*ei;                                    //couple samples (0,b1) & (1,b1)
                    i2 = faceInd + b1-1;
                    addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                }
            }

            //1st edge
            for(int ei=1;ei<edgeRes0;++ei)
            {
                int b0 = ei*(faceRes/edgeRes0);
                if(b0>1)
                {
                    i1 = eI0+(edgeRes0-s0*edgeRes0)/2+s0*ei;                                  //couple samples (b0,faceRes-b0) & (b0-1,faceRes-b0)
                    i2 = faceInd+ (b0-1)*faceRes - b0*(b0+1)/2;
                    addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                }
                if(b0<faceRes-1)
                {
                    i1 = eI0+(edgeRes0-s0*edgeRes0)/2+s0*ei;                                     //couple samples (b0,faceRes-b0) & (b0,faceRes-b0-1)
                    i2 = faceInd+ b0*faceRes - (b0+2)*(b0+1)/2;
                    addAdjacencyEdge(i1,i2,adjListMat,adjMatInd);
                }
            }

        }
    }
    
    log(ALWAYS)<<"[SpaceTimeSampler] : triangles size "<<triangles.size()<<endLog();
    log(ALWAYS)<<"[SpaceTimeSampler] : in_faces size "<<in_faces.size()<<endLog();

    time_diff = boost::posix_time::microsec_clock::local_time() - time_begin;
    log(ALWAYS)<<"[SpaceTimeSampler] : Distribute votes on samples: "<<int(float(time_diff.total_milliseconds())/1000)<<" s"<<endLog();


    // ***********************************************
    // ***              1ST STEP                   ***
    // ***********************************************
    out_colors.resize(color_index_pointer);
    color_map_votes.resize(color_index_pointer);
    color_map_votes_weight.resize(color_index_pointer);
    // samples_adj.resize(color_index_pointer);

    //filterCameraVotes(color_map_votes, color_map_votes_weight, 4);
    //colorWeightedMedianVote(color_map_votes, color_map_votes_weight, out_colors);
    time_begin  = boost::posix_time::microsec_clock::local_time();
    colorWeightedMeanVote(color_map_votes, color_map_votes_weight, out_colors);
    // colorMedianVote(color_map_votes,out_colors);
    time_diff = boost::posix_time::microsec_clock::local_time() - time_begin;
    log(ALWAYS)<<"[SpaceTimeSampler] : Take weighted mean: "<<int(float(time_diff.total_milliseconds())/1000)<<" s"<<endLog();

    //Compute old model for vector of edge color indices, to be used as input for some functions.
    //For now, information is split between edge_indices and out_edge_color_ind (I think? Or something like that)
    //Regular edge indices vector is computed at the end when reordering samples to save space. This is a temporary construct, so that we can correctly access color samples
    std::vector<Vector3li> temp_edge_color_ind(out_edge_color_ind.size());
    for(int tri=0;tri<triangles.size();++tri)
    {
        for(int e=0;e<3;++e)
        {
            if (out_edge_color_ind[tri](e)>=0)
                temp_edge_color_ind[tri](e) = edge_indices[out_edge_color_ind[tri](e)];
            else
                temp_edge_color_ind[tri](e) = - edge_indices[- out_edge_color_ind[tri](e)];
        }
    }

    // ****************************************************************************
    // ***              2ND STEP - Filtering, post-processing                   ***
    // ****************************************************************************
    if (true)
    {
        log(ALWAYS)<<"[SpaceTimeSampler] : filtering colored mesh"<<endLog();

        float sigma_s = 0.005f;
        float sigma_c = 60.0f;
        time_begin  = boost::posix_time::microsec_clock::local_time();

        bleedColor(sigma_s, in_faces, in_points, out_colors, out_face_res, temp_edge_color_ind, out_face_color_ind);
        // filterColoredMeshBilateral(sigma_s, sigma_c, in_faces, in_points, out_colors, out_face_res, temp_edge_color_ind, out_face_color_ind);
        //filterColoredMeshLoG(0.0005f, 0.3, in_faces, in_points, out_colors, out_face_res, temp_edge_color_ind, out_face_color_ind);
        


        time_diff = boost::posix_time::microsec_clock::local_time() - time_begin;
        log(ALWAYS)<<"[SpaceTimeSampler] : Color bleeding: "<<int(float(time_diff.total_milliseconds())/1000)<<" s"<<endLog();
        log(ALWAYS)<<"[SpaceTimeSampler] : Filtering complete"<<endLog();
        // colorTVNormVote(color_map_votes,color_map_votes_weight,/*adj_mat,*/samples_adj,out_colors);
    }


    //sharpenColor(3.0,out_colors,samples_adj);

    // Downsampling part

    //Update input mesh
    in_mesh->setFacesVector(in_faces);
    in_mesh->setPointsVector(in_points);
    in_mesh->setColorsVector(out_colors);
    in_mesh->setFacesResVector(out_face_res);
    in_mesh->setEdgesIndVector(out_edge_color_ind);
    in_mesh->setFacesIndVector(out_face_color_ind);
    in_mesh->setEdgesRealColorInd(edge_indices);
    //*/
    // **********************************
    // rearrange vertices/edges/triangles order to make image compression more efficient
    // Stand-by for now: pain in the ass, and not really effective
    // **********************************
    float quantMatCoefs[] = {1,0.1,0.01};
    
    reIndexColors(in_mesh, default_face_res, quantMatCoefs, downsamplingThreshold);
    
    //The following lines is a necessary if reIndexColors is not run (I think?)
    // out_edge_color_ind=temp_edge_color_ind;
    // in_mesh->setEdgesIndVector(out_edge_color_ind);

    // out_colors[0](0)=1;
    //consistencyTest(out_colors,out_face_res,out_edge_color_ind,out_face_color_ind);

    log(ALWAYS)<<"Cleaning Done. Kept "<<in_points.size()<<" points, and "<<in_faces.size()<<" faces."<<endLog();

    
}


template<class InColor>
float SpaceTimeSampler::getColorDistance(const InColor &c1, const InColor &c2)const{
    return std::abs(c1(0)-c2(0)) + std::abs(c1(1)-c2(1)) + std::abs(c1(2)-c2(2));
}


template<class InTriangle, class InPoint>
std::vector<Vector3f> SpaceTimeSampler::getVerticesNormals(const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points)const{

    std::vector<Vector3f> out_normals(in_points.size(), Vector3f(0.0,0.0,0.0));
    for(int tri=0;tri<in_faces.size();++tri)
    {
        Vector3f v1, v2, n;
        Vector3f Vref, Vedge1, Vedge2;
        const InTriangle &triangle = in_faces[tri]; 
        Vref = in_points[triangle.ref];
        Vedge1 = in_points[triangle.edge1];
        Vedge2 = in_points[triangle.edge2];
        v1 = Vedge1-Vref;
        v2 = Vedge2-Vref;
        n=Vector3f(v1(1)*v2(2)-v1(2)*v2(1) , v1(2)*v2(0)-v1(0)*v2(2) , v1(0)*v2(1)-v1(1)*v2(0));
        n.normalize();
        out_normals[triangle.ref]+=n;
        out_normals[triangle.edge1]+=n;
        out_normals[triangle.edge2]+=n;
    }
    for(int vert=0;vert<in_points.size();++vert)
    {
        out_normals[vert].normalize();
    }
    return out_normals;
}

// ------------------------------------
// Can return -1 is there is no sample on these coordinates (for edge samples)
// see getSampleColor for interpolation
// ------------------------------------
template<class InTriangle, class OutColor>
long SpaceTimeSampler::getSampleColorIndex( const InTriangle &triangle,
                                            const int tri_ind,
                                            const int faceRes,
                                            const int b0,
                                            const int b1,
                                            const std::vector<Vector3li> &in_edge_color_ind,
                                            const std::vector<unsigned long> &in_face_color_ind,
                                            const std::vector<OutColor> &in_colors)const
{

    long myInd=-1;
    long edge_ind;
    int edge_res;
    if(b0==faceRes)         //v ref
    {
        myInd = triangle.ref;
    }
    else if(b1==faceRes)    //v edge1
    {
        myInd = triangle.edge1;   
    }
    else if(b0==0 && b1==0) //v edge2
    {
        myInd = triangle.edge2;
    }
    else if(b0==0)          //2nd edge (v3,v2)
    {
        edge_ind = in_edge_color_ind[tri_ind](1,0);
        edge_res = in_colors[std::abs(edge_ind)](0);
        if((b1*edge_res)%faceRes==0)
        {
            if(edge_ind<0)
            {
                myInd = -edge_ind+edge_res-(b1*edge_res/faceRes);
            }
            else
            {
                myInd = edge_ind+b1*edge_res/faceRes;
            }
        }    
    }
    else if(b1==0)          //3rd edge (v1,v3)
    {
        edge_ind = in_edge_color_ind[tri_ind](2,0);
        edge_res = in_colors[std::abs(edge_ind)](0);
        if((b0*edge_res)%faceRes==0)
        {
            if(edge_ind<0)
            {
                myInd = -edge_ind+b0*edge_res/faceRes;
            }
            else
            {
                myInd = edge_ind+edge_res-b0*edge_res/faceRes;
            }
        }
    }
    else if(b0+b1==faceRes) //1st edge (v2, v1)
    {
        edge_ind = in_edge_color_ind[tri_ind](0,0);
        edge_res = in_colors[std::abs(edge_ind)](0);
        if((b1*edge_res)%faceRes==0)
        {
            if(edge_ind<0)
            {
                myInd = -edge_ind+b1*edge_res/faceRes;
            }
            else
            {
                myInd = edge_ind+b0*edge_res/faceRes;
            }
        }
    }
    else                    //face
    {
        myInd = in_face_color_ind[tri_ind]+(b0-1)*faceRes - (b0*(b0+1))/2 +1 + (b1-1);
    }
    return myInd;
}


template<class InTriangle, class OutColor>
OutColor SpaceTimeSampler::getSampleColor(  const InTriangle &triangle,
                                        const int tri_ind,
                                        const int faceRes,
                                        const int b0,
                                        const int b1,
                                        const std::vector<Vector3li> &in_edge_color_ind,
                                        const std::vector<unsigned long> &in_face_color_ind,
                                        const std::vector<OutColor> &in_colors)const
{

    long myInd=-1;
    long edge_ind;
    int edge_res;
    OutColor myColor;
    if(b0==faceRes)         //v ref
    {
        myInd = triangle.ref;
        myColor = in_colors[myInd];
    }
    else if(b1==faceRes)    //v edge1
    {
        myInd = triangle.edge1;   
        myColor = in_colors[myInd];
    }
    else if(b0==0 && b1==0) //v edge2
    {
        myInd = triangle.edge2;
        myColor = in_colors[myInd];
    }
    else if(b0==0)          //2nd edge (v3,v2)
    {
        edge_ind = in_edge_color_ind[tri_ind](1,0);
        edge_res = in_colors[std::abs(edge_ind)](0);
        if((b1*edge_res)%faceRes==0)
        {
            if(edge_ind<0)
            {
                myInd = -edge_ind+edge_res-(b1*edge_res/faceRes);
            }
            else
            {
                myInd = edge_ind+b1*edge_res/faceRes;
            }
            myColor = in_colors[myInd];
        }
        else    //interpolation
        {

            float coef = float(b1*edge_res)/float(faceRes);
            int b11 = int(coef);
            int b12 = b11+1;
            coef-=b11;
            OutColor c1, c2;
            if(b11==0)
            {
                myInd = triangle.edge2;
            }
            else if(edge_ind<0)
            {
                myInd = -edge_ind+edge_res-b11;
            }
            else
            {
                myInd = edge_ind+b11;
            }
            c1 = in_colors[myInd];
            if(b12==edge_res)
            {
                myInd = triangle.edge1;
            }
            else if(edge_ind<0)
            {
                myInd = -edge_ind+edge_res-b12;
            }
            else
            {
                myInd = edge_ind+b12;
            }
            c2 = in_colors[myInd];
            myColor = OutColor((1-coef)*c1(0)+coef*c2(0),(1-coef)*c1(1)+coef*c2(1),(1-coef)*c1(2)+coef*c2(2));
        }    
    }
    else if(b1==0)          //3rd edge (v1,v3)
    {
        edge_ind = in_edge_color_ind[tri_ind](2,0);
        edge_res = in_colors[std::abs(edge_ind)](0);
        if((b0*edge_res)%faceRes==0)
        {
            if(edge_ind<0)
            {
                myInd = -edge_ind+b0*edge_res/faceRes;
            }
            else
            {
                myInd = edge_ind+edge_res-b0*edge_res/faceRes;
            }
            myColor = in_colors[myInd];
        }
        else    //interpolation
        {
            // float coef = float(b0*edge_res/faceRes);
            float coef = float(b0*edge_res)/float(faceRes);
            int b01 = int(coef);
            int b02 = b01+1;
            coef-=b01;
            OutColor c1, c2;
            if(b01==0)
            {
                myInd = triangle.edge2;
            }
            else if(edge_ind<0)
            {
                myInd = -edge_ind+b01;
            }
            else
            {
                myInd = edge_ind+edge_res-b01;
            }
            c1 = in_colors[myInd];
            if(b02==edge_res)
            {
                myInd = triangle.ref;
            }
            else if(edge_ind<0)
            {
                myInd = -edge_ind+b02;
            }
            else
            {
                myInd = edge_ind+edge_res-b02;
            }
            c2 = in_colors[myInd];
            myColor = OutColor((1-coef)*c1(0)+coef*c2(0),(1-coef)*c1(1)+coef*c2(1),(1-coef)*c1(2)+coef*c2(2));
        }
    }
    else if(b0+b1==faceRes) //1st edge (v2, v1)
    {
        edge_ind = in_edge_color_ind[tri_ind](0,0);
        edge_res = in_colors[std::abs(edge_ind)](0);
        if((b1*edge_res)%faceRes==0)
        {
            if(edge_ind<0)
            {
                myInd = -edge_ind+b1*edge_res/faceRes;
            }
            else
            {
                myInd = edge_ind+b0*edge_res/faceRes;
            }
            myColor = in_colors[myInd];
        }
        else    //interpolation
        {
            // float coef = float(b1*edge_res/faceRes);
            float coef = float(b1*edge_res)/float(faceRes);
            int b11 = int(coef);
            int b12 = b11+1;
            coef-=b11;
            OutColor c1, c2;
            if(b11==0)
            {
                myInd = triangle.ref;
            }
            else if(edge_ind<0)
            {
                myInd = -edge_ind+b11;
            }
            else
            {
                myInd = edge_ind+edge_res-b11;
            }
            c1 = in_colors[myInd];
            if(b12==edge_res)
            {
                myInd = triangle.edge1;
            }
            else if(edge_ind<0)
            {
                myInd = -edge_ind+b12;
            }
            else
            {
                myInd = edge_ind+edge_res-b12;
            }
            c2 = in_colors[myInd];
            myColor = OutColor((1-coef)*c1(0)+coef*c2(0),(1-coef)*c1(1)+coef*c2(1),(1-coef)*c1(2)+coef*c2(2));
        }
    }
    else                    //face
    {
        myInd = in_face_color_ind[tri_ind]+(b0-1)*faceRes - (b0*(b0+1))/2 +1 + (b1-1);
        myColor = in_colors[myInd];
    }
    return myColor;
}

template<class InTriangle, class InPoint>
std::vector<unsigned long> SpaceTimeSampler::getNeighbouringTriangles(const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points, const int myTriangle, const float radius)const{

    std::queue<unsigned long> triangle_queue;   //queue of triangles whithin R, for which we haven't checked neighbours yet
    std::vector<unsigned long> triangle_list;   //list of triangles within raduis R of triangle (i.e. either one of the 3 vertices)
    triangle_queue.push(myTriangle);
    triangle_list.push_back(myTriangle);

    Vector3f Vref, Vedge1, Vedge2;
    const InTriangle &triangle = in_faces[myTriangle]; 
    Vref = in_points[triangle.ref];
    Vedge1 = in_points[triangle.edge1];
    Vedge2 = in_points[triangle.edge2];
    float R2 = pow(radius,2);
    //triangle_queue.pop();
    while(!triangle_queue.empty())
    {   //check for neighbouring triangles;
        Vector3f Tref, Tedge1, Tedge2;  //get 3 vertices
        int32_t ref_ind, edge1_ind, edge2_ind;
        ref_ind = in_faces[triangle_queue.front()].ref;
        edge1_ind = in_faces[triangle_queue.front()].edge1;
        edge2_ind = in_faces[triangle_queue.front()].edge2;
        Tref = in_points[ref_ind];
        Tedge1 = in_points[edge1_ind];
        Tedge2 = in_points[edge2_ind];
        //For each vertex, compute distance between vertex and the 3 vertices of original triangle, and add all neighbouring triangles if vertex is within raduis.
        float dist_vc2;
        // --- ref ---
        dist_vc2 = std::min(pow(Tref(0)-Vref(0),2)+pow(Tref(1)-Vref(1),2)+pow(Tref(2)-Vref(2),2),std::min(pow(Tref(0)-Vedge1(0),2)+pow(Tref(1)-Vedge1(1),2)+pow(Tref(2)-Vedge1(2),2),pow(Tref(0)-Vedge2(0),2)+pow(Tref(1)-Vedge2(1),2)+pow(Tref(2)-Vedge2(2),2)));
        if(dist_vc2<=R2)
        {
            for(int32_t k = 0; k<in_faces.size(); ++k) //For every kept triangle
            {
                if((in_faces[k].ref==ref_ind||in_faces[k].edge1==ref_ind||in_faces[k].edge2==ref_ind)&&(std::find(triangle_list.begin(),triangle_list.end(),k)==triangle_list.end()))
                {   //triangle has this vertex, and it's not in the list yet
                    triangle_queue.push(k);
                    triangle_list.push_back(k);
                }
            }
        }
        // --- edge1 ---
        dist_vc2 = std::min(pow(Tedge1(0)-Vref(0),2)+pow(Tedge1(1)-Vref(1),2)+pow(Tedge1(2)-Vref(2),2),std::min(pow(Tedge1(0)-Vedge1(0),2)+pow(Tedge1(1)-Vedge1(1),2)+pow(Tedge1(2)-Vedge1(2),2),pow(Tedge1(0)-Vedge2(0),2)+pow(Tedge1(1)-Vedge2(1),2)+pow(Tedge1(2)-Vedge2(2),2)));
        if(dist_vc2<=R2)
        {
            for(int32_t k = 0; k<in_faces.size(); ++k) //For every kept triangle
            {
                if((in_faces[k].ref==edge1_ind||in_faces[k].edge1==edge1_ind||in_faces[k].edge2==edge1_ind)&&(std::find(triangle_list.begin(),triangle_list.end(),k)==triangle_list.end()))
                {   //triangle has this vertex, and it's not in the list yet
                    triangle_queue.push(k);
                    triangle_list.push_back(k);
                }
            }
        }
        // --- edge2 ---
        dist_vc2 = std::min(pow(Tedge2(0)-Vref(0),2)+pow(Tedge2(1)-Vref(1),2)+pow(Tedge2(2)-Vref(2),2),std::min(pow(Tedge2(0)-Vedge1(0),2)+pow(Tedge2(1)-Vedge1(1),2)+pow(Tedge2(2)-Vedge1(2),2),pow(Tedge2(0)-Vedge2(0),2)+pow(Tedge2(1)-Vedge2(1),2)+pow(Tedge2(2)-Vedge2(2),2)));
        if(dist_vc2<=R2)
        {
            for(int32_t k = 0; k<in_faces.size(); ++k) //For every kept triangle
            {
                if((in_faces[k].ref==edge2_ind||in_faces[k].edge1==edge2_ind||in_faces[k].edge2==edge2_ind)&&(std::find(triangle_list.begin(),triangle_list.end(),k)==triangle_list.end()))
                {   //triangle has this vertex, and it's not in the list yet
                    triangle_queue.push(k);
                    triangle_list.push_back(k);
                }
            }
        }
        //depop this triangle and go to the next one
        triangle_queue.pop();
    }
    return triangle_list;
}

template<class InColor, class InTriangle, class InPoint>
void SpaceTimeSampler::filterColoredMeshBilateral(const float sigma_s, const float sigma_c, const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points, std::vector<InColor> &in_colors, std::vector<unsigned short> &in_face_res, const std::vector<Vector3li> &in_edge_color_ind, const std::vector<unsigned long> &in_face_color_ind)const{
    // -------------------------------------------------------------------------------
    // -------------------------- bilateral filtering --------------------------------
    // -------------------------------------------------------------------------------

    // params:
    // sigma_s - spatial standard deviation
    // sigma_c - standard deviation in color space
    // for each triangle
        // get 3d coords of vertices
        // compute list of neighbouring triangles within R=3*sigma_s
        // for each sample S
            // get 3D coords of S
            // compute weight and contribution of every sample in list of neighbour triangles, add it to total (use 3D euclidian distance)
    float R2=pow(sigma_s*3,2);  //raduis of spatial window (squared)
    float color_R2 = pow(sigma_c*3,2);  //raduis of color window (squared)
    std::vector<InColor> filtered_out_colors(in_colors);

    #pragma omp parallel for schedule(dynamic)
    for(int32_t tri = 0; tri<in_faces.size(); ++tri) //For every kept triangle
    {
        Vector3f Vref, Vedge1, Vedge2;
        const InTriangle &triangle = in_faces[tri]; 
        Vref = in_points[triangle.ref];
        Vedge1 = in_points[triangle.edge1];
        Vedge2 = in_points[triangle.edge2];
        unsigned short triFaceRes = in_face_res[tri];

        std::vector<unsigned long> triangle_list = getNeighbouringTriangles(in_faces, in_points, tri, sqrt(R2));

        float lambda1,lambda2,lambda3;
        InColor centerColor;
        InColor pointColor;
        long centerIndex;
        int edge_ind;
        //for each sample
        for(int b0 = 0; b0<=triFaceRes; ++b0)
        {
            lambda1=(float)b0/triFaceRes;
            for(int b1=0; b1<=triFaceRes-b0;++b1)
            {
                float weight = 0;
                //OutColor current_color = OutColor(0,0,0);
                float current_red=0;
                float current_blue=0;
                float current_green=0;

                lambda2=(float)b1/triFaceRes;
                lambda3=1-lambda1-lambda2;
                Vector3f center = lambda1*Vref+lambda2*Vedge1+lambda3*Vedge2;
                //get color of center - put this into a function
                centerIndex = getSampleColorIndex(triangle,tri,triFaceRes,b0,b1,in_edge_color_ind,in_face_color_ind, in_colors);
                if(centerIndex>=0)
                {
                    /*
                    std::vector<unsigned long int> center_reds,center_blues,center_greens;
                    int points_count=0;
                    center_reds.clear();
                    center_greens.clear();
                    center_blues.clear();
                    center_reds.reserve(triangle_list.size()*(default_face_res+2)*(default_face_res+1)/2);
                    center_greens.reserve(triangle_list.size()*(default_face_res+2)*(default_face_res+1)/2);
                    center_blues.reserve(triangle_list.size()*(default_face_res+2)*(default_face_res+1)/2);
                    //*/
                    //debugging
                    /*
                    if(b0==triFaceRes||b1==triFaceRes||b0+b1==0){
                        filtered_out_colors[centerIndex]=OutColor(255,255,0);
                    }
                    //*
                    else if(b0==triFaceRes-2 && b1==1)
                    {
                        filtered_out_colors[centerIndex]=OutColor(255,0,0);
                    }
                    else if(b1==triFaceRes-2 && b0==1)
                    {
                        filtered_out_colors[centerIndex]=OutColor(0,255,0);
                    }
                    else if(b0==1 && b1==1)
                    {
                        filtered_out_colors[centerIndex]=OutColor(0,0,255);
                    }
                    
                    else
                    {
                    //*/   
                    centerColor = in_colors[centerIndex];
                    for(int32_t k=0;k<triangle_list.size();++k)
                    {
                        int local_tri = triangle_list[k];
                        unsigned short local_faceRes;
                        local_faceRes = in_face_res[local_tri];
                        float l1,l2,l3;

                        Vector3f Tref, Tedge1, Tedge2;  //get 3 vertices
                        Tref = in_points[in_faces[local_tri].ref];
                        Tedge1 = in_points[in_faces[local_tri].edge1];
                        Tedge2 = in_points[in_faces[local_tri].edge2];
                        for(int l_b0 = 0; l_b0<=local_faceRes; ++l_b0)
                        {
                            l1=(float)l_b0/local_faceRes;
                            for(int l_b1=0; l_b1<=local_faceRes-l_b0;++l_b1)
                            {
                                
                                l2=(float)l_b1/local_faceRes;
                                l3=1-l1-l2;
                                Vector3f sam = l1*Tref + l2*Tedge1 + l3*Tedge2;
                                float dist_vc2;
                                dist_vc2=pow(sam(0)-center(0),2)+pow(sam(1)-center(1),2)+pow(sam(2)-center(2),2);
                                if(dist_vc2<=R2)
                                {
                                    long pointIndex = getSampleColorIndex(in_faces[local_tri],local_tri,local_faceRes,l_b0,l_b1,in_edge_color_ind,in_face_color_ind, in_colors);
                                    if(pointIndex>=0)
                                    {
                                        pointColor = in_colors[pointIndex];
                                        
                                        //get color of sample
                                        if(pointColor!=InColor(0,255,0))
                                        {  //(0,255,0) is the color of voteless samples. We assume it does not appear "in the wild".
                                                                            //Thus, we ignore these neighbours
                                            /*

                                            center_reds.push_back(pointColor(0));
                                            center_greens.push_back(pointColor(1));
                                            center_blues.push_back(pointColor(2));
                                            ++points_count;
                                            //compute weight
                                            /*/
                                            float color_dist2;
                                            if(centerColor==InColor(0,255,0))
                                            {     //if current sample is (0,255,0), do not take color distance (ideally, take distance to average value of all neighbours?)
                                                color_dist2=0;
                                            }
                                            else
                                            {
                                                //color_dist2=0;
                                                color_dist2 = pow((float)centerColor(0)-(float)pointColor(0),2)+pow((float)centerColor(1)-(float)pointColor(1),2)+pow((float)centerColor(2)-(float)pointColor(2),2);    
                                            }
                                            if(color_dist2<=color_R2)
                                            {
                                                float this_weight = exp(-dist_vc2/(2*pow(sigma_s,2))-color_dist2/(2*pow(sigma_c,2)));
                                                weight += this_weight;
                                                current_red+=((float)pointColor(0)*this_weight);
                                                current_green+=((float)pointColor(1)*this_weight);
                                                current_blue+=((float)pointColor(2)*this_weight);
                                            }
                                            //*/
                                        }
                                    }
                                }
                            }
                        }
                    }
                    /*
                    std::sort(center_reds.begin(),center_reds.end());
                    std::sort(center_greens.begin(),center_greens.end());
                    std::sort(center_blues.begin(),center_blues.end());
                    if(points_count==0)
                    {
                        filtered_out_colors[centerIndex]=centerColor;
                    }
                    else
                    {
                        filtered_out_colors[centerIndex]=OutColor(center_reds[points_count/2],center_greens[points_count/2],center_blues[points_count/2]);
                    }
                    //filtered_out_colors[centerIndex]=centerColor;
                    //filtered_out_colors[centerIndex]=OutColor(center_reds[0],center_greens[0],center_blues[0]);
                    /*/
                    if(weight==0)       //no available neighbours
                    {
                        filtered_out_colors[centerIndex]=in_colors[centerIndex];
                    }
                    else
                    {
                        //current_color=OutColor(current_color(0)/weight,current_color(1)/weight,current_color(2)/weight);
                        filtered_out_colors[centerIndex]=InColor(int(current_red/weight),int(current_green/weight),int(current_blue/weight));
                    }
                    //*/
                }
                
            }
        }

    }
    in_colors = filtered_out_colors;
}


template<class InColor, class InTriangle, class InPoint>
void SpaceTimeSampler::filterColoredMeshLoG(const float sigma, const float lambda, const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points, std::vector<InColor> &in_colors, std::vector<unsigned short> &in_face_res, const std::vector<Vector3li> &in_edge_color_ind, const std::vector<unsigned long> &in_face_color_ind)const{
    // -------------------------------------------------------------------------------
    // -------------------------- sharpening filter (laplacian of gaussian) --------------------------------
    // -------------------------------------------------------------------------------

    // params:
    // sigma - spatial standard deviation
    // lambda - strength of the filtering
    // for each triangle
        // get 3d coords of vertices
        // compute list of neighbouring triangles within R=3*sigma_s
        // for each sample S
            // get 3D coords of S
            // compute weight and contribution of every sample in list of neighbour triangles, add it to total (use 3D euclidian distance)
    float R2=pow(sigma*2.5,2);  //raduis of spatial window (squared)
    std::vector<InColor> filtered_out_colors(in_colors);

    #pragma omp parallel for schedule(dynamic)
    for(int32_t tri = 0; tri<in_faces.size(); ++tri) //For every kept triangle
    {
        Vector3f Vref, Vedge1, Vedge2;
        const InTriangle &triangle = in_faces[tri]; 
        Vref = in_points[triangle.ref];
        Vedge1 = in_points[triangle.edge1];
        Vedge2 = in_points[triangle.edge2];
        unsigned short triFaceRes = in_face_res[tri];

        std::vector<unsigned long> triangle_list = getNeighbouringTriangles(in_faces, in_points, tri, sqrt(R2));

        float lambda1,lambda2,lambda3;
        InColor centerColor;
        InColor pointColor;
        long centerIndex;
        int edge_ind;
        //for each sample
        for(int b0 = 0; b0<=triFaceRes; ++b0)
        {
            lambda1=(float)b0/triFaceRes;
            for(int b1=0; b1<=triFaceRes-b0;++b1)
            {
                float weight = 0;
                //OutColor current_color = OutColor(0,0,0);
                float current_red=0;
                float current_blue=0;
                float current_green=0;
                std::vector<long> treatedIndices;
                treatedIndices.clear();
                lambda2=(float)b1/triFaceRes;
                lambda3=1-lambda1-lambda2;
                Vector3f center = lambda1*Vref+lambda2*Vedge1+lambda3*Vedge2;
                //get color of center - put this into a function
                centerIndex = getSampleColorIndex(triangle,tri,triFaceRes,b0,b1,in_edge_color_ind,in_face_color_ind, in_colors);
                if(centerIndex>=0)
                {   
                    centerColor = in_colors[centerIndex];
                    if(centerColor==InColor(0,255,0))
                    {
                        continue;
                    }

                    current_red = float(centerColor(0));
                    current_green = float(centerColor(1));
                    current_blue = float(centerColor(2));
                    int neighboursCount=0;
                    for(int32_t k=0;k<triangle_list.size();++k)
                    {
                        int local_tri = triangle_list[k];
                        unsigned short local_faceRes;
                        local_faceRes = in_face_res[local_tri];
                        float l1,l2,l3;

                        Vector3f Tref, Tedge1, Tedge2;  //get 3 vertices
                        Tref = in_points[in_faces[local_tri].ref];
                        Tedge1 = in_points[in_faces[local_tri].edge1];
                        Tedge2 = in_points[in_faces[local_tri].edge2];
                        for(int l_b0 = 0; l_b0<=local_faceRes; ++l_b0)
                        {
                            l1=(float)l_b0/local_faceRes;
                            for(int l_b1=0; l_b1<=local_faceRes-l_b0;++l_b1)
                            {
                                
                                l2=(float)l_b1/local_faceRes;
                                l3=1-l1-l2;
                                Vector3f sam = l1*Tref + l2*Tedge1 + l3*Tedge2;
                                float dist_vc2;
                                dist_vc2=pow(sam(0)-center(0),2)+pow(sam(1)-center(1),2)+pow(sam(2)-center(2),2);
                                if(dist_vc2<=R2)
                                {
                                    long pointIndex = getSampleColorIndex(in_faces[local_tri],local_tri,local_faceRes,l_b0,l_b1,in_edge_color_ind,in_face_color_ind, in_colors);
                                    if((pointIndex>=0)&&(std::find(treatedIndices.begin(),treatedIndices.end(),pointIndex)==treatedIndices.end()))
                                    {
                                        pointColor = in_colors[pointIndex];
                                        
                                        //get color of sample
                                        if((pointColor!=InColor(0,255,0)))
                                        {  //(0,255,0) is the color of voteless samples. We assume it does not appear "in the wild".
                                                                            //Thus, we ignore these neighbours


                                            if((tri==500)&&(b0==0))
                                            {
                                                log(ALWAYS)<<"triangle neighbour "<<k<<" (tri "<<local_tri<<"), sigma = "<<sigma<<"(R2 = "<<R2<<")"<<endLog();
                                                log(ALWAYS)<<"sample ("<<l_b0<<","<<l_b1<<"), sampleCount: "<<neighboursCount<<", distance = "<<sqrt(dist_vc2)<<endLog();
                                            }
                                            float ratio = dist_vc2/(2*pow(sigma,2));
                                            float this_weight = (1-ratio)*exp(-ratio);
                                            weight += std::abs(this_weight);
                                            current_red+=lambda*((float)pointColor(0)*this_weight);
                                            current_green+=lambda*((float)pointColor(1)*this_weight);
                                            current_blue+=lambda*((float)pointColor(2)*this_weight);
                                            ++neighboursCount;
                                            treatedIndices.push_back(pointIndex);
                                            //*/
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    if(weight==0)       //no available neighbours
                    {
                        filtered_out_colors[centerIndex]=in_colors[centerIndex];
                    }
                    else
                    {
                        current_red=std::min(std::max(0.0f,current_red),255.1f);
                        current_green=std::min(std::max(0.0f,current_green),255.1f);
                        current_blue=std::min(std::max(0.0f,current_blue),255.1f);
                        filtered_out_colors[centerIndex]=InColor(int(current_red),int(current_green),int(current_blue));
                    }
                    //*/
                }
                
            }
        }

    }
    in_colors = filtered_out_colors;
}


template<class InColor, typename T>
void SpaceTimeSampler::sharpenColor(const float lambda, std::vector<InColor> &in_colors, std::vector<std::list<T> > &adjList)const{
    // -------------------------------------------------------------------------------
    // -------------------------- laplacian filtering --------------------------------
    // -------------------------------------------------------------------------------
    //std::vector<float> filter = std::vector<float>(in_colors.size(),0.0f);
    std::vector<float> filterY = std::vector<float>(in_colors.size(),0.0f);
    std::vector<float> filterCb = std::vector<float>(in_colors.size(),0.0f);
    std::vector<float> filterCr = std::vector<float>(in_colors.size(),0.0f);
    for(int i=0;i<in_colors.size();++i)
    {
        int nn = adjList[i].size();
        if(nn>0)
        {
            //float fil=float(in_colors[i](0)+in_colors[i](1)+in_colors[i](2))/3;
            InColor tempCol = in_colors[i];
            rgbToYcc(tempCol);
            float filY=tempCol(0);
            float filCb=tempCol(1);
            float filCr=tempCol(2);
            std::list<int32_t> testList = adjList[i];
            for (std::list<int32_t>::iterator listIter = testList.begin(); listIter != testList.end(); ++listIter)
            {
                int32_t myInd = (*listIter);
                InColor neighCol = in_colors[myInd];
                rgbToYcc(neighCol);
                //fil-=float(in_colors[myInd](0)+in_colors[myInd](1)+in_colors[myInd](2))/(3*nn);
                filY-=float(neighCol(0))/nn;
                filCb-=float(neighCol(1))/nn;
                filCr-=float(neighCol(2))/nn;
            }
            
            //filter[i]=fil;
            filterY[i]=filY;
            filterCb[i]=filCb;
            filterCr[i]=filCr;

        }
    }
    for(int i=0;i<in_colors.size();++i)
    {
        int nn = adjList[i].size();
        if(nn>0)
        {
            if((in_colors[i](0)<=32)&&(in_colors[i](1)==255)&&(in_colors[i](2)==255))
            {
                log(ALWAYS)<<"WARNING! Potential edge res pixel: "<<i<<", res = "<<in_colors[i](0)<<endLog();
            }
            // in_colors[i](0) = (unsigned int)(std::max(0.0f,std::min(255.1f,float(in_colors[i](0))+lambda*filter[i])));
            // in_colors[i](1) = (unsigned int)(std::max(0.0f,std::min(255.1f,float(in_colors[i](1))+lambda*filter[i])));
            // in_colors[i](2) = (unsigned int)(std::max(0.0f,std::min(255.1f,float(in_colors[i](2))+lambda*filter[i])));
            rgbToYcc(in_colors[i]);
            in_colors[i](0) = (unsigned int)(std::max(0.0f,std::min(255.1f,float(in_colors[i](0))+lambda*filterY[i])));
            in_colors[i](1) = (unsigned int)(std::max(0.0f,std::min(255.1f,float(in_colors[i](1))+lambda*filterCb[i])));
            in_colors[i](2) = (unsigned int)(std::max(0.0f,std::min(255.1f,float(in_colors[i](2))+lambda*filterCr[i])));
            yccToRgb(in_colors[i]);
        }
    }

}

template<class InColor, class InTriangle, class InPoint>
void SpaceTimeSampler::bleedColor(float sigma_s, const std::vector<InTriangle> &in_faces, const std::vector<InPoint> &in_points, std::vector<InColor> &in_colors, std::vector<unsigned short> &in_face_res, const std::vector<Vector3li> &in_edge_color_ind, const std::vector<unsigned long> &in_face_color_ind)const{
    //copied from bilateral filtering function:
    //for each uncolored sample, take average of neighbouring colored samples
    // params:
    // sigma_s - spatial standard deviation
    // for each triangle
        // get 3d coords of vertices
        // compute list of neighbouring triangles within R=3*sigma_s
        // for each uncolored sample S
            // get 3D coords of S
            // compute weight and contribution of every sample in list of neighbour triangles, add it to total (use 3D euclidian distance)
    float R2=pow(sigma_s*3,2);  //raduis of spatial window (squared)
    std::vector<InColor> filtered_out_colors(in_colors);

    int colorlessSamples = 1;   //number of remaining uncolored samples
    int correctedSamples;       //number of newly colored samples during the pass
    int passNumber = 0;
    while(colorlessSamples>0)   //If we corrected some voteless samples, we do another pass:
                                //Typically, we will use a very small sigma, and samples surrounded by other voteless samples might not be corrected right away.
                                //In practice, I suppose 1 or 2 passes will suffice, since face resolution depends on the number of samples per triangle
    {
        colorlessSamples=0;
        correctedSamples=0;
        log(ALWAYS)<<"Pass number "<<passNumber<<endLog();
        ++passNumber;
        #pragma omp parallel for schedule(dynamic)
        for(int32_t tri = 0; tri<in_faces.size(); ++tri) //For every kept triangle
        {
            Vector3f Vref, Vedge1, Vedge2;
            const InTriangle &triangle = in_faces[tri]; 
            Vref = in_points[triangle.ref];
            Vedge1 = in_points[triangle.edge1];
            Vedge2 = in_points[triangle.edge2];
            unsigned short triFaceRes = in_face_res[tri];
            bool hasVotelessSamples = false;
            float lambda1,lambda2,lambda3;
            InColor centerColor;
            long centerIndex;
            for(int b0 = 0; b0<=triFaceRes; ++b0)       //for every sample, check if it is colored
            {
                lambda1=(float)b0/triFaceRes;
                for(int b1=0; b1<=triFaceRes-b0;++b1)
                {
                    lambda2=(float)b1/triFaceRes;
                    lambda3=1-lambda1-lambda2;
                    Vector3f center = lambda1*Vref+lambda2*Vedge1+lambda3*Vedge2;
                    //get color of center
                    centerIndex = getSampleColorIndex(triangle,tri,triFaceRes,b0,b1,in_edge_color_ind,in_face_color_ind, in_colors);
                    if(centerIndex>=0)
                    {
                        centerColor = in_colors[centerIndex];
                        if(centerColor==InColor(0,255,0))
                        {
                            hasVotelessSamples = true;
                            goto getNeighbours;         //we found an uncolored sample: compute neighbourhood of triangle and loop over samples again
                        }
                    }
                }
            }
            getNeighbours:
            if(!hasVotelessSamples)     //do not bother checking neighbourhood if triangle is fully colored
                continue;               //go to next triangle
            std::vector<unsigned long> triangle_list = getNeighbouringTriangles(in_faces, in_points, tri, sqrt(R2));
            InColor pointColor;
            int edge_ind;
            //for each sample
            for(int b0 = 0; b0<=triFaceRes; ++b0)   //we know there is at least one uncolored sample. Loop over the samples again.
            {
                lambda1=(float)b0/triFaceRes;
                for(int b1=0; b1<=triFaceRes-b0;++b1)
                {
                    

                    lambda2=(float)b1/triFaceRes;
                    lambda3=1-lambda1-lambda2;
                    Vector3f center = lambda1*Vref+lambda2*Vedge1+lambda3*Vedge2;
                    //get color of center
                    centerIndex = getSampleColorIndex(triangle,tri,triFaceRes,b0,b1,in_edge_color_ind,in_face_color_ind, in_colors);
                    if(centerIndex>=0)
                    {
                        centerColor = in_colors[centerIndex];
                        if(centerColor==InColor(0,255,0))   //if sample is uncolored
                        {
                            ++colorlessSamples;
                            float weight = 0;
                            //OutColor current_color = OutColor(0,0,0);
                            float current_red=0;
                            float current_blue=0;
                            float current_green=0;
                            for(int32_t k=0;k<triangle_list.size();++k)     //for each triangle in neighbourhood
                            {
                                int local_tri = triangle_list[k];
                                unsigned short local_faceRes;
                                local_faceRes = in_face_res[local_tri];
                                float l1,l2,l3;

                                Vector3f Tref, Tedge1, Tedge2;  //get 3 vertices
                                Tref = in_points[in_faces[local_tri].ref];
                                Tedge1 = in_points[in_faces[local_tri].edge1];
                                Tedge2 = in_points[in_faces[local_tri].edge2];
                                for(int l_b0 = 0; l_b0<=local_faceRes; ++l_b0)          //for each sample in triangle
                                {
                                    l1=(float)l_b0/local_faceRes;
                                    for(int l_b1=0; l_b1<=local_faceRes-l_b0;++l_b1)
                                    {
                                        
                                        l2=(float)l_b1/local_faceRes;
                                        l3=1-l1-l2;
                                        Vector3f sam = l1*Tref + l2*Tedge1 + l3*Tedge2;
                                        float dist_vc2;
                                        dist_vc2=pow(sam(0)-center(0),2)+pow(sam(1)-center(1),2)+pow(sam(2)-center(2),2);
                                        if(dist_vc2<=R2)                                //if within the radius
                                        {
                                            long pointIndex = getSampleColorIndex(in_faces[local_tri],local_tri,local_faceRes,l_b0,l_b1,in_edge_color_ind,in_face_color_ind, in_colors);
                                            if(pointIndex>=0)
                                            {
                                                pointColor = in_colors[pointIndex];
                                                
                                                //get color of sample
                                                if(pointColor!=InColor(0,255,0))
                                                {  //(0,255,0) is the color of voteless samples. We assume it does not appear "in the wild".
                                                                                    //Thus, we ignore these neighbours
                                                    
                                                    float this_weight = exp(-dist_vc2/(2*pow(sigma_s,2)));
                                                    weight += this_weight;
                                                    current_red+=((float)pointColor(0)*this_weight);
                                                    current_green+=((float)pointColor(1)*this_weight);
                                                    current_blue+=((float)pointColor(2)*this_weight);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if(weight>0)
                            {
                                filtered_out_colors[centerIndex]=InColor(int(current_red/weight),int(current_green/weight),int(current_blue/weight));
                                ++correctedSamples;
                            }
                        }
                    }
                    
                }
            }

        }
        in_colors = filtered_out_colors;        //update colors
        if(correctedSamples==0)         //if no samples were corrected, make the radius (exponentially) bigger
        {
            sigma_s*=2;
            R2=pow(sigma_s*3,2);
        }
        log(ALWAYS)<<"colorlessSamples: "<<colorlessSamples<<endLog();
        log(ALWAYS)<<"correctedSamples: "<<correctedSamples<<endLog();
        log(ALWAYS)<<"squared radius: "<<R2<<endLog();
        if(sigma_s>100)                 //arbitrary stop criterion, in case some samples are unreachable
            return;
    }
}

template<class InColor>
void SpaceTimeSampler::colorWeightedMedianVote(const std::vector<std::vector<InColor> > &in_color_votes, const std::vector<std::vector<float> > &in_votes_weight, std::vector<InColor> &out_colors)const{
    log(ALWAYS)<<"[SpaceTimeSampler] : weighted median coloring"<<endLog();
    #pragma omp parallel for schedule(dynamic)
    for (long samp = 0; samp < out_colors.size(); ++samp)     //for each color sample
    {
        if(in_color_votes[samp].empty())
        {
            out_colors[samp] = InColor(0,255,0);
        }
        else
        {
            // -------------------------------------------------------------------------------
            // ----------------- weigthed median version (RGB or YUV) ------------------------
            // -------------------------------------------------------------------------------
            std::vector<short int> reds,greens,blues;
            //std::vector<float> cYs,cUs,cVs;
            std::vector<size_t> sorted_idx;
            float red, green, blue;
            //float cY,cU,cV;
            float total_weight=0;
            float current_weight=0;
            size_t s_idx;

            for(unsigned int i = 0; i < in_color_votes[samp].size(); ++i)    //for each vote for this sample
            {
                reds.push_back(in_color_votes[samp][i](0));
                greens.push_back(in_color_votes[samp][i](1));
                blues.push_back(in_color_votes[samp][i](2));
                //cYs.push_back(0.299*(float)in_color_votes[samp][i](0)+0.587*(float)in_color_votes[samp][i](1)+0.114*(float)in_color_votes[samp][i](2));
                //cUs.push_back(((float)in_color_votes[samp][i](2)-cYs[cYs.size()-1])*0.565);
                //cVs.push_back(((float)in_color_votes[samp][i](0)-cYs[cYs.size()-1])*0.713);
                total_weight+=in_votes_weight[samp][i];
            }
            /*
            sorted_idx = sort_indexes(cYs);
            std::sort(cYs.begin(),cYs.end());
            current_weight=0;
            s_idx=0;
            while (current_weight<total_weight/2)
            {
                current_weight+= in_votes_weight[samp][sorted_idx[s_idx]];
                cY=cYs[s_idx];
                s_idx++;
            }
            sorted_idx = sort_indexes(cUs);
            std::sort(cUs.begin(),cUs.end());
            current_weight=0;
            s_idx=0;
            while (current_weight<total_weight/2)
            {
                current_weight+= in_votes_weight[samp][sorted_idx[s_idx]];
                cU=cUs[s_idx];
                s_idx++;
            }
            sorted_idx = sort_indexes(cVs);
            std::sort(cVs.begin(),cVs.end());
            current_weight=0;
            s_idx=0;
            while (current_weight<total_weight/2)
            {
                current_weight+= in_votes_weight[samp][sorted_idx[s_idx]];
                cV=cVs[s_idx];
                s_idx++;
            }

            red = std::max(0.0,std::min(cY + 1.403*cV,255.0));
            green = std::max(0.0,std::min(cY - 0.714 *cV - 0.344 * cU,255.0));
            blue = std::max(0.0,std::min(cY + 1.770*cU,255.0));
            out_colors[samp] = (OutColor((int)red,(int)green,(int)blue));
            //*/
            //*
            sorted_idx = sort_indexes(reds);
            std::sort(reds.begin(),reds.end());
            current_weight=0;
            s_idx=0;
            while (current_weight<total_weight/2)
            {
                current_weight+= in_votes_weight[samp][sorted_idx[s_idx]];
                red=reds[s_idx];
                s_idx++;
            }
            sorted_idx = sort_indexes(greens);
            std::sort(greens.begin(),greens.end());
            current_weight=0;
            s_idx=0;
            while (current_weight<total_weight/2)
            {
                current_weight+= in_votes_weight[samp][sorted_idx[s_idx]];
                green=greens[s_idx];
                s_idx++;
            }
            sorted_idx = sort_indexes(blues);
            std::sort(blues.begin(),blues.end());
            current_weight=0;
            s_idx=0;
            while (current_weight<total_weight/2)
            {
                current_weight+= in_votes_weight[samp][sorted_idx[s_idx]];
                blue=blues[s_idx];
                s_idx++;
            }
            out_colors[samp] = (InColor((int)red,(int)green,(int)blue));
            //TODO: Artificially lower the green component in some way, to account for green reflection in the kinovis platform?
        }
    }
}

template<class InColor>
void SpaceTimeSampler::colorMedianVote(const std::vector<std::vector<InColor> > &in_color_votes, std::vector<InColor> &out_colors)const{
    log(ALWAYS)<<"[SpaceTimeSampler] : median coloring"<<endLog();
    #pragma omp parallel for schedule(dynamic)
    for (long samp = 0; samp < out_colors.size(); ++samp)     //for each color sample
    {
        if(in_color_votes[samp].empty())
        {
            out_colors[samp] = InColor(0,255,0);
        }
        else
        {
            // -------------------------------------------------------------------------------
            // -------------------------- median version -------------------------------------
            // -------------------------------------------------------------------------------
            std::vector<short int> reds,greens,blues;
            for(unsigned int i = 0; i < in_color_votes[samp].size(); ++i)    //for each vote for this sample
            {
                reds.push_back(in_color_votes[samp][i](0));
                greens.push_back(in_color_votes[samp][i](1));
                blues.push_back(in_color_votes[samp][i](2));
            }
            std::sort(reds.begin(),reds.end());
            std::sort(greens.begin(),greens.end());
            std::sort(blues.begin(),blues.end());
            out_colors[samp] = InColor(reds[in_color_votes[samp].size()/2],greens[in_color_votes[samp].size()/2],blues[in_color_votes[samp].size()/2]);
            //TODO: Artificially lower the green component in some way, to account for green reflection in the kinovis platform?
        }
    }   
}

template<class InColor>
void SpaceTimeSampler::colorWeightedMeanVote(const std::vector<std::vector<InColor> > &in_color_votes, const std::vector<std::vector<float> > &in_votes_weight, std::vector<InColor> &out_colors)const{
    log(ALWAYS)<<"[SpaceTimeSampler] : weighted mean coloring"<<endLog();
    #pragma omp parallel for schedule(dynamic)
    for (long samp = 0; samp < out_colors.size(); ++samp)     //for each color sample
    {
        if(in_color_votes[samp].empty())
        {
            out_colors[samp] = InColor(0,255,0);
        }
        else
        {
            // -------------------------------------------------------------------------------
            // ------------------------ weighted mean version --------------------------------
            // -------------------------------------------------------------------------------
            float red=0;
            float green=0;
            float blue=0;
            float total_weight=0;
            for(unsigned int i = 0; i < in_color_votes[samp].size(); ++i)    //for each vote for this sample
            {
                red+=in_color_votes[samp][i](0)*in_votes_weight[samp][i];
                green+=in_color_votes[samp][i](1)*in_votes_weight[samp][i];
                blue+=in_color_votes[samp][i](2)*in_votes_weight[samp][i];
                total_weight+=in_votes_weight[samp][i];
            }

            //if ((green>red)&&(green>blue))
            //{
            //    green = std::max(blue,red);
            //}

            //deactivate green correction (for tests)
            /*
            if (green>(2*red+blue)/3)
            {
                green = (2*red+blue)/3;
            }
            //*/

            //float gray = 0.2126*red+0.7152*green+0.0722*blue;
            //green = std::min(green,gray);
            
            //green = red*(exp(gr)-1)/exp(gr);

            out_colors[samp] = InColor(red/total_weight,green/(total_weight),blue/total_weight);

        }
    }
}

template<class InColor>
void SpaceTimeSampler::colorTVNormVote(const std::vector<std::vector<InColor> > &in_color_votes, const std::vector<std::vector<float> > &in_votes_weight, /*const cv::Mat adj_mat,*/ const std::vector<std::list<int32_t> > &samples_adj, std::vector<InColor> &out_colors)const{
    log(ALWAYS)<<"[SpaceTimeSampler] : TV Norm coloring"<<endLog();
    float gamma = 25;
    int iter_num = 100;
    float lambda = 0.2;
    std::vector<InColor> temp_colors(out_colors);
    std::vector<InColor> temp_colors2(out_colors);
    //std::vector<float> red_c;
    //std::vector<float> green_c;
    //std::vector<float> blue_c;
    //cv::Mat red_c (out_colors.size(),1,CV_32FC1);
    //cv::Mat green_c (out_colors.size(),1,CV_32FC1);
    //cv::Mat blue_c (out_colors.size(),1,CV_32FC1);
    /*for(int samp=0;samp<out_colors.size();++samp)
    {
        red_c.at<float>(samp,0) = out_colors[samp](0);
        green_c.at<float>(samp,0) = out_colors[samp](1);
        blue_c.at<float>(samp,0) = out_colors[samp](2);
        //red_c[samp]=out_colors[samp](0);
        //green_c[samp]=out_colors[samp](1);
        //blue_c[samp]=out_colors[samp](2);
    }*/
    //majorization-minimization algorithm based on http://eeweb.poly.edu/iselesni/lecture_notes/TVDmm/TVDmm.pdf (no)
    //cv::Mat DtD = adj_mat*adj_mat.t();
    for (int it=0;it<iter_num;++it)
    {
        log(ALWAYS)<<"iter "<<it<<endLog();
        /*cv::Mat test = adj_mat * red_c;
        test = lambda * cv::Mat::diag(test);
        test = test + DtD;
        test = test.inv();
        red_c = red_c - adj_mat.t() * test * adj_mat * red_c;
        //temp_colors = out_colors - adj_mat.t()*(lambda* );
        */
        #pragma omp parallel for schedule(dynamic)
        for(int samp=0;samp<out_colors.size();++samp)
        {

            float df_r =0;
            float df_g =0;
            float df_b =0;
            float nbrNeighbours = 0.0f;
            std::list<int32_t> testList = samples_adj[samp];
            for (std::list<int32_t>::iterator listIter = testList.begin(); listIter != testList.end(); ++listIter)
            {
                float diff = float(temp_colors[samp](0))-float(temp_colors[(*listIter)](0));
                if (diff>0)
                {
                    df_r+=lambda;
                }else if(diff<0)
                {
                    df_r-=lambda;
                }
                diff = float(temp_colors[samp](1))-float(temp_colors[(*listIter)](1));
                if (diff>0)
                {
                    df_g+=lambda;
                }else if(diff<0)
                {
                    df_g-=lambda;
                }
                diff = float(temp_colors[samp](2))-float(temp_colors[(*listIter)](2));
                if (diff>0)
                {
                    df_b+=lambda;
                }else if(diff<0)
                {
                    df_b-=lambda;
                }
                ++nbrNeighbours;
            }
            if(nbrNeighbours>0.1f)
            {
                df_r/= nbrNeighbours;
                df_g/= nbrNeighbours;
                df_b/= nbrNeighbours;
            }
            /*float min_diff = 1;
            float min_diff_r = 0;
            float min_diff_g = 0;
            float min_diff_b = 0;
            for(int vote=0;vote<in_color_votes[samp].size();++vote)
            {
                float diff_r = (float(temp_colors[samp](0))-float(in_color_votes[samp][vote](0)))/255;
                float diff_g = (float(temp_colors[samp](1))-float(in_color_votes[samp][vote](1)))/255;
                float diff_b = (float(temp_colors[samp](2))-float(in_color_votes[samp][vote](2)))/255;
                if(diff_r*diff_r+diff_g*diff_g+diff_b*diff_b<min_diff)
                {
                    min_diff = diff_r*diff_r+diff_g*diff_g+diff_b*diff_b;
                    min_diff_r = diff_r;
                    min_diff_g = diff_g;
                    min_diff_b = diff_b;
                }
                /*if(std::abs(float(temp_colors[samp](0))-float(in_color_votes[samp][vote](0)))/255<std::abs(max_diff_r))
                    min_diff_r = (float(temp_colors[samp](0))-float(in_color_votes[samp][vote](0)))/255;
                if(std::abs(float(temp_colors[samp](1))-float(in_color_votes[samp][vote](1)))/255<std::abs(max_diff_g))
                    min_diff_g = (float(temp_colors[samp](1))-float(in_color_votes[samp][vote](1)))/255;
                if(std::abs(float(temp_colors[samp](2))-float(in_color_votes[samp][vote](2)))/255<std::abs(max_diff_b))
                    min_diff_b = (float(temp_colors[samp](2))-float(in_color_votes[samp][vote](2)))/255;*-/
            }
            df_r+=min_diff_r;
            df_g+=min_diff_g;
            df_b+=min_diff_b;*/
            df_r+= (float(temp_colors[samp](0))-float(out_colors[samp](0)))/255;
            df_g+= (float(temp_colors[samp](1))-float(out_colors[samp](1)))/255;
            df_b+= (float(temp_colors[samp](2))-float(out_colors[samp](2)))/255;

            temp_colors2[samp](0) = int(std::max(0.0f,std::min(float(255.0),float(temp_colors2[samp](0))-gamma * df_r)));
            temp_colors2[samp](1)= int(std::max(0.0f,std::min(float(255.0),float(temp_colors2[samp](1))-gamma * df_g)));
            temp_colors2[samp](2) = int(std::max(0.0f,std::min(float(255.0),float(temp_colors2[samp](2))-gamma * df_b)));
            
        }

        temp_colors = temp_colors2;
        //lambda-= lambda/iter_num;
    }
    out_colors = temp_colors;

}

template<class InColor>
void SpaceTimeSampler::filterCameraVotes(std::vector<std::vector<InColor> > &in_color_votes, std::vector<std::vector<float> > &in_votes_weight, const int kept_votes_number)const{
    log(ALWAYS)<<"[SpaceTimeSampler] : filtering camera votes"<<endLog();
    int passed=0;
    int treated=0;
    #pragma omp parallel for schedule(dynamic)
    for (long samp = 0; samp < in_color_votes.size(); ++samp)     //for each color sample
    {

        if(in_votes_weight[samp].size()<=kept_votes_number)      //if there are less votes than the number we want to keep, do nothing
        {
            ++passed;
            continue;
        }

        std::vector<size_t> sorted_idx;
        sorted_idx = sort_indexes(in_votes_weight[samp]);
        int s = in_votes_weight[samp].size();
        std::vector<InColor> new_votes(kept_votes_number);
        std::vector<float> new_weights(kept_votes_number);
        for(int k=0; k<kept_votes_number; k++)
        {
            new_votes[k]=in_color_votes[samp][sorted_idx[s-k-1]];    //it's sorted from lowest value to highest
            new_weights[k]=in_votes_weight[samp][sorted_idx[s-k-1]];
        }
        in_color_votes[samp]=new_votes;
        in_votes_weight[samp]=new_weights;
        ++treated;
    }
    log(ALWAYS)<<"[SpaceTimeSampler] : "<<passed<<" samples untouched, "<<treated<<" samples with truncated votes"<<endLog();
}

template<class InTriangle, class InPoint>
cv::Mat SpaceTimeSampler::generateTextureMap(const unsigned int width, const unsigned int height, const std::vector<InTriangle> &in_faces, const std::vector<Vector2f> &in_tex_coords, const std::vector<Vector3uli> &in_tex_indices, const std::vector<InPoint> &in_points, const std::vector<std::vector<size_t> > &vertices_cam, const std::vector<float> &camera_K, std::vector<cv::Mat> &cam_tri_ind, std::string outputPath)const{

    cv::Mat texMap(width,height,CV_32FC3, cv::Vec3f(0.0f,0.0f,0.0f));  //initialize texture map
    cv::Mat texWeight(width,height,CV_32FC1,0.0f);
    cv::Mat finalTexMap(width,height,CV_8UC3);
    
    for(unsigned long tri=0;tri<in_faces.size();++tri)
    {
        std::vector<int32_t> triverts(3);
        InTriangle triangle = in_faces[tri];
        triverts[0] = triangle.ref;
        triverts[1] = triangle.edge1;
        triverts[2] = triangle.edge2;
        Vector2f tV1, tV2, tV3, sup_left, inf_right;         //texture coordinates of vertices
        
        tV1 = in_tex_coords[in_tex_indices[tri](0)];
        tV2 = in_tex_coords[in_tex_indices[tri](1)];
        tV3 = in_tex_coords[in_tex_indices[tri](2)];

        tV1(0) = tV1(0)*width-0.5f;
        tV1(1)= tV1(1)*height-0.5f;
        tV2(0) = tV2(0)*width-0.5f;
        tV2(1)= tV2(1)*height-0.5f;
        tV3(0) = tV3(0)*width-0.5f;
        tV3(1)= tV3(1)*height-0.5f;

        sup_left = tV1;
        inf_right = tV1;
        //Get bounding box
        //Sup left
        if(tV2(0) < sup_left(0))
            sup_left(0) = tV2(0);
        if(tV3(0) < sup_left(0))
            sup_left(0) = tV3(0);

        if(tV2(1) < sup_left(1))
            sup_left(1) = tV2(1);
        if(tV3(1) < sup_left(1))
            sup_left(1) = tV3(1);

        //Inf Right
        if(tV2(0) > inf_right(0))
            inf_right(0) = tV2(0);
        if(tV3(0) > inf_right(0))
            inf_right(0) = tV3(0);

        if(tV2(1) > inf_right(1))
            inf_right(1) = tV2(1);
        if(tV3(1) > inf_right(1))
            inf_right(1) = tV3(1);
        double x,y,x1,x2,x3,y1,y2,y3,lambda1,lambda2,lambda3,depth1,depth2,depth3,previous_depth,new_depth;
        x1 = tV1(1);     y1 = tV1(0);
        x2 = tV2(1);   y2 = tV2(0);
        x3 = tV3(1);   y3 = tV3(0);

        for(unsigned long int py = std::max(0.0,floor(sup_left(0))) ; py <= std::min(ceil(inf_right(0)),double(width-1)); ++py)
        {
            for(unsigned long int px = std::max(0.0,floor(sup_left(1))); px <= std::min(ceil(inf_right(1)),double(height-1)); ++px)
            {
                
                x = (double)px;
                y = (double)py;

                lambda1 = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3));
                lambda2 = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3));
                lambda3 = 1.0 - lambda1 - lambda2;
                if(lambda1 >= 0.0 && lambda2 >= 0.0 && lambda3 >= 0.0)
                {
                    Vector3f myBarycoords = Vector3f(lambda1,lambda2,lambda3);
                    //Inside triangle
                    for(int verNum=0;verNum<3;++verNum)     //for each vertex
                    {
                        for(int myCam=0;myCam<vertices_cam[triverts[verNum]].size();++myCam)
                        {
                            int camNum = vertices_cam[triverts[verNum]][myCam];
                            //std::vector<Vector3f> testlol;
                            Vector3ui inputColor;
                            bool is_safe = getSurfacePointColor(triangle,in_points,myBarycoords,camNum, inputColor);
                            // bool is_safe = getSurfacePointColorWVis(triangle, tri, in_points,myBarycoords,camNum, inputColor, cam_tri_ind);
                            if(is_safe)
                            {
                                cv::Vec3f cvColor;
                                cvColor(2) = myBarycoords(verNum)*std::min(255.0f,std::max(0.0f,float(inputColor(0))/camera_K[camNum]));
                                cvColor(1) = myBarycoords(verNum)*std::min(255.0f,std::max(0.0f,float(inputColor(1))/camera_K[camNum]));
                                cvColor(0) = myBarycoords(verNum)*std::min(255.0f,std::max(0.0f,float(inputColor(2))/camera_K[camNum]));
                                texMap.at<cv::Vec3f>(height-1-px,py) += cvColor;
                                texWeight.at<float>(height-1-px,py)+=myBarycoords(verNum);
                            }
                        }
                    }
                }
            }
        }
    }
    //normalize votes, and convert to 1 byte
    cv::Mat channel[3];
    split(texMap,channel);
    channel[0] = channel[0]/texWeight;
    channel[1] = channel[1]/texWeight;
    channel[2] = channel[2]/texWeight;

    // Threshold green channel (copied from weighted mean function)
    cv::Mat thresholdedGreen = (2*channel[2]+channel[0])/3;
    cv::Mat greenMask = (channel[1]>thresholdedGreen);
    thresholdedGreen.copyTo(channel[1],greenMask);
    

    //Bleed color (dilate texture)
    cv::Mat silMask = (texWeight>0);

    int treatedTexels;

    int total_iter=10;
    for(int iter=0;iter<total_iter;++iter)
    {
        treatedTexels=0;

        silMask = (texWeight>0);
        cv::Mat dilatedSil;
        cv::dilate(silMask,dilatedSil,cv::Mat(),cv::Point(-1,-1),1);
        cv::Mat contour = (dilatedSil!=silMask);

        #pragma omp parallel for schedule(dynamic)
        for(int i=0;i<silMask.rows;++i)
        {
            for(int j=0;j<silMask.cols;++j)
            {
                if(contour.at<bool>(i,j))
                {
                    int imin=std::max(0,i-1);
                    int imax=std::min(i+1,contour.rows-1);
                    int jmin=std::max(0,j-1);
                    int jmax=std::min(j+1,contour.cols-1);
                    int weight=0;
                    float tempValR=0.0f;
                    float tempValG=0.0f;
                    float tempValB=0.0f;
                    for(int k=imin;k<=imax;++k)
                    {
                        for(int l=jmin;l<=jmax;++l)
                        {
                            if(silMask.at<bool>(k,l))
                            {
                                ++weight;
                                tempValB+=channel[0].at<float>(k,l);
                                tempValG+=channel[1].at<float>(k,l);
                                tempValR+=channel[2].at<float>(k,l);
                            }
                        }
                    }
                    if(weight>=2)
                    {
                        channel[0].at<float>(i,j)=tempValB/weight;
                        channel[1].at<float>(i,j)=tempValG/weight;
                        channel[2].at<float>(i,j)=tempValR/weight;
                        texWeight.at<float>(i,j)=1;
                        #pragma omp critical
                        {
                            ++treatedTexels;
                        }
                    }
                }
            }
        }
    }

    silMask = (texWeight>0);

    merge(channel,3,texMap);
    texMap.convertTo(finalTexMap,CV_8UC3);
    return finalTexMap;
}

template<class InTriangle, class InPoint>
void SpaceTimeSampler::setPixelsWhereTrianglesProjectCloser(const std::vector<InTriangle> &triangles, const std::vector<InPoint> &points, cv::Mat &out_image, const Camera &cam, const float cleaning_factor)const{
    Vector2f sup_left,inf_right;
    Vector2f ref_coords,edge1_coords,edge2_coords;
    const Vector3f &camera_center = cam.getPosition();
    for(int32_t triangle_idx = 0; triangle_idx < triangles.size(); ++triangle_idx)
    {
        const InTriangle &tri = triangles[triangle_idx];
        
        //Project every point
        cam.getTextureCoords(points[tri.ref],ref_coords);
        cam.getTextureCoords(points[tri.edge1],edge1_coords);
        cam.getTextureCoords(points[tri.edge2],edge2_coords);

        ref_coords *= cleaning_factor;
        edge1_coords *= cleaning_factor;
        edge2_coords *= cleaning_factor;

        sup_left = ref_coords;
        inf_right = ref_coords;

        //Sup left
        if(edge1_coords(0) < sup_left(0))
            sup_left(0) = edge1_coords(0);
        if(edge2_coords(0) < sup_left(0))
            sup_left(0) = edge2_coords(0);

        if(edge1_coords(1) < sup_left(1))
            sup_left(1) = edge1_coords(1);
        if(edge2_coords(1) < sup_left(1))
            sup_left(1) = edge2_coords(1);

        //Inf Right
        if(edge1_coords(0) > inf_right(0))
            inf_right(0) = edge1_coords(0);
        if(edge2_coords(0) > inf_right(0))
            inf_right(0) = edge2_coords(0);

        if(edge1_coords(1) > inf_right(1))
            inf_right(1) = edge1_coords(1);
        if(edge2_coords(1) > inf_right(1))
            inf_right(1) = edge2_coords(1);

        //Fill area if dist is inferior to already registered
        for(unsigned long int py = floor(sup_left(0)) ; py < ceil(inf_right(0)); ++py)
            for(unsigned long int px = floor(sup_left(1)); px < ceil(inf_right(1)); ++px)
            {
                double x,y,x1,x2,x3,y1,y2,y3,lambda1,lambda2,depth1,depth2,depth3,previous_depth,new_depth;
                x = (double)px;
                y = (double)py;
                //Recover depth of the triangle vertices
                depth1 = Vector3f(points[tri.ref] - camera_center).norm();
                depth2 = Vector3f(points[tri.edge1] - camera_center).norm();
                depth3 = Vector3f(points[tri.edge2] - camera_center).norm();

                //Compute barycentric coordinates

                x1 = ref_coords(1);     y1 = ref_coords(0);
                x2 = edge1_coords(1);   y2 = edge1_coords(0);
                x3 = edge2_coords(1);   y3 = edge2_coords(0);

                lambda1 = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)) ;
                lambda2 = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3));


                if( lambda1 <= 1.0 && lambda2 <= 1.0 && lambda1 >= 0.0 && lambda2 >= 0.0 && 1.0 - lambda1 - lambda2 >= 0.0 && 1.0 - lambda1 - lambda2 <= 1.0 )
                {
                    //Inside triangle
                    new_depth = lambda1 * depth1 + lambda2 * depth2 + (1.0 - lambda1 - lambda2)* depth3 ;
                    //get pixel value
                    if(px > 0 && px < (IMG_WIDTH*cleaning_factor) && py > 0 && py < IMG_HEIGHT*cleaning_factor)
                    {
                        int32_t &previous_triangle = out_image.at<int32_t>(py,px);
                        int32_t new_idx = triangle_idx;

                        if(previous_triangle > 0)
                        {
                            depth1 = Vector3f(points[triangles[previous_triangle-1].ref] - camera_center).norm();
                            depth2 = Vector3f(points[triangles[previous_triangle-1].edge1] - camera_center).norm();
                            depth3 = Vector3f(points[triangles[previous_triangle-1].edge2] - camera_center).norm();

                            Vector2f previous_ref_coords, previous_edge1_coords,previous_edge2_coords;
                            cam.getTextureCoords(points[triangles[previous_triangle-1].ref],previous_ref_coords);
                            cam.getTextureCoords(points[triangles[previous_triangle-1].edge1],previous_edge1_coords);
                            cam.getTextureCoords(points[triangles[previous_triangle-1].edge2],previous_edge2_coords);

                            previous_ref_coords *= cleaning_factor;
                            previous_edge1_coords *= cleaning_factor;
                            previous_edge2_coords *= cleaning_factor;


                            //Check if this one is closer than previous
                            x1 = previous_ref_coords(1);    y1 = previous_ref_coords(0);
                            x2 = previous_edge1_coords(1);  y2 = previous_edge1_coords(0);
                            x3 = previous_edge2_coords(1);  y3 = previous_edge2_coords(0);

                            lambda1 = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)) ;
                            lambda2 = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3));

                            previous_depth = lambda1 * depth1 + lambda2 * depth2 + (1.0 - lambda1 - lambda2)* depth3  ;

                            if(new_depth > previous_depth)
                                new_idx = previous_triangle-1; //Don't change index if depth was not closer
                        }
                        previous_triangle = new_idx+1;// 0 is only used for no association so image contains triangle index + 1
                    }
                }
            }
    }
}


template<class InTriangle, class InPoint>
void SpaceTimeSampler::setPixelsWhereTrianglesProjectCloser2(const std::vector<InTriangle> &triangles, const std::vector<InPoint> &points, cv::Mat &out_image, cv::Mat &depth_map, const Camera &cam, const int window_rad, const double depth_threshold, const float cleaning_factor)const{

    //int window_size = 2*8*cleaning_factor+1;
    //double depth_threshold = 0.03;   //bad solution, for now
    int rad = window_rad*cleaning_factor;
    long unsigned int long_zero=0;

    Vector2f sup_left,inf_right;
    Vector2f ref_coords,edge1_coords,edge2_coords;
    const Vector3f &camera_center = cam.getPosition();
    for(int32_t triangle_idx = 0; triangle_idx < triangles.size(); ++triangle_idx)
    {
        const InTriangle &tri = triangles[triangle_idx];
        
        //Project every point
        cam.getTextureCoords(points[tri.ref],ref_coords);
        cam.getTextureCoords(points[tri.edge1],edge1_coords);
        cam.getTextureCoords(points[tri.edge2],edge2_coords);

        ref_coords *= cleaning_factor;
        edge1_coords *= cleaning_factor;
        edge2_coords *= cleaning_factor;

        sup_left = ref_coords;
        inf_right = ref_coords;

        //Sup left
        if(edge1_coords(0) < sup_left(0))
            sup_left(0) = edge1_coords(0);
        if(edge2_coords(0) < sup_left(0))
            sup_left(0) = edge2_coords(0);

        if(edge1_coords(1) < sup_left(1))
            sup_left(1) = edge1_coords(1);
        if(edge2_coords(1) < sup_left(1))
            sup_left(1) = edge2_coords(1);

        //Inf Right
        if(edge1_coords(0) > inf_right(0))
            inf_right(0) = edge1_coords(0);
        if(edge2_coords(0) > inf_right(0))
            inf_right(0) = edge2_coords(0);

        if(edge1_coords(1) > inf_right(1))
            inf_right(1) = edge1_coords(1);
        if(edge2_coords(1) > inf_right(1))
            inf_right(1) = edge2_coords(1);

        //Fill area if dist is inferior to already registered
        for(unsigned long int py = std::max(0.0,floor(sup_left(0))) ; py < std::min(ceil(inf_right(0)),double(IMG_HEIGHT*cleaning_factor)); ++py)
            for(unsigned long int px = std::max(0.0,floor(sup_left(1))); px < std::min(ceil(inf_right(1)),double(IMG_WIDTH*cleaning_factor)); ++px)
            {
                double x,y,x1,x2,x3,y1,y2,y3,lambda1,lambda2,depth1,depth2,depth3,previous_depth,new_depth;
                x = (double)px;
                y = (double)py;
                //Recover depth of the triangle vertices
                depth1 = Vector3f(points[tri.ref] - camera_center).norm();
                depth2 = Vector3f(points[tri.edge1] - camera_center).norm();
                depth3 = Vector3f(points[tri.edge2] - camera_center).norm();

                //Compute barycentric coordinates

                x1 = ref_coords(1);     y1 = ref_coords(0);
                x2 = edge1_coords(1);   y2 = edge1_coords(0);
                x3 = edge2_coords(1);   y3 = edge2_coords(0);

                lambda1 = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)) ;
                lambda2 = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3));


                if(lambda1 >= 0.0 && lambda2 >= 0.0 && 1.0 - lambda1 - lambda2 >= 0.0)
                {
                    //Inside triangle
                    new_depth = lambda1 * depth1 + lambda2 * depth2 + (1.0 - lambda1 - lambda2)* depth3 ;
                    //get pixel value
                    if(px > 0 && px < (IMG_WIDTH*cleaning_factor) && py > 0 && py < IMG_HEIGHT*cleaning_factor)
                    {
                        int32_t &previous_triangle = out_image.at<int32_t>(py,px);
                        double &previous_depth = depth_map.at<double>(py,px);
                        //double &previous_depth = depth_map.at<double>(0,0);
                        //double previous_depth=0;
                        int32_t new_idx = triangle_idx;
                        //*
                        if(previous_triangle > 0)
                        {
                            if(new_depth > previous_depth)
                            {
                                new_idx = previous_triangle-1; //Don't change index if depth was not closer
                                new_depth = previous_depth;
                            }
                        }
                        
                        previous_triangle = new_idx+1;// 0 is only used for no association so image contains triangle index + 1
                        previous_depth = new_depth;

                    }
                }
            }
    }
    //every triangle projected.
    //Now, grow safety margins around foreground triangles


    //check pixel, and set neighbourhood to zero if condition is met
    for(unsigned long int x = 0; x<IMG_WIDTH*cleaning_factor; ++x)
    {
        for(unsigned long int y = 0; y < IMG_HEIGHT*cleaning_factor; ++y)   //for each pixel
        {
            //int32_t &myTri = out_image.at<int32_t>(y,x);
            

            
            if(depth_map.at<double>(y,x)!=0.0f)
            {
                double grad_max=0;
                double ref_depth = depth_map.at<double>(y,x);
                //lateral thinking
                unsigned long minx = x;
                unsigned long miny = y;
                if((x<IMG_WIDTH*cleaning_factor-1)&&(depth_map.at<double>(y,x+1)!=0.0f))
                {
                    grad_max=std::max(std::abs(depth_map.at<double>(y,x)-depth_map.at<double>(y,x+1)),grad_max);
                    if(depth_map.at<double>(y,x+1)<ref_depth)
                    {
                        ref_depth = depth_map.at<double>(y,x+1);
                        ++minx;
                    }
                }
                    //ref_depth = std::min(ref_depth, depth_map.at<double>(y,x+1));
                    //grad_max=std::max(depth_map.at<double>(y,x)-depth_map.at<double>(y,x+1),grad_max);
                    
                if((y<IMG_HEIGHT*cleaning_factor-1)&&(depth_map.at<double>(y+1,x)!=0.0f))
                {
                    grad_max=std::max(std::abs(depth_map.at<double>(y,x)-depth_map.at<double>(y+1,x)),grad_max);
                    if(depth_map.at<double>(y+1,x)<ref_depth)
                    {
                        ref_depth = depth_map.at<double>(y+1,x);
                        ++miny;
                    }
                    //ref_depth = std::min(ref_depth, depth_map.at<double>(y+1,x));
                }

                /*if((x>0)&&(depth_map.at<double>(y,x-1)!=0))
                    grad_max=std::max(std::abs(depth_map.at<double>(y,x)-depth_map.at<double>(y,x-1)),grad_max);
                    ref_depth = std::min(ref_depth, depth_map.at<double>(y,x-1));

                if((y>0)&&(depth_map.at<double>(y-1,x)!=0))
                    grad_max=std::max(std::abs(depth_map.at<double>(y,x)-depth_map.at<double>(y-1,x)),grad_max);
                    ref_depth = std::min(ref_depth, depth_map.at<double>(y-1,x));
                */

                if(ref_depth==0.0f)
                {
                    log(ALWAYS)<<"WARNING!!!!!!!!!!!"<<endLog();
                    log(ALWAYS)<<"(y,x): "<<depth_map.at<double>(y,x)<<endLog();
                    if(y<IMG_HEIGHT*cleaning_factor-1)
                        log(ALWAYS)<<"(y+1,x): "<<depth_map.at<double>(y+1,x)<<endLog();
                    if(x<IMG_WIDTH*cleaning_factor-1)
                        log(ALWAYS)<<"(y,x+1): "<<depth_map.at<double>(y,x+1)<<endLog();

                }

                double maxVisibleDepth = (depth_threshold+ref_depth);
                //double maxVisibleDepth = (depth_threshold+depth_map.at<double>(miny,minx));

                if(grad_max>depth_threshold)
                {

                    unsigned long imin = std::max(long_zero,x-rad);
                    unsigned long imax = std::min(x+rad,(unsigned long)(IMG_WIDTH*cleaning_factor)-1);
                    unsigned long jmin = std::max(long_zero,y-rad);
                    unsigned long jmax = std::min(y+rad,(unsigned long)(IMG_HEIGHT*cleaning_factor)-1);
                    for(unsigned long i = imin; i <=imax; ++i)
                    {
                        for(unsigned long j = jmin; j <= jmax; ++j)
                        {
                            
                            int32_t &myTri = out_image.at<int32_t>(j,i);
                            if (myTri>0)
                            {
                                double myDepth = depth_map.at<double>(j,i);
                                if(myDepth>maxVisibleDepth)//if(myDepth>depth_threshold)
                                {
                                    myTri=0;
                                }
                                // else
                                // {
                                //     myTri=0;
                                // }
                            }
                        }
                    }

                }
                
            }
            //end pixel
        }
    }

    // //New check: erode silhouette!
    // for(unsigned long int x = 0; x<IMG_WIDTH*cleaning_factor; ++x)
    // {
    //     for(unsigned long int y = 0; y < IMG_HEIGHT*cleaning_factor; ++y)   //for each pixel
    //     {
    //         bool bDel = false;
    //         if(depth_map.at<double>(y,x)==0.0f)     //empty
    //         {
    //             if((x<IMG_WIDTH*cleaning_factor-1)&&(depth_map.at<double>(y,x+1)!=0.0f))
    //                 bDel=true;
    //             if((y<IMG_HEIGHT*cleaning_factor-1)&&(depth_map.at<double>(y+1,x)!=0.0f))
    //                 bDel=true;
    //             if((x>0)&&(depth_map.at<double>(y,x-1)!=0.0f))
    //                 bDel=true;
    //             if((y>0)&&(depth_map.at<double>(y-1,x)!=0.0f))
    //                 bDel=true;
    //             if(bDel)
    //             {
    //                 unsigned long imin = std::max(long_zero,x-rad);
    //                 unsigned long imax = std::min(x+rad,(unsigned long)(IMG_WIDTH*cleaning_factor)-1);
    //                 unsigned long jmin = std::max(long_zero,y-rad);
    //                 unsigned long jmax = std::min(y+rad,(unsigned long)(IMG_HEIGHT*cleaning_factor)-1);
    //                 for(unsigned long i = imin; i <=imax; ++i)
    //                 {
    //                     for(unsigned long j = jmin; j <= jmax; ++j)
    //                     {
    //                         out_image.at<int32_t>(j,i)=0;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

}


template<class InTriangle, class InPoint>
void SpaceTimeSampler::setPixelsWhereTrianglesProjectCloserWConfidence(const std::vector<InTriangle> &triangles, const std::vector<InPoint> &points, cv::Mat &out_image, cv::Mat &depth_map, cv::Mat &confidence_map, const Camera &cam, const int window_rad, const double depth_threshold, const float cleaning_factor)const{

    int rad = window_rad*cleaning_factor;
    long unsigned int long_zero=0;

    Vector2f sup_left,inf_right;
    Vector2f ref_coords,edge1_coords,edge2_coords;
    const Vector3f &camera_center = cam.getPosition();
    for(int32_t triangle_idx = 0; triangle_idx < triangles.size(); ++triangle_idx)
    {
        const InTriangle &tri = triangles[triangle_idx];
        
        //Project every point
        cam.getTextureCoords(points[tri.ref],ref_coords);
        cam.getTextureCoords(points[tri.edge1],edge1_coords);
        cam.getTextureCoords(points[tri.edge2],edge2_coords);

        ref_coords *= cleaning_factor;
        edge1_coords *= cleaning_factor;
        edge2_coords *= cleaning_factor;

        sup_left = ref_coords;
        inf_right = ref_coords;

        //Sup left
        if(edge1_coords(0) < sup_left(0))
            sup_left(0) = edge1_coords(0);
        if(edge2_coords(0) < sup_left(0))
            sup_left(0) = edge2_coords(0);

        if(edge1_coords(1) < sup_left(1))
            sup_left(1) = edge1_coords(1);
        if(edge2_coords(1) < sup_left(1))
            sup_left(1) = edge2_coords(1);

        //Inf Right
        if(edge1_coords(0) > inf_right(0))
            inf_right(0) = edge1_coords(0);
        if(edge2_coords(0) > inf_right(0))
            inf_right(0) = edge2_coords(0);

        if(edge1_coords(1) > inf_right(1))
            inf_right(1) = edge1_coords(1);
        if(edge2_coords(1) > inf_right(1))
            inf_right(1) = edge2_coords(1);

        //Fill area if dist is inferior to already registered
        for(unsigned long int py = std::max(0.0,floor(sup_left(0))) ; py < std::min(ceil(inf_right(0)),double(IMG_HEIGHT*cleaning_factor)); ++py)
            for(unsigned long int px = std::max(0.0,floor(sup_left(1))); px < std::min(ceil(inf_right(1)),double(IMG_WIDTH*cleaning_factor)); ++px)
            {
                double x,y,x1,x2,x3,y1,y2,y3,lambda1,lambda2,depth1,depth2,depth3,previous_depth,new_depth;
                x = (double)px;
                y = (double)py;
                //Recover depth of the triangle vertices
                depth1 = Vector3f(points[tri.ref] - camera_center).norm();
                depth2 = Vector3f(points[tri.edge1] - camera_center).norm();
                depth3 = Vector3f(points[tri.edge2] - camera_center).norm();

                //Compute barycentric coordinates

                x1 = ref_coords(1);     y1 = ref_coords(0);
                x2 = edge1_coords(1);   y2 = edge1_coords(0);
                x3 = edge2_coords(1);   y3 = edge2_coords(0);

                lambda1 = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)) ;
                lambda2 = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3));


                if(lambda1 >= 0.0 && lambda2 >= 0.0 && 1.0 - lambda1 - lambda2 >= 0.0)
                {
                    //Inside triangle
                    new_depth = lambda1 * depth1 + lambda2 * depth2 + (1.0 - lambda1 - lambda2)* depth3 ;
                    //get pixel value
                    if(px > 0 && px < (IMG_WIDTH*cleaning_factor) && py > 0 && py < IMG_HEIGHT*cleaning_factor)
                    {
                        int32_t &previous_triangle = out_image.at<int32_t>(py,px);
                        double &previous_depth = depth_map.at<double>(py,px);
                        int32_t new_idx = triangle_idx;
                        //*
                        if(previous_triangle > 0)
                        {
                            if(new_depth > previous_depth)
                            {
                                new_idx = previous_triangle-1; //Don't change index if depth was not closer
                                new_depth = previous_depth;
                            }
                        }
                        
                        previous_triangle = new_idx+1;// 0 is only used for no association so image contains triangle index + 1
                        previous_depth = new_depth;

                    }
                }
            }
    }
    //every triangle projected.
    //Now, grow safety margins around foreground triangles


    //check pixel, and set neighbourhood to zero if condition is met
    for(unsigned long int x = 0; x<IMG_WIDTH*cleaning_factor; ++x)
    {
        for(unsigned long int y = 0; y < IMG_HEIGHT*cleaning_factor; ++y)   //for each pixel
        {
            //int32_t &myTri = out_image.at<int32_t>(y,x);
            

            
            if(depth_map.at<double>(y,x)!=0.0f)
            {
                double grad_max=0;
                double ref_depth = depth_map.at<double>(y,x);
                //lateral thinking
                unsigned long minx = x;
                unsigned long miny = y;
                if((x<IMG_WIDTH*cleaning_factor-1)&&(depth_map.at<double>(y,x+1)!=0.0f))
                {
                    grad_max=std::max(std::abs(depth_map.at<double>(y,x)-depth_map.at<double>(y,x+1)),grad_max);
                    if(depth_map.at<double>(y,x+1)<ref_depth)
                    {
                        ref_depth = depth_map.at<double>(y,x+1);
                        ++minx;
                    }
                }
                    //ref_depth = std::min(ref_depth, depth_map.at<double>(y,x+1));
                    //grad_max=std::max(depth_map.at<double>(y,x)-depth_map.at<double>(y,x+1),grad_max);
                    
                if((y<IMG_HEIGHT*cleaning_factor-1)&&(depth_map.at<double>(y+1,x)!=0.0f))
                {
                    grad_max=std::max(std::abs(depth_map.at<double>(y,x)-depth_map.at<double>(y+1,x)),grad_max);
                    if(depth_map.at<double>(y+1,x)<ref_depth)
                    {
                        ref_depth = depth_map.at<double>(y+1,x);
                        ++miny;
                    }
                    //ref_depth = std::min(ref_depth, depth_map.at<double>(y+1,x));
                }

                /*if((x>0)&&(depth_map.at<double>(y,x-1)!=0))
                    grad_max=std::max(std::abs(depth_map.at<double>(y,x)-depth_map.at<double>(y,x-1)),grad_max);
                    ref_depth = std::min(ref_depth, depth_map.at<double>(y,x-1));

                if((y>0)&&(depth_map.at<double>(y-1,x)!=0))
                    grad_max=std::max(std::abs(depth_map.at<double>(y,x)-depth_map.at<double>(y-1,x)),grad_max);
                    ref_depth = std::min(ref_depth, depth_map.at<double>(y-1,x));
                */

                if(ref_depth==0.0f)
                {
                    log(ALWAYS)<<"WARNING!!!!!!!!!!!"<<endLog();
                    log(ALWAYS)<<"(y,x): "<<depth_map.at<double>(y,x)<<endLog();
                    if(y<IMG_HEIGHT*cleaning_factor-1)
                        log(ALWAYS)<<"(y+1,x): "<<depth_map.at<double>(y+1,x)<<endLog();
                    if(x<IMG_WIDTH*cleaning_factor-1)
                        log(ALWAYS)<<"(y,x+1): "<<depth_map.at<double>(y,x+1)<<endLog();

                }

                double maxVisibleDepth = (depth_threshold+ref_depth);
                //double maxVisibleDepth = (depth_threshold+depth_map.at<double>(miny,minx));

                if(grad_max>depth_threshold)
                {

                    unsigned long imin = std::max(long_zero,x-rad);
                    unsigned long imax = std::min(x+rad,(unsigned long)(IMG_WIDTH*cleaning_factor)-1);
                    unsigned long jmin = std::max(long_zero,y-rad);
                    unsigned long jmax = std::min(y+rad,(unsigned long)(IMG_HEIGHT*cleaning_factor)-1);
                    for(unsigned long i = imin; i <=imax; ++i)
                    {
                        for(unsigned long j = jmin; j <= jmax; ++j)
                        {
                            
                            int32_t &myTri = out_image.at<int32_t>(j,i);
                            if (myTri>0)
                            {
                                double myDepth = depth_map.at<double>(j,i);
                                if(myDepth>maxVisibleDepth)//if(myDepth>depth_threshold)
                                {
                                    //linear kernel
                                    float dist = sqrt(pow((i-x),2)+pow((j-y),2)) - rad/2;
                                    confidence_map.at<double>(j,i) = std::max(0.0f,std::min(1.0f,2*dist/rad));
                                    // confidence_map.at<double>(j,i) = 0.0f;
                                }
                                else
                                {
                                    float dist = sqrt(pow((i-x),2)+pow((j-y),2)) - rad/2;
                                    confidence_map.at<double>(j,i) = std::max(0.0f,std::min(1.0f,2*dist/rad));
                                    // confidence_map.at<double>(j,i) = 0.0f;
                                }
                            }
                        }
                    }

                }
                
            }
            //end pixel
        }
    }

    //New check: erode silhouette!
    for(unsigned long int x = 0; x<IMG_WIDTH*cleaning_factor; ++x)
    {
        for(unsigned long int y = 0; y < IMG_HEIGHT*cleaning_factor; ++y)   //for each pixel
        {
            bool bDel = false;
            if(depth_map.at<double>(y,x)==0.0f)     //empty
            {
                if((x<IMG_WIDTH*cleaning_factor-1)&&(depth_map.at<double>(y,x+1)!=0.0f))
                    bDel=true;
                if((y<IMG_HEIGHT*cleaning_factor-1)&&(depth_map.at<double>(y+1,x)!=0.0f))
                    bDel=true;
                if((x>0)&&(depth_map.at<double>(y,x-1)!=0.0f))
                    bDel=true;
                if((y>0)&&(depth_map.at<double>(y-1,x)!=0.0f))
                    bDel=true;
                if(bDel)
                {
                    unsigned long imin = std::max(long_zero,x-rad);
                    unsigned long imax = std::min(x+rad,(unsigned long)(IMG_WIDTH*cleaning_factor)-1);
                    unsigned long jmin = std::max(long_zero,y-rad);
                    unsigned long jmax = std::min(y+rad,(unsigned long)(IMG_HEIGHT*cleaning_factor)-1);
                    for(unsigned long i = imin; i <=imax; ++i)
                    {
                        for(unsigned long j = jmin; j <= jmax; ++j)
                        {
                            //linear kernel
                            float dist = sqrt(pow((i-x),2)+pow((j-y),2)) - rad/2;
                            confidence_map.at<double>(j,i) = std::max(0.0f,std::min(1.0f,2*dist/rad));
                            // confidence_map.at<double>(j,i) = 0.0f;
                        }
                    }
                }
            }
        }
    }

}

template<class InTriangle, class InPoint, class InColor, class MySpecialMesh>
void SpaceTimeSampler::reIndexColors(   MySpecialMesh *in_mesh,
                                        int default_face_res,
                                        float quantMatCoefs[],
                                        int downsamplingThreshold
                                    )const
{
    

    std::vector<InTriangle> in_faces;
    std::vector<InPoint> in_points;
    std::vector<InColor> in_colors;
    std::vector<unsigned short> in_face_res;
    std::vector<Vector3li> in_edge_color_ind;
    std::vector<unsigned long> in_face_color_ind;
    std::vector<size_t> in_edge_indices;
    in_mesh->getFacesVector(in_faces);
    in_mesh->getPointsVector(in_points);
    in_mesh->getColorsVector(in_colors);
    in_mesh->getFacesResVector(in_face_res);
    in_mesh->getEdgesIndVector(in_edge_color_ind);
    in_mesh->getFacesIndVector(in_face_color_ind);
    in_mesh->getEdgesRealColorInd(in_edge_indices);
    std::string outputPath = output_folder_;

    int edgeResPixelsNumber=0;

    //computing temp edge vector
    std::vector<Vector3li> temp_edge_color_ind(in_edge_color_ind.size());
    if(default_face_res<40)
    {
        
        for(int tri=0;tri<in_faces.size();++tri)
        {
            for(int e=0;e<3;++e)
            {
                if (in_edge_color_ind[tri](e)>=0)
                    temp_edge_color_ind[tri](e) = in_edge_indices[in_edge_color_ind[tri](e)];
                else
                    temp_edge_color_ind[tri](e) = - in_edge_indices[- in_edge_color_ind[tri](e)];
            }
        }
    }
    else
    {
        temp_edge_color_ind = in_edge_color_ind;
    }
    downsampleMeshColor(in_faces, in_colors, in_face_res, temp_edge_color_ind, in_face_color_ind, downsamplingThreshold);
    

    boost::posix_time::ptime time_begin;     //to measure computation time of different algorithmic blocks.
    boost::posix_time::time_duration time_diff;
    // ----------------
    // vertices
    // ----------------

    //1st try: keep all vertices together to save changes due to 'edge resolution' pixels
    // then, process edges and triangles
    
    std::vector<int> points_intensity(in_points.size());
    std::vector<InColor> vert_colors(in_colors.begin(), in_colors.begin()+in_points.size());
    long in_colors_size = in_colors.size();
    std::vector<InColor> temp_colors(in_colors);
    //cv::Mat cvColorMat = cv::Mat(in_colors.size(),1,CV_8UC3,cv::Vec3f(0.0f,0.0f,0.0f));     //we put the color vector into an openCV Mat, that can be saved and loaded,
                                                                                            //so that the colormap can be shared between frames
    in_colors.clear();
    in_colors.reserve(in_colors_size);

    
    //order vertices
    log(ALWAYS)<<"ordering vertices ("<<in_points.size()<<" points)"<<endLog();
    for(int v=0;v<in_points.size();++v)
    {
        points_intensity[v]=vert_colors[v].sum();
    }
    std::vector<size_t> new_ind = sort_indexes(points_intensity);
    std::vector<size_t> inv_new_ind(new_ind.size());
    std::vector<InPoint> points = in_points;
    in_points.clear();
    
    //We choose a starting vertex, then, for each next vertex, we choose the closest one to the current vertex in terms of color distance
    
    int curInd = new_ind[0];                            //used to store the current/next index to be written
    InColor curColor = vert_colors[curInd];            //used to store color of current/next vertex to be written
    std::vector<int> is_written(points.size(),0);       //boolean vector to keep track of vertices already processed
    is_written[curInd]=1;
    int total_written=1;                                //current number of processed vertices
    in_points.push_back(points[curInd]);
    in_colors.push_back(curColor);
    inv_new_ind[curInd]=0;
    //loop
    while(total_written<points.size())
    {
        float min_dist = 3*255;
        for(int next_v=0;next_v<points.size();++next_v)
        {
            if (is_written[next_v]==0)
            {
                //float &colorDist;
                //getColorDistance(curColor, const vert_colors[next_v],colorDist);
                const InColor sampColor = vert_colors[next_v];
                float colorDist = getColorDistance(curColor, sampColor);
                if ((colorDist<min_dist)||(colorDist==min_dist && points_intensity[next_v]<points_intensity[curInd]))
                {
                    min_dist=colorDist;
                    curInd=next_v;
                    if(colorDist==0){
                        break;
                    }
                }
            }
        }
        is_written[curInd]=1;
        curColor = vert_colors[curInd];
        in_points.push_back(points[curInd]);
        in_colors.push_back(curColor);
        inv_new_ind[curInd]=total_written;
        ++total_written;
    }
    
    log(ALWAYS)<<"in_colors size: "<<in_colors.size()<<endLog();
    //reindex triangles with new vertices order
    log(ALWAYS)<<"Reindexing triangles ("<<in_faces.size()<<" triangles)"<<endLog();
    std::vector<InTriangle> triangles = in_faces;
    in_faces.clear();
    in_faces.reserve(triangles.size());
    for(int tri=0; tri<triangles.size();tri++)
    {
        InTriangle myTri;
        myTri.ref = inv_new_ind[triangles[tri].ref];
        myTri.edge1 = inv_new_ind[triangles[tri].edge1];
        myTri.edge2 = inv_new_ind[triangles[tri].edge2];
        in_faces.push_back(myTri);
    }
    

    // ----------------
    // faces
    // ----------------
    std::vector<float> triangles_avg_intensity(in_faces.size(),0);
    std::vector<float> edges_avg_intensity(in_edge_indices.size(),0);
    int maxNumSamples = (default_face_res-2)*(default_face_res-1)/2;
    std::vector<unsigned long> temp_face_color_ind(in_face_color_ind);


    //rewrite vertices colors in temp_colors
    //not ideal, the whole function has been repurposed. Will do for now.
    for(int i=0;i<in_colors.size();++i)
    {
        temp_colors[i]=in_colors[i];
    }

    //order triangles
    log(ALWAYS)<<"order triangles"<<endLog();
    for(int tri=0;tri<in_faces.size();++tri)
    {
        int faceRes = in_face_res[tri];
        int num_samples = (faceRes-1)*(faceRes-2)/2;
        for(int i=0;i<num_samples;++i)
        {
            triangles_avg_intensity[tri] += temp_colors[in_face_color_ind[tri]+i].sum();
        }
        triangles_avg_intensity[tri]/=num_samples;
    }

    std::vector<size_t> new_tri_ind = sort_indexes(triangles_avg_intensity);
    
    long savedPixels = 0;   //To keep track of image size reduction
    //For now, look for exact match
    for(int myRes = default_face_res;myRes>2;--myRes)    //write max res color samples first (longest vectors), then move to second max res and so on
    {
        time_begin  = boost::posix_time::microsec_clock::local_time();
        //get number of samples for triangles with that level of resolution
        int samplesNumber = (myRes-2)*(myRes-1)/2;

        //make list of triangles with current resolution
        std::vector<size_t> curResFaces(in_face_res.size(),0);
        int curInd=0;
        for(int tri=0;tri<in_faces.size();++tri)
        {
            if(in_face_res[new_tri_ind[tri]]==myRes)    //We add triangles to the list, following the order given by intensity mean.
            {                                           //That way, each level of resolution will be processed following this order
                curResFaces[curInd]=new_tri_ind[tri];
                ++curInd;
            }
        }
        curResFaces.erase(curResFaces.begin()+curInd,curResFaces.end());
        //Now, we have the correct vector of index
        //Loop through it
        for(int tri=0;tri<curResFaces.size();++tri)
        {
            if(curResFaces[tri]>=in_face_color_ind.size())
            {
                log(ERROR)<<"ERROR: bad index: tri "<<tri<<", curResFaces[tri] = "<<curResFaces[tri]<<" on "<<in_face_color_ind.size()<<endLog();
            }
            in_face_color_ind[curResFaces[tri]] = in_colors.size();
            in_colors.insert(in_colors.end(), temp_colors.begin()+temp_face_color_ind[curResFaces[tri]], temp_colors.begin()+temp_face_color_ind[curResFaces[tri]]+samplesNumber);
        }
        time_diff = boost::posix_time::microsec_clock::local_time() - time_begin;
    }

    log(ALWAYS)<<"Total saved pixels: "<<savedPixels<<" out of "<<temp_colors.size()<<endLog();
    log(ALWAYS)<<"in_colors size: "<<in_colors.size()<<endLog();
    //set index to 0 for faces with no sample (res 1 or 2)
    for(int tri=0;tri<in_faces.size();++tri)
    {
        if(in_face_res[tri]<=2)
        {
            in_face_color_ind[tri]=0;
        }
    }

    // -------------------
    // edges
    // -------------------
    //order edges
    edgeResPixelsNumber = in_edge_indices.size();
    
    log(ALWAYS)<<"reorder edges"<<endLog();
    for(int e=0;e<in_edge_indices.size();++e)
    {
        int edge_res = temp_colors[in_edge_indices[e]](0);
        for(int i=1; i<edge_res; ++i){
            edges_avg_intensity[e] += temp_colors[in_edge_indices[e]+i].sum();
        }
        edges_avg_intensity[e]/= (edge_res-1);
    }
    std::vector<size_t> new_edge_ind = sort_indexes(edges_avg_intensity);

    std::vector<size_t> temp_edge_indices = in_edge_indices;

    //do some process similar to faces
    savedPixels=0;
    for(int myRes = default_face_res;myRes>1;--myRes)
    {
        time_begin  = boost::posix_time::microsec_clock::local_time();
        //make list of edges with current resolution
        std::vector<size_t> curResEdges(in_edge_indices.size(),0);
        int curInd=0;
        //long startingInd = in_colors.size();
        for(int e=0;e<in_edge_indices.size();++e)
        {
            if(temp_colors[temp_edge_indices[new_edge_ind[e]]](0)==myRes)    //We add edges to the list, following the order given by intensity mean.
            {                                           //That way, each level of resolution will be processed following this order
                curResEdges[curInd]=new_edge_ind[e];
                ++curInd;
            }
        }
        curResEdges.erase(curResEdges.begin()+curInd,curResEdges.end());

        //loop through selected edges
        for(int e=0;e<curResEdges.size();++e)
        {
            in_edge_indices[curResEdges[e]] = in_colors.size();
            in_colors.insert(in_colors.end(), temp_colors.begin()+temp_edge_indices[curResEdges[e]], temp_colors.begin()+temp_edge_indices[curResEdges[e]]+myRes);
        }
        time_diff = boost::posix_time::microsec_clock::local_time() - time_begin;
    }
    int edge_res_pixels_saved=0;
    log(ALWAYS)<<"Total saved pixels: "<<savedPixels<<" out of "<<temp_colors.size()<<" (edges)"<<endLog();
    log(ALWAYS)<<"in_colors size: "<<in_colors.size()<<endLog();
    //set index to zero for edges with no sample
    for(int e=0;e<in_edge_indices.size();++e)
    {
        if(temp_colors[temp_edge_indices[e]](0)<=1)
        {
            in_edge_indices[e]=0;
            ++edge_res_pixels_saved;
            --edgeResPixelsNumber;
        }
    }
    log(ALWAYS)<<"useless (edge) pixels saved: "<<edge_res_pixels_saved<<endLog();

    log(ALWAYS)<<"reindex edges"<<endLog();
    for(int tri=0;tri<triangles.size();++tri)
    {
        for(int e=0;e<3;++e)
        {
            if (in_edge_color_ind[tri](e)>=0)
                temp_edge_color_ind[tri](e) = in_edge_indices[in_edge_color_ind[tri](e)];
            else
                temp_edge_color_ind[tri](e) = - in_edge_indices[- in_edge_color_ind[tri](e)];
        }
    }

    in_edge_color_ind = temp_edge_color_ind;


    in_colors[0](0)=1;  //for edges with res 1

    //Update input mesh
    in_mesh->setFacesVector(in_faces);
    in_mesh->setPointsVector(in_points);
    in_mesh->setColorsVector(in_colors);
    in_mesh->setFacesResVector(in_face_res);
    in_mesh->setEdgesIndVector(in_edge_color_ind);
    in_mesh->setFacesIndVector(in_face_color_ind);

    in_mesh->setEdgesRealColorInd(in_edge_indices);

}


template<class InTriangle, class InPoint, class InColor, class MySpecialMesh>
void SpaceTimeSampler::compressColor(   MySpecialMesh *in_mesh,
                                        std::vector<size_t> &in_edge_indices,
                                        int default_face_res,
                                        int quantFactor,
                                        float quantMatCoefs[],
                                        int downsamplingThreshold
                                    )const
{
    
    log(ALWAYS)<<"Starting comp..."<<endLog();
    // std::vector<InTriangle> &in_faces = in_mesh->getRealFacesVector();
    // in_faces = in_mesh->getFacesVector();
    std::vector<InTriangle> in_faces;
    std::vector<InPoint> in_points;
    std::vector<InColor> in_colors;
    std::vector<BitArray> out_bit_array;
    std::vector<unsigned short> in_face_res;
    std::vector<Vector3li> in_edge_color_ind;
    std::vector<unsigned long> in_face_color_ind;
    std::string compFileName = in_mesh->getAppearanceFileName();

    in_mesh->getFacesVector(in_faces);
    in_mesh->getPointsVector(in_points);
    in_mesh->getColorsVector(in_colors);
    in_mesh->getFacesResVector(in_face_res);
    in_mesh->getEdgesIndVector(in_edge_color_ind);
    in_mesh->getFacesIndVector(in_face_color_ind);

    std::string outputPath = output_folder_;

    std::vector<Vector3li> temp_edge_color_ind = in_edge_color_ind;
    
    boost::posix_time::ptime time_begin;     //to measure computation time of different algorithmic blocks.
    boost::posix_time::time_duration time_diff;
    
    std::vector<InTriangle> triangles(in_faces);
    std::vector<unsigned long> temp_face_color_ind(in_face_color_ind);

    log(ALWAYS)<<"Preparation Ok"<<endLog();

    // --- PCA --- 

    std::vector<InColor> temp_colors(in_colors);
    MeshEncoder meshEncoder = MeshEncoder();

    int samplesNumber, chromaSamplesNumber;
    int minIndex, maxIndex, triLength, chromaTriLength;
    bool processFullTri = true;

    std::map<int,cv::Mat> resDCTList;
    std::map<int,std::vector<float>> resQTList;
    //std::map<int,cv::PCA> rescvPCAList;
    std::map<int,int> quantMultipliers;
    std::map<int,cv::Mat> resEigenVecList;
    for(int targetRes = default_face_res;targetRes>=2;targetRes=targetRes/2)
    {
        //make matrix = array of color vectors
        //cv::Mat colorPatterns = cv::Mat(triangles.size(),maxNumSamples*3,CV_32FC1, -1.0f);      //-1 means there is no data stored there. We can fill it with whatever we want.
        
        if(processFullTri)
        {
            //compressing edge samples with it
            samplesNumber = (targetRes+2)*(targetRes+1)/2-3;
            chromaSamplesNumber = (targetRes/2+2)*(targetRes/2+1)/2-3;
            log(ALWAYS)<<"samplesNumber = "<<samplesNumber<<", chromaSamplesNumber = "<<chromaSamplesNumber<<endLog();
            minIndex=0;
            maxIndex=targetRes;
            triLength = targetRes+1;
            chromaTriLength = targetRes/2+1;
        }
        else
        {
            samplesNumber = (targetRes-2)*(targetRes-1)/2;
            chromaSamplesNumber = (targetRes/2-2)*(targetRes/2-1)/2;
            minIndex=1;
            maxIndex=targetRes-1;
            triLength = targetRes-2;
            chromaTriLength = targetRes/2-2;
        }



        int ctri=0;         //index for triangles with current resolution
        int patternsNum=0;  //Number of triangles with current resolution
        //Compute number of triangle with this res, + compute their mean color value:
        log(ALWAYS)<<"triangles size "<<triangles.size()<<endLog();
        for(int tri=0;tri<triangles.size();++tri)
        {
            if(in_face_res[tri]==targetRes)
            {
                ++patternsNum;
            }
        }
        log(ALWAYS)<<"patternNum = "<<patternsNum<<endLog();
        if(patternsNum==0)
        {
            continue;
        }

        //cv::Mat colorPatterns = cv::Mat(patternsNum,3*samplesNumber,CV_32FC1, -1.0f);     //full RGB version
        cv::Mat colorPatterns = cv::Mat(patternsNum,samplesNumber+2*chromaSamplesNumber,CV_32FC1, -1.0f);   //with decimated chroma components

        //Fill matrix with color patterns as row vectors
        for(int tri=0; tri<triangles.size();tri++)
        {
            int myRes = in_face_res[tri];
            if(myRes==targetRes)
            {
                //face only version (faster in that case)
                    //float R,G,B;
                    //for(int j=0;j<samplesNumber;++j)
                    //{
                    //    long myInd = temp_face_color_ind[tri]+j;
                    //    R+=float(in_colors[myInd](0))/255;
                    //    G+=float(in_colors[myInd](1))/255;
                    //    B+=float(in_colors[myInd](2))/255;
                    //    colorPatterns.at<float>(ctri,3*j) = float(in_colors[myInd](0))/255;    //red
                    //    colorPatterns.at<float>(ctri,3*j+1) = float(in_colors[myInd](1))/255;  //green
                    //    colorPatterns.at<float>(ctri,3*j+2) = float(in_colors[myInd](2))/255;  //blue
                    //}

                //with vertices and edges version
                int sampId=0;
                int chromaSampId=0;
                InColor myColor;
                for(int l1=minIndex;l1<=maxIndex;++l1)    //loop through barycentric coordinates
                {
                    for(int l2=minIndex;l2<=maxIndex-l1;++l2)
                    {

                        if((l1+l2==0)||(l1==maxIndex)||(l2==maxIndex))
                        {
                            continue;
                        }
                        myColor = getSampleColor(in_faces[tri], tri, myRes, l1, l2, temp_edge_color_ind, temp_face_color_ind, in_colors);

                        Vector3f colorOffset = Vector3f(float(myColor(0)),float(myColor(1)),float(myColor(2)));
                        //write color, increment index
                        //YCbCr
                        //colorPatterns.at<float>(ctri,sampId) = (float(myColor(0))*0.299+float(myColor(1))*0.587+float(myColor(2))*0.114-128)/255;      //Y
                        colorPatterns.at<float>(ctri,sampId) = (colorOffset(0)*0.299+colorOffset(1)*0.587+colorOffset(2)*0.114-128)/255;      //Y
                        if((l1%2==0)&&(l2%2==0))
                        {
                            colorPatterns.at<float>(ctri,chromaSampId+samplesNumber) = (-colorOffset(0)*0.168736-colorOffset(1)*0.331264+colorOffset(2)*0.5)/255;     //Cb
                            colorPatterns.at<float>(ctri,chromaSampId+samplesNumber+chromaSamplesNumber) = (colorOffset(0)*0.5-colorOffset(1)*0.418688-colorOffset(2)*0.081312)/255;  //Cr
                            ++chromaSampId;
                        }
                        //colorPatterns.at<float>(ctri,sampId+samplesNumber) = (-float(myColor(0))*0.168736-float(myColor(1))*0.331264+float(myColor(2))*0.5)/255;     //Cb
                        //colorPatterns.at<float>(ctri,sampId+2*samplesNumber) = (float(myColor(0))*0.5-float(myColor(1))*0.418688-float(myColor(2))*0.081312)/255;  //Cr

                        ++sampId;
                    }
                }
                ++ctri;
            }
        }
        //compute PCA
        int pc_number = std::min(samplesNumber+2*chromaSamplesNumber,colorPatterns.rows-1);
        log(ALWAYS)<<"pc_number = "<<pc_number<<endLog();
        cv::PCA pca (colorPatterns, cv::Mat(), cv::PCA::DATA_AS_ROW, pc_number);       //with no precomputed mean
        log(ALWAYS)<<"colorPatterns size: "<<colorPatterns.rows<<endLog();
        
        //compress and backproject
        cv::Mat reconstructed;
        cv::Mat compressed;
        cv::Mat absCompressed;
        compressed.create(colorPatterns.rows,pc_number,colorPatterns.type());
        reconstructed.create(colorPatterns.rows, colorPatterns.cols, colorPatterns.type());
        log(ALWAYS)<<"compressed size: ("<<compressed.rows<<","<<compressed.cols<<") (res "<<targetRes<<")"<<endLog();
        for(int i=0;i<colorPatterns.rows;++i)
        {
            cv::Mat coeffs = compressed.row(i);
            cv::Mat vec = colorPatterns.row(i);
            pca.project(vec,coeffs);
        }

        absCompressed = abs(compressed);
        cv::Mat squareCompressed = compressed.mul(compressed);
        cv::Mat row_mean, row_meanSquare, row_min, row_max;
        cv::Mat absMax;
        reduce(absCompressed,absMax,0,CV_REDUCE_MAX);
        //normalize projection space according to max value of principal component.
        int quantMultiplier = int(quantFactor/(absMax.at<float>(0)));
        cv::Mat eigenVec = pca.mean;

        eigenVec.push_back(pca.eigenvectors);

        resEigenVecList[targetRes] = ((std::pow(2,QUANT_BITS-1)-1)*eigenVec);        //puts everything in the coding range ([-32768,32767] by default, if QUANT_BITS=16, to be coded on two bytes)

        quantMultipliers[targetRes] = quantMultiplier;
        log(ALWAYS)<<"quantFactor = "<<quantFactor<<endLog();
        log(ALWAYS)<<"quantMultiplier res "<<targetRes<<": "<<quantMultiplier<<endLog();
        log(ALWAYS)<<"Abs max = "<<(absMax.at<float>(0))<<endLog();
        cv::Mat eigenVecAbsMax;
        cv::Mat absEigenVec;
        absEigenVec = abs(eigenVec);
        cv::Mat finalAbsEigenVec;
        reduce(absEigenVec,eigenVecAbsMax,0,CV_REDUCE_MAX);
        reduce(eigenVecAbsMax,finalAbsEigenVec,0,CV_REDUCE_MAX);
        log(ALWAYS)<<"Eigen Vec max = "<<(finalAbsEigenVec.at<float>(0))<<endLog();


        reduce(absCompressed,row_mean,0,CV_REDUCE_AVG);
        reduce(squareCompressed,row_meanSquare,0,CV_REDUCE_AVG);
        reduce(compressed,row_max,0,CV_REDUCE_MAX);
        reduce(compressed,row_min,0,CV_REDUCE_MIN);

        std::vector<float> quantizationMat;
        quantizationMat.clear();
        quantizationMat.reserve(pc_number);
        for(int i=0;i<pc_number;++i)
        {
            if(targetRes>1)
            {
                float quantCoef = (quantMatCoefs[0] + quantMatCoefs[1]*i+quantMatCoefs[2]*i*i);
                if (quantCoef>255)  //written on one byte in the binary file. Available range is [0,255]
                {
                    quantCoef=255;
                }
                quantizationMat.push_back(floor(quantCoef));
            }
            else
            {
                quantizationMat.push_back(1);
            }
            //quantizationMat.push_back(1);
        }

        // --- quantization ---
        log(ALWAYS)<<"samplesNumber: "<<samplesNumber<<", quantizationMat length: "<<quantizationMat.size()<<endLog();
        for(int i=0;i<colorPatterns.rows;++i)      //trying to remove big frequency components
        {
            for(int j=0;j<pc_number;++j)
            {
                compressed.at<float>(i,j) = floor(0.5+compressed.at<float>(i,j)*float(quantMultiplier)/float(quantizationMat[j]));
            }
        }


        cv::Mat coefSum(1,pc_number,compressed.type());
        cv::reduce(compressed,coefSum,0,CV_REDUCE_MAX);

        //do not bother encoding components that are never expressed (takes extra space for quantization matrix and PCA decomposition)
        int componentsKeptNumber = pc_number;
        while(std::abs(coefSum.at<float>(0,componentsKeptNumber-1))<0.5)    //i.e. = 0. comparison because float
        {
            --componentsKeptNumber;
        }
        cv::Mat truncatedCompressed = compressed.colRange(0,componentsKeptNumber);

        log(ALWAYS)<<"compressed size: ("<<truncatedCompressed.rows<<","<<truncatedCompressed.cols<<")"<<endLog();
        //end encoding
        resDCTList[targetRes]=truncatedCompressed.t();
        log(ALWAYS)<<"Truncating QT"<<endLog();
        quantizationMat.erase(quantizationMat.begin()+componentsKeptNumber,quantizationMat.end());
        log(ALWAYS)<<"Adding QT"<<endLog();
        resQTList[targetRes]=quantizationMat;
        log(ALWAYS)<<"Adding PCA"<<endLog();
        //rescvPCAList[targetRes] = pca;

    }       //end resolution loop
    //encode data
    log(ALWAYS)<<"encoding data..."<<endLog();
    //std::string compFileName = "comp_"+std::to_string(quantFactor)+"_"+std::to_string(10*quantMatCoefs[1])+"_"+std::to_string(100*quantMatCoefs[2]);
    // std::string compFileName = (boost::format("comp%1im_%i_%.1f_%.1f") % downsamplingThreshold % quantFactor % (10*quantMatCoefs[1]) % (100*quantMatCoefs[2])).str();
                

    meshEncoder.writeJPEGMeshColor(resDCTList, resQTList, resEigenVecList, quantMultipliers, outputPath, compFileName);
    //out_bit_array = meshEncoder.getMeshBinaryColor(resDCTList, in_faces.size());
    log(ALWAYS)<<"data encoded: "<<outputPath<<compFileName<<endLog();
    
    return;

}



template<class InTriangle, class InPoint, class InColor, class MySpecialMesh>
void SpaceTimeSampler::decodeCompressedColor(   MySpecialMesh *in_mesh,
                                        int default_face_res,
                                        int quantFactor
                                    )const
{
    log(ALWAYS)<<"Starting decodeCompressedColor"<<endLog();
    
    // std::string outputPath = output_folder_;
    // TEMP: (TODO: change)
    int downsamplingThreshold=-1;
    float quantMatCoefs[] = {1.0,1.0,0.01};
    // std::string compFileName = (boost::format("comp%1im_%i_%.1f_%.1f") % downsamplingThreshold % quantFactor % (10*quantMatCoefs[1]) % (100*quantMatCoefs[2])).str();
    
    std::string compFileName = in_mesh->getAppearanceFileName();
    
    std::vector<InTriangle> in_faces;
    std::vector<InPoint> in_points;
    std::vector<InColor> out_colors;
    std::vector<InColor> old_out_colors;
    std::vector<BitArray> out_bit_array;
    std::vector<unsigned short> out_face_res;
    std::vector<Vector3li> temp_edge_color_ind;
    std::vector<unsigned long> out_face_color_ind;

    out_face_res.clear();
    temp_edge_color_ind.clear();
    out_face_color_ind.clear();

    in_mesh->getFacesVector(in_faces);
    in_mesh->getPointsVector(in_points);
    in_mesh->getColorsVector(old_out_colors);

    out_face_res.clear();
    out_face_res.reserve(in_faces.size());
    out_face_color_ind.clear();
    out_face_color_ind.reserve(in_faces.size());

    int K = 20;

    cv::Mat vNeighbourMat = cv::Mat::zeros(in_points.size(),K,CV_32S);
    cv::Mat vColorIndMat = cv::Mat::zeros(in_points.size(),K,CV_32S);

    std::vector<std::vector<int>> vNeighbourVec(in_points.size());
    std::vector<std::vector<int>> vColorIndVec(in_points.size());

    std::vector<int> vMatInd(in_points.size(),0);


    int old_out_colors_size = old_out_colors.size();
    out_colors.clear();

    out_colors = std::vector<InColor>(old_out_colors_size + 18000000);

    for(int myInd=0;myInd<in_points.size();++myInd)
    {
        out_colors[myInd]=old_out_colors[myInd];
    }


    std::vector<int32_t> triverts(3);

    bool processFullTri = true;
    int samplesNumber, chromaSamplesNumber;
    int minIndex, maxIndex, triLength, chromaTriLength;
    //decode
    std::map<int,cv::Mat> resEigenVecList;
    resEigenVecList.clear();
    MeshEncoder meshEncoder = MeshEncoder();

    std::map<int,float> quantMultipliers;
    log(ALWAYS)<<"decoding from "<<compFileName<<endLog();
    std::map<int,cv::Mat> resPCAList = meshEncoder.decodeCompressedData(resEigenVecList,compFileName);
    log(ALWAYS)<<"decoding complete"<<endLog();


    unsigned long color_index_pointer = in_points.size();            //used as an incremented pointer into color_map for edges and faces values


    std::vector<size_t> edge_indices;               //temp vector to store a list of edges with their color indices
    edge_indices.clear();                           //makes color reindexing easier down the road
    edge_indices.reserve(in_faces.size()*3/2);
    edge_indices.push_back(0);

    log(ALWAYS)<<"Out_colors size = "<<out_colors.size()<<endLog();
    log(ALWAYS)<<"in_points size = "<<in_points.size()<<endLog();
    long globalTriInd=0;

    int lostEdgePixels=0;
    //start decoding
    for(int targetRes = default_face_res;targetRes>=2;targetRes=targetRes/2)
    {

        cv::Mat compressed=resPCAList[targetRes];
        log(ALWAYS)<<"res "<<targetRes<<", compressed size ("<<compressed.rows<<","<<compressed.cols<<")"<<endLog();
        if(compressed.cols==0)
        {
            continue;
        }
        //redefine local variables
        if(processFullTri)
        {
            //compressing edge samples with it
            samplesNumber = (targetRes+2)*(targetRes+1)/2-3;
            chromaSamplesNumber = (targetRes/2+2)*(targetRes/2+1)/2-3;
            minIndex=0;
            maxIndex=targetRes;
            triLength = targetRes+1;
            chromaTriLength = targetRes/2+1;
        }
        else
        {
            samplesNumber = (targetRes-2)*(targetRes-1)/2;
            chromaSamplesNumber = (targetRes/2-2)*(targetRes/2-1)/2;
            minIndex=1;
            maxIndex=targetRes-1;
            triLength = targetRes-2;
            chromaTriLength = targetRes/2-2;
        }

        int colorSpaceSize = samplesNumber+2*chromaSamplesNumber;

        int pc_number = compressed.cols;
        
        // --- reprojection ---
        cv::Mat reconstructed;
        reconstructed.create(compressed.rows, colorSpaceSize, compressed.type());

        cv::Mat newEigenVectors;

        resEigenVecList[targetRes].convertTo(newEigenVectors, CV_32FC1, 1/(std::pow(2,QUANT_BITS-1)-1));
        
        cv::Mat pcaMeanVector = newEigenVectors.row(0);     //mean vector is stored as the 1st row
        newEigenVectors = newEigenVectors.rowRange(1,newEigenVectors.rows);
        if(newEigenVectors.cols>newEigenVectors.rows)       //pad with null components if need be, so that there are enough eigen vectors
        {
            cv::Mat eigenPadding = cv::Mat::zeros(newEigenVectors.cols-newEigenVectors.rows,newEigenVectors.cols,CV_32FC1);
            newEigenVectors.push_back(eigenPadding);
        }
        
        log(ALWAYS)<<"compressed size: ("<<compressed.rows<<","<<compressed.cols<<")"<<endLog();
        log(ALWAYS)<<"colorSpaceSize = "<<colorSpaceSize<<endLog();
        cv::Mat zPad(compressed.rows,colorSpaceSize-compressed.cols,compressed.type(),0.0f);

        cv::hconcat(compressed,zPad,compressed);

        cv::Mat oneColumn = cv::Mat::ones(compressed.rows,1, CV_32FC1);

        reconstructed = compressed * newEigenVectors + oneColumn * pcaMeanVector;   //add row vector 'pcaMeanVector' to every row of 'reconstructed'
        log(ALWAYS)<<"reconstructed size: ("<<reconstructed.rows<<","<<reconstructed.cols<<")"<<endLog();

        int ctri=0;
        log(ALWAYS)<<"Rewriting..."<<endLog();
        //rewrite in_colors with compressed->uncompressed data

        // for(int tri=globalTriInd;tri<in_faces.size();++tri)
        for(int tri=globalTriInd;tri<globalTriInd+compressed.rows;++tri)
        {
            if(color_index_pointer>=out_colors.size())
            {
                log(ERROR)<<"ERROR: bad color_index_pointer: res "<<targetRes<<", tri "<<tri<<", ctri "<<ctri<<endLog();
                log(ERROR)<<"color_index_pointer = "<<color_index_pointer<<endLog();
                return;
            }
            unsigned long face_color_index=color_index_pointer;   //reserve space to write this face's colors
            if(targetRes==1||targetRes==2)
                face_color_index=0;
            color_index_pointer+=(targetRes-1)*(targetRes-2)/2;
            
            InTriangle &triangle = in_faces[tri];
            triverts[0] = triangle.ref;
            triverts[1] = triangle.edge1;
            triverts[2] = triangle.edge2;
            Vector3li edge_ind = Vector3li(0,0,0);
            long current_color_index;
            
            //with vertices and edges version
            int sampId=0;
            int chromaSampId=0;
            InColor myColor;

            for(int l1=minIndex;l1<=maxIndex;++l1)    //loop through barycentric coordinates
            {
                for(int l2=minIndex;l2<=maxIndex-l1;++l2)
                {
                    if((l1+l2==0)||(l1==maxIndex)||(l2==maxIndex))
                    {
                        continue;
                    }
                    float Y,Cb,Cr;
                    //decimated chroma version
                    if(ctri>=reconstructed.rows)
                    {
                        log(ALWAYS)<<"WARNING!!! triangle out of range: tri "<<ctri<<", res="<<targetRes<<", rows: "<<reconstructed.rows<<endLog();
                    }
                    if(sampId>=reconstructed.cols)
                    {
                        log(ALWAYS)<<"WARNING!! sample id out of range: tri "<<ctri<<", sampId = "<<sampId<<endLog();
                    }
                    
                    Y = 255*reconstructed.at<float>(ctri,sampId);
                    Cb=0;
                    Cr=0;

                    if((l1%2==0)&&(l2%2==0))
                    {
                        if(chromaSamplesNumber+samplesNumber+chromaSampId>=reconstructed.cols)
                        {
                            log(ALWAYS)<<"WARNING!! chroma sample id out of range: tri "<<ctri<<", chromaSampId = "<<chromaSampId<<", l1,l2 = ("<<l1<<","<<l2<<")"<<endLog();
                        }

                        Cb = 255*reconstructed.at<float>(ctri,chromaSampId+samplesNumber);
                        Cr = 255*reconstructed.at<float>(ctri,chromaSampId+samplesNumber+chromaSamplesNumber);
                        ++chromaSampId;
                    }


                    float R = Y + 1.402 * Cr + 128;
                    float G = Y - 0.344136 * Cb - 0.714136 * Cr + 128;
                    float B = Y + 1.772 * Cb + 128;

                    InColor myColor = InColor(int(std::min(255.0,std::max(0.0,R+0.5))),int(std::min(255.0,std::max(0.0,G+0.5))),int(std::min(255.0,std::max(0.0,B+0.5))));

                    int edgeSampInd=-1;
                    int edge_num, edge_res;
                    int vI=-1;
                    int vI2=-1;

                    bool edgeMean=false;

                    if(l1==0)   //2nd edge (v3,v2)
                    {
                        vI=2;
                        vI2=1;
                        edge_num=1;
                        edgeSampInd=l2;
                    }
                    else if (l2==0)     //3rd edge (v1,v3)
                    {
                        vI=0;
                        vI2=2;
                        edge_num=2;
                        edgeSampInd=targetRes-l1;
                    }
                    else if (l1+l2==targetRes)     //1st edge, (v2,v1)
                    {
                        vI=1;
                        vI2=0;
                        edge_num=0;
                        edgeSampInd=l1;
                    }

                    if(vI>=0)       //Edge
                    {
                        int vert0 = triverts[vI];
                        int vert1 = triverts[vI2];
                        
                        int edgeColInd=0;
                        for(int tempInd = 0;tempInd<vMatInd[vert0];++tempInd)
                        {

                            if(vNeighbourVec[vert0][tempInd]==vert1)
                            {
                                edgeColInd=vColorIndVec[vert0][tempInd];
                                break;
                            }
                        }
                        
                        if(edgeColInd>0)
                        {
                            current_color_index = edge_indices[edgeColInd];
                            edge_ind(edge_num,0)=edgeColInd;
                        }
                        else
                        {
                            
                            if(edgeColInd<0)
                            {

                                int temp = vI;      //does this work as intended?
                                vI = vI2;
                                vI2=temp;
                                current_color_index = edge_indices[-edgeColInd];    //get color index from edge map

                                edge_ind(edge_num,0)=edgeColInd;
                                edgeMean=true;
                            }


                            else    //not recorded yet
                            {
                                current_color_index = color_index_pointer;

                                edge_ind(edge_num,0)=edge_indices.size();

                                vNeighbourVec[vert0].push_back(vert1);
                                vNeighbourVec[vert1].push_back(vert0);
                                vColorIndVec[vert0].push_back(edge_indices.size());
                                vColorIndVec[vert1].push_back(-edge_indices.size());

                                vMatInd[vert0]+=1;
                                vMatInd[vert1]+=1;

                                if (vMatInd[vert0]>=K){
                                    log(WARN)<<"WARNING: saturated neighbours: vert "<<vert0<<endLog();
                                }
                                if (vMatInd[vert1]>=K){
                                    log(WARN)<<"WARNING: saturated neighbours: vert "<<vert1<<endLog();
                                }
                                edge_indices.push_back(color_index_pointer);
                                color_index_pointer+=targetRes;   //faceRes-1 color samples + 1 value for edge Res
                                //Display edge res in cyan for debugging

                                out_colors[current_color_index] = InColor(targetRes,255,255);      //record edgeRes
                            }
                        }

                        if(out_colors[current_color_index](0)>targetRes){        //this triangle has lower resolution than the one sharing its edge.
                            lostEdgePixels+=out_colors[current_color_index](0)-targetRes;
                            edge_res=out_colors[current_color_index](0);    //In this case, we choose the lower resolution for this edge, for a more consistent looking image
                            out_colors[current_color_index](0)=targetRes;

                            for(int edge_i=1;edge_i<=targetRes-1;edge_i++){            //Drop extra samples and move the "good" ones to their new place
                                out_colors[current_color_index+edge_i] = out_colors[current_color_index+(edge_res*edge_i/targetRes)];
                            }
                        }

                        edge_res=out_colors[current_color_index](0);
                        if((edgeSampInd*edge_res)%targetRes==0)          //if the edge has a lower resolution than the face, only add vote if the current edge point is a sample
                        {
                            current_color_index+=edgeSampInd*edge_res/targetRes;     // (-1 +1 because of 1st value used for the edgeRes)
                        }
                        else
                        {
                            current_color_index=-1;     //don't do anything
                        }
                        if(edge_ind(edge_num,0)<0)
                        {
                            current_color_index=-1;
                        }
                    }
                    //edge case taken care of. Only the face to go!!
                    else
                    {
                        current_color_index = face_color_index + (l1-1)*targetRes - (l1*(l1+1))/2 +1 + (l2-1);
                    }

                    if(current_color_index>=0)
                    {
                        if (edgeMean)
                        {
                            out_colors[current_color_index] = out_colors[current_color_index]/2 + myColor/2;
                            // out_colors[current_color_index](0) = (int)((float)(out_colors[current_color_index](0))*0.5+(float)(myColor(0))*0.5);
                            // out_colors[current_color_index](1) = (int)((float)(out_colors[current_color_index](1))*0.5+(float)(myColor(1))*0.5);
                            // out_colors[current_color_index](2) = (int)((float)(out_colors[current_color_index](2))*0.5+(float)(myColor(2))*0.5);
                        }
                        else
                        {
                            out_colors[current_color_index] = myColor;
                        }
                        // if((tri<1)&&((l1==0)||(l2==0)||(l1+l2==targetRes)))
                        // {
                        //     log(ALWAYS)<<"Tri "<<tri<<", samp ("<<l1<<","<<l2<<"): color value = ("<<out_colors[current_color_index](0)<<","<<out_colors[current_color_index](1)<<","<<out_colors[current_color_index](2)<<")"<<endLog();
                        //     log(ALWAYS)<<"face_color_index = "<<face_color_index<<", current_color_index = "<<current_color_index<<endLog();
                        // }
                    }
                    //increment index
                    ++sampId;
                }
            }

            if(face_color_index>out_colors.size())
            {
                log(ERROR)<<"ERROR: bad face color ind: res "<<targetRes<<", tri "<<tri<<endLog();
                log(ERROR)<<"face_color_index = "<<face_color_index<<endLog();
            }
            out_face_color_ind.push_back(face_color_index);
            temp_edge_color_ind.push_back(edge_ind);
            out_face_res.push_back(targetRes);

            ++ctri;

        }   //end rewriting
        log(ALWAYS)<<"rewriting finished for res "<<targetRes<<endLog();
        globalTriInd+=compressed.rows;
    }

    log(ALWAYS)<<"rewriting finished"<<endLog();
    // in_colors=temp_colors;
    log(ALWAYS)<<"compression finished..."<<endLog();

    out_colors[0](0)=1;  //for edges with res 1

    log(ALWAYS)<<"color_index_pointer = "<<color_index_pointer<<endLog();
    log(ALWAYS)<<"lostEdgePixels = "<<lostEdgePixels<<endLog();
    log(ALWAYS)<<"out_face_color_ind size = "<<out_face_color_ind.size()<<endLog();
    log(ALWAYS)<<"temp_edge_color_ind size = "<<temp_edge_color_ind.size()<<endLog();
    log(ALWAYS)<<"out_face_res size = "<<out_face_res.size()<<endLog();
    log(ALWAYS)<<"edge_indices size = "<<edge_indices.size()<<endLog();
    log(ALWAYS)<<"in_faces size = "<<in_faces.size()<<endLog();

    out_colors.erase(out_colors.begin()+color_index_pointer,out_colors.end());

    Vector3li edge0 = Vector3li(0,0,0);
    for(int triInd=globalTriInd;triInd<in_faces.size();++triInd)
    {
        out_face_color_ind.push_back(0);
        out_face_res.push_back(1);
        temp_edge_color_ind.push_back(edge0);
    }

    std::vector<Vector3li> out_edge_color_ind(temp_edge_color_ind.size());
    for(int tri=0;tri<in_faces.size();++tri)
    {
        for(int e=0;e<3;++e)
        {
            if (std::abs(temp_edge_color_ind[tri](e))>=edge_indices.size())
            {
                log(ERROR)<<"case 0: temp_edge_color_ind[tri](e) = "<<temp_edge_color_ind[tri](e)<<endLog();
                log(ERROR)<<"edge_indices.size() = "<<edge_indices.size()<<endLog();
            }
            if (tri>=temp_edge_color_ind.size())
            {
                log(ERROR)<<"case 1"<<endLog();
            }
            if (tri>=out_edge_color_ind.size())
            {
                log(ERROR)<<"case 2"<<endLog();
            }
            if (temp_edge_color_ind[tri](e)>=0)
                out_edge_color_ind[tri](e) = edge_indices[temp_edge_color_ind[tri](e)];
            else
                out_edge_color_ind[tri](e) = - edge_indices[- temp_edge_color_ind[tri](e)];
        }
    }

    log(ALWAYS)<<"Edges ok"<<endLog();


    for(int tri=0;tri<in_faces.size();++tri)
    {
        bool writeLog=false;
        InTriangle myTri = in_faces[tri];
        unsigned short myRes = out_face_res[tri];
        long v1I = myTri.ref;
        long v2I = myTri.edge1;
        long v3I = myTri.edge2;
        long e1I = out_edge_color_ind[tri](0);
        long e2I = out_edge_color_ind[tri](1);
        long e3I = out_edge_color_ind[tri](2);
        downsampleEdgeChroma(e1I, v2I, v1I, out_colors,writeLog);
        downsampleEdgeChroma(e2I, v3I, v2I, out_colors,false);
        downsampleEdgeChroma(e3I, v1I, v3I, out_colors,false);
        
        downsampleTriangleChroma(tri,myTri,myRes,out_face_color_ind,out_edge_color_ind,out_colors);
    }
    log(ALWAYS)<<"Downsampling ok"<<endLog();

    //Update input mesh
    in_mesh->setColorsVector(out_colors);
    in_mesh->setFacesResVector(out_face_res);
    in_mesh->setEdgesIndVector(out_edge_color_ind);
    in_mesh->setFacesIndVector(out_face_color_ind);
    in_mesh->setColorsBitArray(out_bit_array);
    in_mesh->setEdgesRealColorInd(edge_indices);

}






template<class InColor, class InTriangle>
int SpaceTimeSampler::downsampleTriangle(   unsigned long tri,
                                            const InTriangle &myTri,
                                            int triRes,
                                            const std::vector<unsigned long>&in_face_color_ind,
                                            const std::vector<Vector3li> &in_edge_color_ind,
                                            std::vector<InColor> &in_colors,
                                            float maxIPThreshold
                                            )const
{
    //Get edges resolution
    int e0Res = in_colors[std::abs(in_edge_color_ind[tri](0))](0);
    int e1Res = in_colors[std::abs(in_edge_color_ind[tri](1))](0);
    int e2Res = in_colors[std::abs(in_edge_color_ind[tri](2))](0);
    unsigned long myTri_color_ind = in_face_color_ind[tri];
    //float maxIPThreshold = 15;
    while(triRes>std::max(e0Res,std::max(e1Res,e2Res)))     //A triangle's resolution must be at least equal to its edges resolution. (Thus, we downsample edges first).
    {
        //loop over color samples
        for(int i=1;i<triRes-1;++i)
        {
            for(int j=1;j<triRes-i;++j)
            {
                if((i%2==1)||(j%2==1))      //sample is not present in the lower level of resolution if at least one of its barycentric index is odd
                {
                    //compare sample value with value obtained by interpolation
                    //each sample is in the middle of a segment between two lower-level samples.
                    int indexOffset = (i-1)*triRes - i*(i+1)/2 + j;
                    InColor sampColor = in_colors[myTri_color_ind+indexOffset];
                    InColor n1, n2;     //neighbours

                    //get neighbouring samples
                    if(i%2==0)  //neighbours are (i,j-1) and (i,j+1)
                    {
                        //n1
                        if(j-1==0)  //edge sample
                        {
                            n1 = getSampleColor(myTri, tri, triRes, i, j-1, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            n1 = in_colors[myTri_color_ind+indexOffset-1];
                        }
                        //n2
                        if(j+1+i==triRes)   //edge sample
                        {
                            n2 = getSampleColor(myTri, tri, triRes, i, j+1, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            n2 = in_colors[myTri_color_ind+indexOffset+1];
                        }
                    }
                    else if(j%2==0) //neighbours are (i-1,j) and (i+1,j)
                    {
                        //n1
                        if(i-1==0)  //edge sample
                        {
                            n1 = getSampleColor(myTri, tri, triRes, i-1, j, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            int n1Offset = (i-2)*triRes - i*(i-1)/2 + j;
                            n1 = in_colors[myTri_color_ind+n1Offset];
                        }
                        //n2
                        if(i+1+j==triRes)   //edge sample
                        {
                            n2 = getSampleColor(myTri, tri, triRes, i+1, j, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            int n2Offset = i*triRes - (i+2)*(i+1)/2 + j;
                            n2 = in_colors[myTri_color_ind+n2Offset];
                        }
                    }
                    else    //j and i both odd. neighbours are (i-1,j+1) and (i+1,j-1)
                    {
                        //n1
                        if(i-1==0)      //edge samples
                        {
                            n1 = getSampleColor(myTri, tri, triRes, i-1, j+1, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            int n1Offset = (i-2)*triRes - i*(i-1)/2 + j+1;
                            n1 = in_colors[myTri_color_ind+n1Offset];
                        }
                        //n2
                        if(j-1==0)      //edge sample
                        {
                            n2 = getSampleColor(myTri, tri, triRes, i+1, j-1, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            int n2Offset = i*triRes - (i+2)*(i+1)/2 + j-1;
                            n2 = in_colors[myTri_color_ind+n2Offset];
                        }
                    }
                    //interpolate and compute distance
                    float sampDist = pow(0.5*float(n1(0)+n2(0))-float(sampColor(0)),2) + pow(0.5*float(n1(1)+n2(1))-float(sampColor(1)),2) + pow(0.5*float(n1(2)+n2(2))-float(sampColor(2)),2);
                    //compare to threshold
                    if(sampDist>maxIPThreshold)
                    {
                        //keep resolution for this face. Get out of here
                        return triRes;
                    }
                    
                }
            }
        }
        //We've been through all samples and they satisfy the condition: downsample
        triRes/=2;
        //rewrite samples
        int newOffset=0;
        //loop over color samples
        for(int i=1;i<triRes-1;++i)
        {
            for(int j=1;j<triRes-i;++j)
            {
                int indexOffset = (2*i-1)*2*triRes - i*(2*i+1) + 2*j;
                InColor sampColor = in_colors[myTri_color_ind+indexOffset];
                in_colors[myTri_color_ind+newOffset]=sampColor;
                ++newOffset;
            }
        }

    }
    //We've reached the max edge resolution.
    //We cannot do more
    return triRes;
}


template<class InColor, class InTriangle>
int SpaceTimeSampler::downsampleTriangleMean(   unsigned long tri,
                                            const InTriangle &myTri,
                                            int triRes,
                                            const std::vector<unsigned long>&in_face_color_ind,
                                            const std::vector<Vector3li> &in_edge_color_ind,
                                            std::vector<InColor> &in_colors,
                                            float maxIPThreshold
                                            )const
{
    //Get edges resolution
    int e0Res = in_colors[std::abs(in_edge_color_ind[tri](0))](0);
    int e1Res = in_colors[std::abs(in_edge_color_ind[tri](1))](0);
    int e2Res = in_colors[std::abs(in_edge_color_ind[tri](2))](0);
    unsigned long myTri_color_ind = in_face_color_ind[tri];
    while(triRes>std::max(e0Res,std::max(e1Res,e2Res)))     //A triangle's resolution must be at least equal to its edges resolution. (Thus, we downsample edges first).
    {
        float sampDist=0.0f;
        int sampNum=0;
        if(triRes==1)
        {
            return triRes;
        }

        //loop over color samples
        for(int i=1;i<triRes-1;++i)
        {
            for(int j=1;j<triRes-i;++j)
            {
                if((i%2==1)||(j%2==1))      //sample is not present in the lower level of resolution if at least one of its barycentric index is odd
                {
                    //compare sample value with value obtained by interpolation
                    //each sample is in the middle of a segment between two lower-level samples.
                    int indexOffset = (i-1)*triRes - i*(i+1)/2 + j;
                    InColor sampColor = in_colors[myTri_color_ind+indexOffset];
                    InColor n1, n2;     //neighbours

                    //get neighbouring samples
                    if(i%2==0)  //neighbours are (i,j-1) and (i,j+1)
                    {
                        //n1
                        if(j-1==0)  //edge sample
                        {
                            n1 = getSampleColor(myTri, tri, triRes, i, j-1, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            n1 = in_colors[myTri_color_ind+indexOffset-1];
                        }
                        //n2
                        if(j+1+i==triRes)   //edge sample
                        {
                            n2 = getSampleColor(myTri, tri, triRes, i, j+1, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            n2 = in_colors[myTri_color_ind+indexOffset+1];
                        }
                    }
                    else if(j%2==0) //neighbours are (i-1,j) and (i+1,j)
                    {
                        //n1
                        if(i-1==0)  //edge sample
                        {
                            n1 = getSampleColor(myTri, tri, triRes, i-1, j, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            int n1Offset = (i-2)*triRes - i*(i-1)/2 + j;
                            n1 = in_colors[myTri_color_ind+n1Offset];
                        }
                        //n2
                        if(i+1+j==triRes)   //edge sample
                        {
                            n2 = getSampleColor(myTri, tri, triRes, i+1, j, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            int n2Offset = i*triRes - (i+2)*(i+1)/2 + j;
                            n2 = in_colors[myTri_color_ind+n2Offset];
                        }
                    }
                    else    //j and i both odd. neighbours are (i-1,j+1) and (i+1,j-1)
                    {
                        //n1
                        if(i-1==0)      //edge samples
                        {
                            n1 = getSampleColor(myTri, tri, triRes, i-1, j+1, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            int n1Offset = (i-2)*triRes - i*(i-1)/2 + j+1;
                            n1 = in_colors[myTri_color_ind+n1Offset];
                        }
                        //n2
                        if(j-1==0)      //edge sample
                        {
                            n2 = getSampleColor(myTri, tri, triRes, i+1, j-1, in_edge_color_ind, in_face_color_ind, in_colors);
                        }
                        else    //face sample
                        {
                            int n2Offset = i*triRes - (i+2)*(i+1)/2 + j-1;
                            n2 = in_colors[myTri_color_ind+n2Offset];
                        }
                    }
                    //interpolate and compute distance
                    sampDist += pow(0.5*float(n1(0)+n2(0))-float(sampColor(0)),2) + pow(0.5*float(n1(1)+n2(1))-float(sampColor(1)),2) + pow(0.5*float(n1(2)+n2(2))-float(sampColor(2)),2);
                    ++sampNum;
                }
            }
        }
        //compare to threshold
        if(sampDist>maxIPThreshold*sampNum)
        {
            return triRes; //keep resolution for this face. Get out of here
        }
        //We've been through all samples and they satisfy the condition: downsample
        triRes/=2;
        //rewrite samples
        int newOffset=0;
        //loop over color samples
        for(int i=1;i<triRes-1;++i)
        {
            for(int j=1;j<triRes-i;++j)
            {
                int indexOffset = (2*i-1)*2*triRes - i*(2*i+1) + 2*j;
                InColor sampColor = in_colors[myTri_color_ind+indexOffset];
                in_colors[myTri_color_ind+newOffset]=sampColor;
                ++newOffset;
            }
        }
    }
    //We've reached the max edge resolution.
    //We cannot do more
    return triRes;
}

template <class InTriangle, class InColor>
void SpaceTimeSampler::downsampleMeshColor( std::vector<InTriangle> &in_faces,
                                        std::vector<InColor> &in_colors,
                                        std::vector<unsigned short> &in_face_res,
                                        std::vector<Vector3li> &in_edge_color_ind,
                                        std::vector<unsigned long> &in_face_color_ind,
                                        int downsamplingThreshold)const
{
    if(downsamplingThreshold>=0)
    {
        log(ALWAYS)<<"[SpaceTimeSampler] : Downsampling..."<<endLog();
        for(int tri=0;tri<in_faces.size();++tri)
        {
            InTriangle myTri = in_faces[tri];
            long v1I = myTri.ref;
            long v2I = myTri.edge1;
            long v3I = myTri.edge2;
            long e1I = in_edge_color_ind[tri](0);
            long e2I = in_edge_color_ind[tri](1);
            long e3I = in_edge_color_ind[tri](2);
            // downsampleEdgeMeanTemp(e1I, v2I, v1I, in_colors,downsamplingThreshold);
            // downsampleEdgeMeanTemp(e2I, v3I, v2I, in_colors,downsamplingThreshold);
            // downsampleEdgeMeanTemp(e3I, v1I, v3I, in_colors,downsamplingThreshold);
            downsampleEdgeMean(e1I, v2I, v1I, in_colors,downsamplingThreshold);
            downsampleEdgeMean(e2I, v3I, v2I, in_colors,downsamplingThreshold);
            downsampleEdgeMean(e3I, v1I, v3I, in_colors,downsamplingThreshold);
            in_face_res[tri] = downsampleTriangleMean(tri,myTri,in_face_res[tri],in_face_color_ind,in_edge_color_ind,in_colors,downsamplingThreshold);
        }
        log(ALWAYS)<<"[SpaceTimeSampler] : Downsampling: done"<<endLog();
    }
}

// Takes point p of the surface, given by (myTri, baryCoords)
// Gets its 2D coordinates in image of cameraNumber
// Returns its color, using bilinear  interpolation
template<class InPoint, class InTriangle>
bool SpaceTimeSampler::getSurfacePointColor(InTriangle &myTri, const std::vector<InPoint> &in_points, Vector3f baryCoords, int cameraNumber, Vector3ui &out_color, bool writeLog, bool downsample)const
{
    const Camera &temp_cam = v_cameras_[cameraNumber];
    Vector3f cam_pos = temp_cam.getPosition();
    cv::Vec3b col,coltl,coltr,colbl,colbr;//BGR order

    bool is_safe = true;
    
    Vector2f ref_coords,edge1_coords,edge2_coords, tex_coords;
    temp_cam.getTextureCoords(in_points[myTri.ref],ref_coords);
    temp_cam.getTextureCoords(in_points[myTri.edge1],edge1_coords);
    temp_cam.getTextureCoords(in_points[myTri.edge2],edge2_coords);


    tex_coords(0) = ref_coords(0) * baryCoords(0) + edge1_coords(0) * baryCoords(1) + edge2_coords(0) * baryCoords(2);
    tex_coords(1) = ref_coords(1) * baryCoords(0) + edge1_coords(1) * baryCoords(1) + edge2_coords(1) * baryCoords(2);
    

    // int intX = int(floor(tex_coords(0)*2048-0.5f)+0.1f);
    // int intY = int(floor(tex_coords(0)*2048-0.5f)+0.1f);
    // float weightX = tex_coords(0)*2048-0.5f-intX;
    // float weightY = tex_coords(0)*2048-0.5f-intY;

    float intX;
    float intY;
    float weightX;
    float weightY;
    Vector2uli tl_coords, tr_coords, bl_coords, br_coords;
    
    
    bool safetl;
    bool safetr;
    bool safebl;
    bool safebr;

    if(downsample)
    {
        intX = floor((tex_coords(0)-0.5)/2);
        intY = floor((tex_coords(1)-0.5)/2);
        weightX = (tex_coords(0)-0.5)/2-intX;
        weightY = (tex_coords(1)-0.5)/2-intY;
        tl_coords = Vector2uli(2*intX,2*intY);
        tr_coords = Vector2uli(2*intX+2,2*intY);
        bl_coords = Vector2uli(2*intX,2*intY+2);
        br_coords = Vector2uli(2*intX+2,2*intY+2);
        safetl = temp_cam.getPixelColorDS(tl_coords,coltl);
        safetr = temp_cam.getPixelColorDS(tr_coords,coltr);
        safebl = temp_cam.getPixelColorDS(bl_coords,colbl);
        safebr = temp_cam.getPixelColorDS(br_coords,colbr);
    }
    else
    {
        intX = floor(tex_coords(0));
        intY = floor(tex_coords(1));
        weightX = tex_coords(0)-intX;
        weightY = tex_coords(1)-intY;
        tl_coords = Vector2uli(intX,intY);
        tr_coords = Vector2uli(intX+1,intY);
        bl_coords = Vector2uli(intX,intY+1);
        br_coords = Vector2uli(intX+1,intY+1);
        safetl = temp_cam.getPixelColor(tl_coords,coltl);
        safetr = temp_cam.getPixelColor(tr_coords,coltr);
        safebl = temp_cam.getPixelColor(bl_coords,colbl);
        safebr = temp_cam.getPixelColor(br_coords,colbr);
    }
    

    // temp_cam.getPixelColor(tex_coords,coltl);
    // float r=float(coltl[2]);
    // float g=float(coltl[1]);
    // float b=float(coltl[0]);
    // out_color = Vector3ui((unsigned int)(r),(unsigned int)(g),(unsigned int)(b));

    // return true;

    if((!safetl)&&(!safetr)&&(!safebl)&&(!safebr))
    {
        return false;
    }

    float wtl = (1-weightX)*(1-weightY);
    float wtr = weightX*(1-weightY);
    float wbl = (1-weightX)*weightY;
    float wbr = weightX*weightY;

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float w = 0.0f;

    if(safetl);
    {
        r+=float(coltl[2])*wtl;
        g+=float(coltl[1])*wtl;
        b+=float(coltl[0])*wtl;
        w+=wtl;
    }
    if(safetr)
    {
        r+=float(coltr[2])*wtr;
        g+=float(coltr[1])*wtr;
        b+=float(coltr[0])*wtr;
        w+=wtr;
    }
    if(safebl)
    {
        r+=float(colbl[2])*wbl;
        g+=float(colbl[1])*wbl;
        b+=float(colbl[0])*wbl;
        w+=wbl;
    }
    if(safebr)
    {
        r+=float(colbr[2])*wbr;
        g+=float(colbr[1])*wbr;
        b+=float(colbr[0])*wbr;
        w+=wbr;
    }

    r/=w;
    b/=w;
    g/=w;

    out_color = Vector3ui((unsigned int)(r),(unsigned int)(g),(unsigned int)(b));

    return true;

}

// SHOULD NOT BE USED!!
// Corrected bug in getSurfacePointColor. Most likely also present in this function
template<class InPoint, class InTriangle>
bool SpaceTimeSampler::getSurfacePointColorWVis(InTriangle &myTri, int triangle_idx, const std::vector<InPoint> &in_points, Vector3f baryCoords, int cameraNumber, Vector3ui &out_color, std::vector<cv::Mat> &cam_tri_ind, bool writeLog)const
{
    cv::Mat &cam_image = cam_tri_ind[cameraNumber];

    const Camera &temp_cam = v_cameras_[cameraNumber];
    Vector3f cam_pos = temp_cam.getPosition();
    cv::Vec3b col,coltl,coltr,colbl,colbr;//BGR order

    bool is_safe = true;
    
    Vector2f ref_coords,edge1_coords,edge2_coords, tex_coords;
    temp_cam.getTextureCoords(in_points[myTri.ref],ref_coords);
    temp_cam.getTextureCoords(in_points[myTri.edge1],edge1_coords);
    temp_cam.getTextureCoords(in_points[myTri.edge2],edge2_coords);


    tex_coords(0) = ref_coords(0) * baryCoords(0) + edge1_coords(0) * baryCoords(1) + edge2_coords(0) * baryCoords(2);
    tex_coords(1) = ref_coords(1) * baryCoords(0) + edge1_coords(1) * baryCoords(1) + edge2_coords(1) * baryCoords(2);
    

    // int intX = int(floor(tex_coords(0)*2048-0.5f)+0.1f);
    // int intY = int(floor(tex_coords(0)*2048-0.5f)+0.1f);
    // float weightX = tex_coords(0)*2048-0.5f-intX;
    // float weightY = tex_coords(0)*2048-0.5f-intY;

    float intX = floor(tex_coords(0));
    float intY = floor(tex_coords(1));
    float weightX = tex_coords(0)-intX;
    float weightY = tex_coords(1)-intY;
    Vector2uli tl_coords, tr_coords, bl_coords, br_coords;
    tl_coords = Vector2uli(intX,intY);
    tr_coords = Vector2uli(intX+1,intY);
    bl_coords = Vector2uli(intX,intY+1);
    br_coords = Vector2uli(intX+1,intY+1);
    bool safetl = temp_cam.getPixelColor(tl_coords,coltl);
    bool safetr = temp_cam.getPixelColor(tr_coords,coltr);
    bool safebl = temp_cam.getPixelColor(bl_coords,colbl);
    bool safebr = temp_cam.getPixelColor(br_coords,colbr);

    if((!safetl)&&(!safetr)&&(!safebl)&&(!safebr))
    {
        return false;
    }

    float wtl = (1-weightX)*(1-weightY);
    float wtr = weightX*(1-weightY);
    float wbl = (1-weightX)*weightY;
    float wbr = weightX*weightY;

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float w = 0.0f;

    triangle_idx+=1;    //Adding 1 because of offset in cam_image...

    if (writeLog)
    {
        log(ALWAYS)<<"current tri = "<<triangle_idx<<", cam num = "<<cameraNumber<<endLog();
        for(int delta_cfx=0;delta_cfx<CLEANING_FACTOR;++delta_cfx)
        {
            for(int delta_cfy=0;delta_cfy<CLEANING_FACTOR;++delta_cfy)
            {
                log(ALWAYS)<<"cam tri = "<<cam_image.at<int32_t>(intX*CLEANING_FACTOR+delta_cfx,intY*CLEANING_FACTOR+delta_cfy)<<endLog();
            }
        }
    }



    bool testPix = false;

    //Visibility per pixel
        // if(safetl);
        // {
        //     bool testPix = false;
        //     for(int delta_cfx=0;delta_cfx<CLEANING_FACTOR;++delta_cfx)
        //     {
        //         for(int delta_cfy=0;delta_cfy<CLEANING_FACTOR;++delta_cfy)
        //         {

        //             if (triangle_idx == cam_image.at<int32_t>(intX*CLEANING_FACTOR+delta_cfx,intY*CLEANING_FACTOR+delta_cfy))
        //             {
        //                 // r+=float(coltl[2])*wtl;
        //                 // g+=float(coltl[1])*wtl;
        //                 // b+=float(coltl[0])*wtl;
        //                 // w+=wtl;
        //                 testPix = true;
        //             }
        //         }
        //     }
        //     if (testPix)
        //     {
        //         r+=float(coltl[2])*wtl;
        //         g+=float(coltl[1])*wtl;
        //         b+=float(coltl[0])*wtl;
        //         w+=wtl;
        //     }

        // }
        // if(safetr)
        // {
        //     bool testPix = false;
        //     for(int delta_cfx=0;delta_cfx<CLEANING_FACTOR;++delta_cfx)
        //     {
        //         for(int delta_cfy=0;delta_cfy<CLEANING_FACTOR;++delta_cfy)
        //         {
        //             if (triangle_idx == cam_image.at<int32_t>((intX+1)*CLEANING_FACTOR+delta_cfx,intY*CLEANING_FACTOR+delta_cfy))
        //             {
        //                 // r+=float(coltr[2])*wtr;
        //                 // g+=float(coltr[1])*wtr;
        //                 // b+=float(coltr[0])*wtr;
        //                 // w+=wtr;
        //                 testPix = true;
        //             }
        //         }
        //     }
        //     if (testPix)
        //     {
        //         r+=float(coltr[2])*wtr;
        //         g+=float(coltr[1])*wtr;
        //         b+=float(coltr[0])*wtr;
        //         w+=wtr;
        //     }
        // }
        // if(safebl)
        // {
        //     bool testPix = false;
        //     for(int delta_cfx=0;delta_cfx<CLEANING_FACTOR;++delta_cfx)
        //     {
        //         for(int delta_cfy=0;delta_cfy<CLEANING_FACTOR;++delta_cfy)
        //         {
        //             if (triangle_idx == cam_image.at<int32_t>(intX*CLEANING_FACTOR+delta_cfx,(intY+1)*CLEANING_FACTOR+delta_cfy))
        //             {
        //                 // r+=float(colbl[2])*wbl;
        //                 // g+=float(colbl[1])*wbl;
        //                 // b+=float(colbl[0])*wbl;
        //                 // w+=wbl;
        //                 testPix = true;
        //             }
        //         }
        //     }
        //     if (testPix)
        //     {
        //         r+=float(colbl[2])*wbl;
        //         g+=float(colbl[1])*wbl;
        //         b+=float(colbl[0])*wbl;
        //         w+=wbl;
        //     }
        // }
        // if(safebr)
        // {
        //     bool testPix = false;
        //     for(int delta_cfx=0;delta_cfx<CLEANING_FACTOR;++delta_cfx)
        //     {
        //         for(int delta_cfy=0;delta_cfy<CLEANING_FACTOR;++delta_cfy)
        //         {
        //             if (triangle_idx == cam_image.at<int32_t>((intX+1)*CLEANING_FACTOR+delta_cfx,(intY+1)*CLEANING_FACTOR+delta_cfy))
        //             {
        //                 // r+=float(colbr[2])*wbr;
        //                 // g+=float(colbr[1])*wbr;
        //                 // b+=float(colbr[0])*wbr;
        //                 // w+=wbr;
        //                 testPix = true;
        //             }
        //         }
        //     }
        //     if (testPix)
        //     {
        //         r+=float(colbr[2])*wbr;
        //         g+=float(colbr[1])*wbr;
        //         b+=float(colbr[0])*wbr;
        //         w+=wbr;
        //     }
        // }

    //Visibility for the whole 4-pixels group
    if(safetl)
    {
        for(int delta_cfx=0;delta_cfx<CLEANING_FACTOR;++delta_cfx)
        {
            for(int delta_cfy=0;delta_cfy<CLEANING_FACTOR;++delta_cfy)
            {

                if (triangle_idx == cam_image.at<int32_t>(intX*CLEANING_FACTOR+delta_cfx,intY*CLEANING_FACTOR+delta_cfy))
                {
                    testPix = true;
                }
            }
        }
    }
    if(safetr)
    {
        for(int delta_cfx=0;delta_cfx<CLEANING_FACTOR;++delta_cfx)
        {
            for(int delta_cfy=0;delta_cfy<CLEANING_FACTOR;++delta_cfy)
            {
                if (triangle_idx == cam_image.at<int32_t>((intX+1)*CLEANING_FACTOR+delta_cfx,intY*CLEANING_FACTOR+delta_cfy))
                {
                    testPix = true;
                }
            }
        }
    }
    if(safebl)
    {
        for(int delta_cfx=0;delta_cfx<CLEANING_FACTOR;++delta_cfx)
        {
            for(int delta_cfy=0;delta_cfy<CLEANING_FACTOR;++delta_cfy)
            {
                if (triangle_idx == cam_image.at<int32_t>(intX*CLEANING_FACTOR+delta_cfx,(intY+1)*CLEANING_FACTOR+delta_cfy))
                {

                    testPix = true;
                }
            }
        }
    }
    if(safebr)
    {
        for(int delta_cfx=0;delta_cfx<CLEANING_FACTOR;++delta_cfx)
        {
            for(int delta_cfy=0;delta_cfy<CLEANING_FACTOR;++delta_cfy)
            {
                if (triangle_idx == cam_image.at<int32_t>((intX+1)*CLEANING_FACTOR+delta_cfx,(intY+1)*CLEANING_FACTOR+delta_cfy))
                {
                    testPix = true;
                }
            }
        }
    }
    if (testPix)
    {
        if(safetl)
        {
            r+=float(coltl[2])*wtl;
            g+=float(coltl[1])*wtl;
            b+=float(coltl[0])*wtl;
            w+=wtl;
        }
        if(safetr)
        {

            r+=float(coltr[2])*wtr;
            g+=float(coltr[1])*wtr;
            b+=float(coltr[0])*wtr;
            w+=wtr;
        }
        if(safebl)
        {
            r+=float(colbl[2])*wbl;
            g+=float(colbl[1])*wbl;
            b+=float(colbl[0])*wbl;
            w+=wbl;
        }
        if(safebr)
        {
            r+=float(colbr[2])*wbr;
            g+=float(colbr[1])*wbr;
            b+=float(colbr[0])*wbr;
            w+=wbr;
        }
    }

    r *= pow(1.0f/CLEANING_FACTOR,2);
    g *= pow(1.0f/CLEANING_FACTOR,2);
    b *= pow(1.0f/CLEANING_FACTOR,2);
    w *= pow(1.0f/CLEANING_FACTOR,2);

    if (w<=0.001f)
    {
        return false;
    }

    r/=w;
    b/=w;
    g/=w;

    out_color = Vector3ui((unsigned int)(r),(unsigned int)(g),(unsigned int)(b));

    return true;

}

// Takes point p of the surface, given by (myTri, baryCoords)
// Gets its 2D coordinates in image of cameraNumber
// Returns its color, using Nearest Neighbour interpolation
// SHOULD NOT BE USED!!
// Corrected bug in getSurfacePointColor. Most likely also present in this function
template<class InPoint, class InTriangle>
bool SpaceTimeSampler::getSurfacePointColorNN(InTriangle &myTri, const std::vector<InPoint> &in_points, Vector3f baryCoords, int cameraNumber, Vector3ui &out_color, bool writeLog)const
{
    
    const Camera &temp_cam = v_cameras_[cameraNumber];
    Vector3f cam_pos = temp_cam.getPosition();
    cv::Vec3b col,coltl,coltr,colbl,colbr;//BGR order

    bool is_safe = true;
    
    Vector2f ref_coords,edge1_coords,edge2_coords, tex_coords;
    temp_cam.getTextureCoords(in_points[myTri.ref],ref_coords);
    temp_cam.getTextureCoords(in_points[myTri.edge1],edge1_coords);
    temp_cam.getTextureCoords(in_points[myTri.edge2],edge2_coords);


    tex_coords(0) = ref_coords(0) * baryCoords(0) + edge1_coords(0) * baryCoords(1) + edge2_coords(0) * baryCoords(2);
    tex_coords(1) = ref_coords(1) * baryCoords(0) + edge1_coords(1) * baryCoords(1) + edge2_coords(1) * baryCoords(2);
    

    // int intX = int(floor(tex_coords(0)*2048-0.5f)+0.1f);
    // int intY = int(floor(tex_coords(0)*2048-0.5f)+0.1f);
    // float weightX = tex_coords(0)*2048-0.5f-intX;
    // float weightY = tex_coords(0)*2048-0.5f-intY;

    float intX = floor(tex_coords(0));
    float intY = floor(tex_coords(1));
    float weightX = tex_coords(0)-intX;
    float weightY = tex_coords(1)-intY;
    Vector2uli tl_coords, tr_coords, bl_coords, br_coords;
    tl_coords = Vector2uli(intX,intY);
    tr_coords = Vector2uli(intX+1,intY);
    bl_coords = Vector2uli(intX,intY+1);
    br_coords = Vector2uli(intX+1,intY+1);
    bool safetl = temp_cam.getPixelColor(tl_coords,coltl);
    bool safetr = temp_cam.getPixelColor(tr_coords,coltr);
    bool safebl = temp_cam.getPixelColor(bl_coords,colbl);
    bool safebr = temp_cam.getPixelColor(br_coords,colbr);

    if((!safetl)&&(!safetr)&&(!safebl)&&(!safebr))
    {
        return false;
    }

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    // Not dealing with the case where closest pixel does not exist...
    // Only preventing error (returns black pixel)

    if(weightX > 1-weightX)         //choose pixel on the right
    {
        if(weightY > 1-weightY)     //choose pixel on the bottom
        {
            if(safebr)
            {
                r=float(colbr[2]);
                g=float(colbr[1]);
                b=float(colbr[0]);
            }
        }
        else                        //choose pixel on the top
        {
            if(safetr)
            {
                r=float(coltr[2]);
                g=float(coltr[1]);
                b=float(coltr[0]);
            }
        }
    }
    else
    {
        if(weightY > 1-weightY)     //choose pixel on the bottom
        {
            if(safebl)
            {
                r=float(colbl[2]);
                g=float(colbl[1]);
                b=float(colbl[0]);
            }
        }
        else                        //choose pixel on the top
        {
            if(safetl)
            {
                r=float(coltl[2]);
                g=float(coltl[1]);
                b=float(coltl[0]);
            }
        }
    }

    out_color = Vector3ui((unsigned int)(r),(unsigned int)(g),(unsigned int)(b));

    return true;

}



template<class InColor, class InTriangle>
void SpaceTimeSampler::downsampleTriangleChroma( unsigned long tri,
                                                const InTriangle &myTri,
                                                int triRes,
                                                const std::vector<unsigned long>&in_face_color_ind,
                                                const std::vector<Vector3li> &in_edge_color_ind,
                                                std::vector<InColor> &in_colors
                                                )const
{
    unsigned long myTri_color_ind = in_face_color_ind[tri];
    //loop over color samples
    for(int i=1;i<triRes-1;++i)
    {
        for(int j=1;j<triRes-i;++j)
        {
            if((i%2==1)||(j%2==1))      //sample is not present in the lower level of resolution if at least one of its barycentric index is odd
            {
                //compare sample value with value obtained by interpolation
                //each sample is in the middle of a segment between two lower-level samples.
                int indexOffset = (i-1)*triRes - i*(i+1)/2 + j;
                InColor &sampColor = in_colors[myTri_color_ind+indexOffset];
                InColor n1, n2;     //neighbours

                //get neighbouring samples
                if(i%2==0)  //neighbours are (i,j-1) and (i,j+1)
                {
                    //n1
                    if(j-1==0)  //edge sample
                    {
                        n1 = getSampleColor(myTri, tri, triRes, i, j-1, in_edge_color_ind, in_face_color_ind, in_colors);
                    }
                    else    //face sample
                    {
                        n1 = in_colors[myTri_color_ind+indexOffset-1];
                    }
                    //n2
                    if(j+1+i==triRes)   //edge sample
                    {
                        n2 = getSampleColor(myTri, tri, triRes, i, j+1, in_edge_color_ind, in_face_color_ind, in_colors);
                    }
                    else    //face sample
                    {
                        n2 = in_colors[myTri_color_ind+indexOffset+1];
                    }
                }
                else if(j%2==0) //neighbours are (i-1,j) and (i+1,j)
                {
                    //n1
                    if(i-1==0)  //edge sample
                    {
                        n1 = getSampleColor(myTri, tri, triRes, i-1, j, in_edge_color_ind, in_face_color_ind, in_colors);
                    }
                    else    //face sample
                    {
                        int n1Offset = (i-2)*triRes - i*(i-1)/2 + j;
                        n1 = in_colors[myTri_color_ind+n1Offset];
                    }
                    //n2
                    if(i+1+j==triRes)   //edge sample
                    {
                        n2 = getSampleColor(myTri, tri, triRes, i+1, j, in_edge_color_ind, in_face_color_ind, in_colors);
                    }
                    else    //face sample
                    {
                        int n2Offset = i*triRes - (i+2)*(i+1)/2 + j;
                        n2 = in_colors[myTri_color_ind+n2Offset];
                    }
                }
                else    //j and i both odd. neighbours are (i-1,j+1) and (i+1,j-1)
                {
                    //n1
                    if(i-1==0)      //edge samples
                    {
                        n1 = getSampleColor(myTri, tri, triRes, i-1, j+1, in_edge_color_ind, in_face_color_ind, in_colors);
                    }
                    else    //face sample
                    {
                        int n1Offset = (i-2)*triRes - i*(i-1)/2 + j+1;
                        n1 = in_colors[myTri_color_ind+n1Offset];
                    }
                    //n2
                    if(j-1==0)      //edge sample
                    {
                        n2 = getSampleColor(myTri, tri, triRes, i+1, j-1, in_edge_color_ind, in_face_color_ind, in_colors);
                    }
                    else    //face sample
                    {
                        int n2Offset = i*triRes - (i+2)*(i+1)/2 + j-1;
                        n2 = in_colors[myTri_color_ind+n2Offset];
                    }
                }
                rgbToYcc(n1);
                rgbToYcc(n2);
                rgbToYcc(sampColor);
                sampColor(1) = int((float(n1(1))+float(n2(1)))/2.0);
                sampColor(2) = int((float(n1(2))+float(n2(2)))/2.0);
                // sampColor(1) = 128;
                // sampColor(2) = 128;
                yccToRgb(sampColor);
            }
        }
    }
}




template<class InColor>
void SpaceTimeSampler::downsampleEdge(  long edgeInd,
                                        long v1Ind,         //v1 is the vertex close to the first edge sample (in writing order)
                                        long v2Ind,
                                        std::vector<InColor> &in_colors,
                                        float maxIPThreshold
                                        )const
{

    if(edgeInd<0)
    {
        long temp=v1Ind;
        v1Ind =v2Ind;
        v2Ind=temp;
        edgeInd=-edgeInd;
    }
    //get edge resolution
    int myRes = in_colors[edgeInd](0);

    while(myRes>1)
    {
        for(int i=1;i<myRes;++i)
        {
            if(i%2==1)  //sample is lost in lower resolution
            {
                InColor n1,n2;
                InColor sampColor = in_colors[edgeInd+i];
                if(i==1)
                {
                    n1 = in_colors[v1Ind];
                }
                else
                {
                    n1 = in_colors[edgeInd+i-1];
                }
                if(i==myRes-1)
                {
                    n2 = in_colors[v2Ind];
                }
                else
                {
                    n2 = in_colors[edgeInd+i+1];
                }

                //interpolate and compute distance
                float sampDist = pow(0.5*float(n1(0)+n2(0))-float(sampColor(0)),2) + pow(0.5*float(n1(1)+n2(1))-float(sampColor(1)),2) + pow(0.5*float(n1(2)+n2(2))-float(sampColor(2)),2);
                //compare to threshold
                if(sampDist>maxIPThreshold)
                {
                    //Stop here, keep current resolution
                    return;
                }
            }
        }
        //If we reach this point, all samples passed the test, we can downsample the edge
        myRes/=2;
        in_colors[edgeInd](0) = myRes;
        for(int i=1;i<myRes;++i)
        {
            in_colors[edgeInd+i] = in_colors[edgeInd+2*i];
        }
    }
    return;
}

template<class InColor>
void SpaceTimeSampler::downsampleEdgeMean(  long edgeInd,
                                        long v1Ind,         //v1 is the vertex close to the first edge sample (in writing order)
                                        long v2Ind,
                                        std::vector<InColor> &in_colors,
                                        float maxIPThreshold
                                        )const
{
    if(edgeInd<0)
    {
        long temp=v1Ind;
        v1Ind =v2Ind;
        v2Ind=temp;
        edgeInd=-edgeInd;
    }
    //get edge resolution
    int myRes = in_colors[edgeInd](0);

    while(myRes>1)
    {
        float sampDist=0.0f;
        for(int i=1;i<myRes;++i)
        {
            if(i%2==1)  //sample is lost in lower resolution
            {
                InColor n1,n2;
                InColor sampColor = in_colors[edgeInd+i];
                if(i==1)
                {
                    n1 = in_colors[v1Ind];
                }
                else
                {
                    n1 = in_colors[edgeInd+i-1];
                }
                if(i==myRes-1)
                {
                    n2 = in_colors[v2Ind];
                }
                else
                {
                    n2 = in_colors[edgeInd+i+1];
                }

                //interpolate and compute distance
                sampDist += pow(0.5*float(n1(0)+n2(0))-float(sampColor(0)),2) + pow(0.5*float(n1(1)+n2(1))-float(sampColor(1)),2) + pow(0.5*float(n1(2)+n2(2))-float(sampColor(2)),2);

            }
        }
        if(sampDist>maxIPThreshold*(myRes-1))   //compare to threshold
        {
            return; //Stop here, keep current resolution
        }
        //If we reach this point, all samples passed the test, we can downsample the edge
        myRes/=2;
        in_colors[edgeInd](0) = myRes;
        for(int i=1;i<myRes;++i)
        {
            in_colors[edgeInd+i] = in_colors[edgeInd+2*i];
        }
    }
    return;
}

template<class InColor>
void SpaceTimeSampler::downsampleEdgeChroma(long edgeInd,
                                            long v1Ind,         //v1 is the vertex close to the first edge sample (in writing order)
                                            long v2Ind,
                                            std::vector<InColor> &in_colors,
                                            bool writeLog
                                            )const
{
    if(edgeInd<0)
    {
        long temp=v1Ind;
        v1Ind =v2Ind;
        v2Ind=temp;
        edgeInd=-edgeInd;
    }
    //get edge resolution
    int myRes = in_colors[edgeInd](0);
    for(int i=1;i<myRes;++i)
    {
        if(writeLog)
        {
            log(ALWAYS)<<"Log edge: res = "<<myRes<<endLog();
        }
        if(i%2==1)  //sample is lost in lower resolution
        {
            InColor n1,n2;
            InColor &sampColor = in_colors[edgeInd+i];

            if(edgeInd+i==13545186)
            {
                log(WARN)<<"WHOOOA, res = "<<myRes<<", i = "<<i<<endLog();
                writeLog=true;
            }
            if(i==1)
            {
                n1 = in_colors[v1Ind];
            }
            else
            {
                n1 = in_colors[edgeInd+i-1];
            }
            if(i==myRes-1)
            {
                n2 = in_colors[v2Ind];
            }
            else
            {
                n2 = in_colors[edgeInd+i+1];
            }
            if(writeLog)
            {
                log(ALWAYS)<<"i = "<<i<<endLog();
                log(ALWAYS)<<"n1 = ("<<n1(0)<<","<<n1(1)<<","<<n1(2)<<")"<<endLog();
                log(ALWAYS)<<"n2 = ("<<n2(0)<<","<<n2(1)<<","<<n2(2)<<")"<<endLog();
                log(ALWAYS)<<"sampColor = ("<<sampColor(0)<<","<<sampColor(1)<<","<<sampColor(2)<<")"<<endLog();
            }
            rgbToYcc(n1);
            rgbToYcc(n2);
            rgbToYcc(sampColor);
            if(writeLog)
            {
                log(ALWAYS)<<"YCC transform:"<<endLog();
                log(ALWAYS)<<"n1 = ("<<n1(0)<<","<<n1(1)<<","<<n1(2)<<")"<<endLog();
                log(ALWAYS)<<"n2 = ("<<n2(0)<<","<<n2(1)<<","<<n2(2)<<")"<<endLog();
                log(ALWAYS)<<"sampColor = ("<<sampColor(0)<<","<<sampColor(1)<<","<<sampColor(2)<<")"<<endLog();
            }
            sampColor(1) = int((float(n1(1))+float(n2(1)))/2.0);
            sampColor(2) = int((float(n1(2))+float(n2(2)))/2.0);
            if(writeLog)
            {
                log(ALWAYS)<<"interpolated sampColor = ("<<sampColor(0)<<","<<sampColor(1)<<","<<sampColor(2)<<")"<<endLog();
            }
            // sampColor(1) = 128;
            // sampColor(2) = 128;
            yccToRgb(sampColor);
            if(writeLog)
            {
                log(ALWAYS)<<"Final RGB = ("<<sampColor(0)<<","<<sampColor(1)<<","<<sampColor(2)<<")"<<endLog();
            }

        }
    }
}

//Returns a result in [0,255]³
template<class InColor>
void SpaceTimeSampler::rgbToYcc(InColor &myColor)const
{
    unsigned int Y = (unsigned int)(std::min(255.0,std::max(0.0,(float(myColor(0))*0.299+float(myColor(1))*0.587+float(myColor(2))*0.114)))+0.5);              //Y
    unsigned int Cb = (unsigned int)(std::min(255.0,std::max(0.0,(-float(myColor(0))*0.168736-float(myColor(1))*0.331264+float(myColor(2))*0.5+128)))+0.5);     //Cb
    unsigned int Cr = (unsigned int)(std::min(255.0,std::max(0.0,(float(myColor(0))*0.5-float(myColor(1))*0.418688-float(myColor(2))*0.081312+128)))+0.5);      //Cr
    myColor = InColor(Y,Cb,Cr);
}


template<class InColor>
void SpaceTimeSampler::yccToRgb(InColor &myColor)const
{
    unsigned int R = (unsigned int)(std::min(255.0,std::max(0.0,(float(myColor(0)) + 1.402 * (float(myColor(2)) - 128.0))))+0.5);
    unsigned int G = (unsigned int)(std::min(255.0,std::max(0.0,(float(myColor(0)) - 0.344136 * (float(myColor(1)) - 128.0) - 0.714136 * (float(myColor(2)) - 128.0))))+0.5);
    unsigned int B = (unsigned int)(std::min(255.0,std::max(0.0,(float(myColor(0)) + 1.772 * (float(myColor(1)) - 128.0))))+0.5);
    myColor = InColor(R,G,B);
}


template<class InColor>
void SpaceTimeSampler::consistencyTest( std::vector<InColor> &in_colors,
                                        std::vector<unsigned short> &in_face_res,
                                        std::vector<Vector3li> &in_edge_color_ind,
                                        std::vector<unsigned long> &in_face_color_ind
                                    )const
{
    //Checking for errors
    //checking if edge references a face sample
    for(unsigned long gertrude=0; gertrude<in_face_color_ind.size();++gertrude)
    {
        if(in_face_color_ind[gertrude]!=0)
        {
            int myFR = in_face_res[gertrude];
            int faceSamples = (myFR-2)*(myFR-1)/2;
            for(unsigned long gerard=0; gerard<in_edge_color_ind.size();++gerard)
            {
            
                if ((in_face_color_ind[gertrude]<=in_edge_color_ind[gerard](0))&&in_face_color_ind[gertrude]+faceSamples>in_edge_color_ind[gerard](0))
                {
                    log(ALWAYS)<<"edge indexing error detected: edge "<<gerard<<", face "<<gertrude<<", color ind "<<in_face_color_ind[gertrude]<<endLog();
                }
                if ((in_face_color_ind[gertrude]<=in_edge_color_ind[gerard](1))&&in_face_color_ind[gertrude]+faceSamples>in_edge_color_ind[gerard](1))
                {
                    log(ALWAYS)<<"edge indexing error detected: edge "<<gerard<<", face "<<gertrude<<", color ind "<<in_face_color_ind[gertrude]<<endLog();
                }
                if ((in_face_color_ind[gertrude]<=in_edge_color_ind[gerard](2))&&in_face_color_ind[gertrude]+faceSamples>in_edge_color_ind[gerard](2))
                {
                    log(ALWAYS)<<"edge indexing error detected: edge "<<gerard<<", face "<<gertrude<<", color ind "<<in_face_color_ind[gertrude]<<endLog();
                }

            }
        }
    }
    
    //checking if face samples look like "edge resolution" pixels
    for(unsigned long gertrude=0; gertrude<in_face_color_ind.size();++gertrude)
    {
        int myFR = in_face_res[gertrude];
        int faceSamples = (myFR-2)*(myFR-1)/2;
        for(int georges=0;georges<faceSamples;++georges)
        {
            if(in_colors[in_face_color_ind[gertrude]+georges](2)==255 && in_colors[in_face_color_ind[gertrude]+georges](1)==53)
            {
                log(ALWAYS)<<"edge resolution pixel detected in face: "<<gertrude<<", color ind "<<in_face_color_ind[gertrude]<<", sample "<<georges<<endLog();
            }
        }
    }

    //checking if edge samples look like "edge resolution" pixels
    for(unsigned long gerard=0; gerard<in_edge_color_ind.size();++gerard)  //for each triangle
    {
        for(int k=0;k<3;++k)   //for each edge
        {
            unsigned long edgeI = std::abs(in_edge_color_ind[gerard](k));
            int edgeRes = in_colors[edgeI](0);
            for(int bernard = 1;bernard<edgeRes;++bernard)  //for each sample
            {
                if(in_colors[edgeI+bernard](2)==255 && in_colors[edgeI+bernard](1)==255 && in_colors[edgeI+bernard](0)<=32)
                {
                    log(ALWAYS)<<"edge resolution pixel detected in edge: "<<gerard<<", color ind "<<in_edge_color_ind[gerard](k)<<", sample "<<bernard<<endLog();
                }
            }
        }
    }
}





#endif // SPACE_TIME_SAMPLER
