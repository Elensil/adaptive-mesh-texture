#include "mesh.h"
#include "math.h"
#include <boost/filesystem.hpp>

MySpecialMesh::MySpecialMesh(long int frame, std::string filename):active_frame_(frame),s_file_name_(filename){
    if(!filename.empty())
    {
        std::string format = boost::filesystem::extension(filename);
        format = format.substr(1);
        if(!format.compare("obj"))
        {
            if(!loadOBJ(filename))
            {
                log(ERROR)<<"[Mesh OBJ loader] Error: Could not load OBJ file."<<endLog();
            }
        }
        else if(!format.compare("off"))
        {
            if(!loadMOFF(filename))
            {
                log(ERROR)<<"[Mesh OFF loader] Error: Could not load OFF file."<<endLog();
            }
        }
        else
            log(ERROR)<<"[MySpecialMesh()] Error, unknown input file format."<<endLog();
    }
}

/**
 * @brief MySpecialMesh::MySpecialMesh
 * @param in : copy constructor
 */
MySpecialMesh::MySpecialMesh(const MySpecialMesh &in){

    active_frame_ = in.getActiveFrame();

    //Mesh components
    s_file_name_ = in.getFileName();
    in.getPointsVector(v_points_);
    in.getFacesVector(v_faces_);
}

MySpecialMesh::~MySpecialMesh(){
}

bool MySpecialMesh::loadOBJ(const std::string objFile, bool clear){
    std::cout<<"Reading file "<<objFile<<std::endl;
    if(clear)
    {
        v_points_.clear();
        v_faces_.clear();
        v_points_separator_.clear();
        v_faces_separator_.clear();
    }
    v_points_separator_.push_back(v_points_.size());//Index of the first vertex of this mesh
    v_faces_separator_.push_back(v_faces_.size());//Index of the first face of this mesh
    std::vector<int32_t> old_indices;
    int32_t count = 0;

    std::vector<int32_t> nan_points;

    bool isfile_readable = false;

    int matches = 0;

    //Face managment variables (only working for triangles) TBD: improve this
    int32_t vertexIndex[4];

    FILE * file = fopen(objFile.c_str(), "r");
    if( file == NULL ){
        printf("Could not open the .obj file !\n");
        return false;
    }
    while( 1 ){
        char lineHeader[128];//we assume that the first word of a line won’t be longer than 128, which is a very silly assumption. But for a toy parser, it’s all right
        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // reached EOF.  Quit the loop.
        // else : parse lineHeader
        ///Comment
        if ( strcmp( lineHeader, "#" ) == 0 ){
            char word[128];

            int dummy = fscanf(file, "%s\n", word);
            if(dummy == EOF)
                break;
        }
        ///Vertex
        if ( strcmp( lineHeader, "v" ) == 0 ){
            isfile_readable = true;
            Vector3f vertex;
            matches = fscanf(file, "%f %f %f\n", &vertex[0], &vertex[1], &vertex[2] );
            if (matches == 3)
            {
                if(vertex(0) != vertex(0) || vertex(1) != vertex(1) || vertex(2) != vertex(2))
                {
                    nan_points.push_back(v_points_.size());
                    old_indices.push_back(-1);
                }
                else
                {
                    v_points_.push_back(vertex);
                    old_indices.push_back(count++);
                }
            }else
                isfile_readable = false;
        //Texture coordinates
        }else if (strcmp(lineHeader, "vt" ) == 0 ){
            Vector2f texcoord;
            matches = fscanf(file, "%f %f\n", &texcoord[0], &texcoord[1]);
            if (matches==2)
            {
                if(texcoord(0) != texcoord(0) || texcoord(1) != texcoord(1))
                {
                    std::cout<<"Erro with texture coordinates"<<std::endl;
                    return false;
                }
                tex_coords_.push_back(texcoord);
            }
        }
        ///Face
        else if ( strcmp( lineHeader, "f" ) == 0 ){
        ///Face
            Vector3uli texCoordIndex(0,0,0);
            unsigned long texCoordInd4=0;
            matches = fscanf(file, " %d/%d %d/%d %d/%d %d/%d\n", &vertexIndex[0], &texCoordIndex[0], &vertexIndex[1], &texCoordIndex[1], &vertexIndex[2], &texCoordIndex[2], &vertexIndex[3], &texCoordInd4);
            if(matches==8){
                if (    (unsigned int)(vertexIndex[0] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                        (unsigned int)(vertexIndex[1] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                        (unsigned int)(vertexIndex[2] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                        (unsigned int)(vertexIndex[3] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ){
                    std::cout<<"Error! Vertex index out of range: nb_points "<<v_points_.size()<<std::endl;

                    std::cout<<"Indices: "<<vertexIndex[0]-1 <<" "<< vertexIndex[1]-1 <<" "<<vertexIndex[2]-1<<std::endl;
                    return false;
                }
                else{ //TBD: add non triangular faces management
                    bool is_triangle_safe = true;
                    const unsigned long long int temp_index_ref = vertexIndex[0] - 1 + v_points_separator_[v_points_separator_.size()-1]; //From 1 to 0 starting point
                    const unsigned long long int temp_index_edge1 = vertexIndex[1] - 1 + v_points_separator_[v_points_separator_.size()-1];
                    const unsigned long long int temp_index_edge2 = vertexIndex[2] - 1 + v_points_separator_[v_points_separator_.size()-1];
                    const unsigned long long int temp_index_edge3 = vertexIndex[3] - 1 + v_points_separator_[v_points_separator_.size()-1];
                    for(int32_t j = 0; j < nan_points.size(); ++j)
                    {
                        if(nan_points[j] == temp_index_ref || nan_points[j] == temp_index_edge1  || nan_points[j] == temp_index_edge2 || nan_points[j] == temp_index_edge3)
                        {
                            is_triangle_safe = false;
                            break;
                        }
                    }
                    if(is_triangle_safe)
                    {
                        MyTriangle triangle;
                        triangle.ref = old_indices[vertexIndex[0] - 1]   + v_points_separator_[v_points_separator_.size()-1];
                        triangle.edge1 = old_indices[vertexIndex[1] - 1] + v_points_separator_[v_points_separator_.size()-1];
                        triangle.edge2 = old_indices[vertexIndex[2] - 1] + v_points_separator_[v_points_separator_.size()-1];
                        v_faces_.push_back(triangle);
                        MyTriangle triangle2;
                        triangle2.ref = old_indices[vertexIndex[2] - 1]   + v_points_separator_[v_points_separator_.size()-1];
                        triangle2.edge1 = old_indices[vertexIndex[3] - 1] + v_points_separator_[v_points_separator_.size()-1];
                        triangle2.edge2 = old_indices[vertexIndex[0] - 1] + v_points_separator_[v_points_separator_.size()-1];
                        v_faces_.push_back(triangle2);
                    }
                }
                texCoordIndex[0]--;
                texCoordIndex[1]--;
                texCoordIndex[2]--;
                texCoordInd4--;
                Vector3uli texCoordIndex2(0,0,0);
                texCoordIndex2[0]=texCoordIndex[2];
                texCoordIndex2[1]=texCoordInd4;
                texCoordIndex2[2]=texCoordIndex[0];
                tex_indices_.push_back(texCoordIndex);
                tex_indices_.push_back(texCoordIndex2);
            }


            else if (matches == 6){
                

                //first part copied from below, with small addition for face texture indices
                if (    (unsigned int)(vertexIndex[0] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                        (unsigned int)(vertexIndex[1] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                        (unsigned int)(vertexIndex[2] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ){
                    std::cout<<"Error! Vertex index out of range: nb_points "<<v_points_.size()<<std::endl;

                    std::cout<<"Indices: "<<vertexIndex[0]-1 <<" "<< vertexIndex[1]-1 <<" "<<vertexIndex[2]-1<<std::endl;
                    return false;
                }
                else{ //TBD: add non triangular faces management
                    bool is_triangle_safe = true;
                    const unsigned long long int temp_index_ref = vertexIndex[0] - 1 + v_points_separator_[v_points_separator_.size()-1]; //From 1 to 0 starting point
                    const unsigned long long int temp_index_edge1 = vertexIndex[1] - 1 + v_points_separator_[v_points_separator_.size()-1];
                    const unsigned long long int temp_index_edge2 = vertexIndex[2] - 1 + v_points_separator_[v_points_separator_.size()-1];
                    for(int32_t j = 0; j < nan_points.size(); ++j)
                    {
                        if(nan_points[j] == temp_index_ref || nan_points[j] == temp_index_edge1  || nan_points[j] == temp_index_edge2  )
                        {
                            is_triangle_safe = false;
                            break;
                        }
                    }
                    if(is_triangle_safe)
                    {
                        MyTriangle triangle;
                        triangle.ref = old_indices[vertexIndex[0] - 1]   + v_points_separator_[v_points_separator_.size()-1];
                        triangle.edge1 = old_indices[vertexIndex[1] - 1] + v_points_separator_[v_points_separator_.size()-1];
                        triangle.edge2 = old_indices[vertexIndex[2] - 1] + v_points_separator_[v_points_separator_.size()-1];
                        v_faces_.push_back(triangle);
                    }
                }
                texCoordIndex[0]--;
                texCoordIndex[1]--;
                texCoordIndex[2]--;
                tex_indices_.push_back(texCoordIndex);
            }
            else if (matches==1)    //we suppose the 1st test failed because there are no texture and normal coordinates. vertexIndex[0] should be read already, just get the other two.
            {
                matches = fscanf(file, " %d %d\n", &vertexIndex[1], &vertexIndex[2] );
                if (matches == 2){
                    if (    (unsigned int)(vertexIndex[0] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                            (unsigned int)(vertexIndex[1] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                            (unsigned int)(vertexIndex[2] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ){
                        std::cout<<"Error! Vertex index out of range: nb_points "<<v_points_.size()<<std::endl;

                        std::cout<<"Indices: "<<vertexIndex[0]-1 <<" "<< vertexIndex[1]-1 <<" "<<vertexIndex[2]-1<<std::endl;
                        return false;
                    }
                    else{ //TBD: add non triangular faces management
                        bool is_triangle_safe = true;
                        const unsigned long long int temp_index_ref = vertexIndex[0] - 1 + v_points_separator_[v_points_separator_.size()-1]; //From 1 to 0 starting point
                        const unsigned long long int temp_index_edge1 = vertexIndex[1] - 1 + v_points_separator_[v_points_separator_.size()-1];
                        const unsigned long long int temp_index_edge2 = vertexIndex[2] - 1 + v_points_separator_[v_points_separator_.size()-1];
                        for(int32_t j = 0; j < nan_points.size(); ++j)
                        {
                            if(nan_points[j] == temp_index_ref || nan_points[j] == temp_index_edge1  || nan_points[j] == temp_index_edge2  )
                            {
                                is_triangle_safe = false;
                                break;
                            }
                        }
                        if(is_triangle_safe)
                        {
                            MyTriangle triangle;
                            triangle.ref = old_indices[vertexIndex[0] - 1]   + v_points_separator_[v_points_separator_.size()-1];
                            triangle.edge1 = old_indices[vertexIndex[1] - 1] + v_points_separator_[v_points_separator_.size()-1];
                            triangle.edge2 = old_indices[vertexIndex[2] - 1] + v_points_separator_[v_points_separator_.size()-1];
                            v_faces_.push_back(triangle);
                        }
                    }
                }
                else{
                    std::cout<<"File can't be read by this simple parser : ( Try exporting with other options, or check if faces are triangles... others not managed yet)"<<std::endl;
                    return false;
                }
            }
            else if (matches==2)    //we supose the 1st failed because there are vertex normal indices (v/vt/vn). New scanning, then, if parts copied from above.
            {
                int *ptr = NULL;
                unsigned long garbage1, garbage2, garbage3, garbage4;
                matches = fscanf(file, "/%d %d/%d/%d %d/%d/%d %d/%d/%d\n", &garbage1, &vertexIndex[1], &texCoordIndex[1], &garbage2, &vertexIndex[2], &texCoordIndex[2], &garbage3, &vertexIndex[3], &texCoordInd4, &garbage4);
                // log(ALWAYS)<<"v/vt/vn case: matches = "<<matches<<endLog();
                if (matches==10)
                {
                    if (    (unsigned int)(vertexIndex[0] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                        (unsigned int)(vertexIndex[1] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                        (unsigned int)(vertexIndex[2] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                        (unsigned int)(vertexIndex[3] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ){
                        std::cout<<"Error! Vertex index out of range: nb_points "<<v_points_.size()<<std::endl;

                        std::cout<<"Indices: "<<vertexIndex[0]-1 <<" "<< vertexIndex[1]-1 <<" "<<vertexIndex[2]-1<<std::endl;
                        return false;
                    }
                    else{ //TBD: add non triangular faces management
                        bool is_triangle_safe = true;
                        const unsigned long long int temp_index_ref = vertexIndex[0] - 1 + v_points_separator_[v_points_separator_.size()-1]; //From 1 to 0 starting point
                        const unsigned long long int temp_index_edge1 = vertexIndex[1] - 1 + v_points_separator_[v_points_separator_.size()-1];
                        const unsigned long long int temp_index_edge2 = vertexIndex[2] - 1 + v_points_separator_[v_points_separator_.size()-1];
                        const unsigned long long int temp_index_edge3 = vertexIndex[3] - 1 + v_points_separator_[v_points_separator_.size()-1];
                        for(int32_t j = 0; j < nan_points.size(); ++j)
                        {
                            if(nan_points[j] == temp_index_ref || nan_points[j] == temp_index_edge1  || nan_points[j] == temp_index_edge2 || nan_points[j] == temp_index_edge3)
                            {
                                is_triangle_safe = false;
                                break;
                            }
                        }
                        if(is_triangle_safe)
                        {
                            MyTriangle triangle;
                            triangle.ref = old_indices[vertexIndex[0] - 1]   + v_points_separator_[v_points_separator_.size()-1];
                            triangle.edge1 = old_indices[vertexIndex[1] - 1] + v_points_separator_[v_points_separator_.size()-1];
                            triangle.edge2 = old_indices[vertexIndex[2] - 1] + v_points_separator_[v_points_separator_.size()-1];
                            v_faces_.push_back(triangle);
                            MyTriangle triangle2;
                            triangle2.ref = old_indices[vertexIndex[2] - 1]   + v_points_separator_[v_points_separator_.size()-1];
                            triangle2.edge1 = old_indices[vertexIndex[3] - 1] + v_points_separator_[v_points_separator_.size()-1];
                            triangle2.edge2 = old_indices[vertexIndex[0] - 1] + v_points_separator_[v_points_separator_.size()-1];
                            v_faces_.push_back(triangle2);
                        }
                    }
                    texCoordIndex[0]--;
                    texCoordIndex[1]--;
                    texCoordIndex[2]--;
                    texCoordInd4--;
                    Vector3uli texCoordIndex2(0,0,0);
                    texCoordIndex2[0]=texCoordIndex[2];
                    texCoordIndex2[1]=texCoordInd4;
                    texCoordIndex2[2]=texCoordIndex[0];
                    tex_indices_.push_back(texCoordIndex);
                    tex_indices_.push_back(texCoordIndex2);
                }
                else if (matches==7)
                {
                    //first part copied from below, with small addition for face texture indices
                    if (    (unsigned int)(vertexIndex[0] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                            (unsigned int)(vertexIndex[1] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                            (unsigned int)(vertexIndex[2] - 1 + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ){
                        std::cout<<"Error! Vertex index out of range: nb_points "<<v_points_.size()<<std::endl;

                        std::cout<<"Indices: "<<vertexIndex[0]-1 <<" "<< vertexIndex[1]-1 <<" "<<vertexIndex[2]-1<<std::endl;
                        return false;
                    }
                    else{ //TBD: add non triangular faces management
                        bool is_triangle_safe = true;
                        const unsigned long long int temp_index_ref = vertexIndex[0] - 1 + v_points_separator_[v_points_separator_.size()-1]; //From 1 to 0 starting point
                        const unsigned long long int temp_index_edge1 = vertexIndex[1] - 1 + v_points_separator_[v_points_separator_.size()-1];
                        const unsigned long long int temp_index_edge2 = vertexIndex[2] - 1 + v_points_separator_[v_points_separator_.size()-1];
                        for(int32_t j = 0; j < nan_points.size(); ++j)
                        {
                            if(nan_points[j] == temp_index_ref || nan_points[j] == temp_index_edge1  || nan_points[j] == temp_index_edge2  )
                            {
                                is_triangle_safe = false;
                                break;
                            }
                        }
                        if(is_triangle_safe)
                        {
                            MyTriangle triangle;
                            triangle.ref = old_indices[vertexIndex[0] - 1]   + v_points_separator_[v_points_separator_.size()-1];
                            triangle.edge1 = old_indices[vertexIndex[1] - 1] + v_points_separator_[v_points_separator_.size()-1];
                            triangle.edge2 = old_indices[vertexIndex[2] - 1] + v_points_separator_[v_points_separator_.size()-1];
                            v_faces_.push_back(triangle);
                        }
                    }
                    texCoordIndex[0]--;
                    texCoordIndex[1]--;
                    texCoordIndex[2]--;
                    tex_indices_.push_back(texCoordIndex);
                }
            }
        }
    }


    std::cout<<"[MySpecialMesh] Loaded "<<v_points_.size()<<" points, "<< v_faces_.size()<<" triangles."<<std::endl;
    if(!nan_points.empty())
        std::cout<<"[MySpecialMesh] Warning: Found "<< nan_points.size() <<" NAN coordinates, cleaned faces and points accordingly."<<std::endl<<std::endl;

    return isfile_readable;
}

bool MySpecialMesh::loadMOFF(const std::string moffFile, bool clear){
    std::cout<<"Reading file "<<moffFile<<std::endl;
    if(clear)
    {
        v_points_.clear();
        v_faces_.clear();
        v_points_separator_.clear();
        v_faces_separator_.clear();
        v_colors_.clear();
        v_edge_color_ind_.clear();
        v_face_color_ind_.clear();
        v_face_res_.clear();
        v_edge_real_color_ind_.clear();
    }
    v_points_separator_.push_back(v_points_.size());//Index of the first vertex of this mesh
    v_faces_separator_.push_back(v_faces_.size());//Index of the first face of this mesh
    std::vector<int32_t> old_indices;
    int32_t count = 0;

    std::vector<int32_t> nan_points;

    bool isfile_readable = false;

    int matches = 0;

    //Face managment variables (only working for triangles) TBD: improve this
    int32_t vertexIndex[3];

    FILE * file = fopen(moffFile.c_str(), "r");
    if( file == NULL ){
        printf("Could not open the .off file !\n");
        return false;
    }

    // ---- Read header ----
    char fileHeader[128];
    int res = fscanf(file, "%s\n", fileHeader);
    if(res==EOF)
        return false;
    if(strcmp(fileHeader,"MOFF")!=0)
    {
        printf("Header does not match mesh colors off format!\n");
        return false;
    }

    char colormapName[128];
    res = fscanf(file, "%s\n", colormapName);
    if(res==EOF)
        return false;
    //TODO: check png extension?

    //extract filename of png file from path
    std::size_t lastSlash = moffFile.rfind("/");
    std::string pngfilename;
    if (lastSlash!=std::string::npos)
        pngfilename = moffFile.substr(0,lastSlash) + "/" + colormapName;
    log(ALWAYS)<<"color file: "<<pngfilename<<endLog();
    //read colormap
    cv::Mat cvColorMat;
    cv::Mat cvImage = cv::imread(pngfilename, CV_LOAD_IMAGE_COLOR);
    cvImage.convertTo(cvColorMat, CV_8UC3);
    int colormapSize = cvColorMat.rows * cvColorMat.cols;
    log(ALWAYS)<<"colormap size = "<<colormapSize<<endLog();
    v_colors_.reserve(colormapSize);
    for(int i=0;i<cvColorMat.rows;++i)
    {
        for(int j=0;j<cvColorMat.cols;++j)
        {
            Vector3ui myColor = Vector3ui(cvColorMat.at<cv::Vec3b>(i,j)[2],cvColorMat.at<cv::Vec3b>(i,j)[1],cvColorMat.at<cv::Vec3b>(i,j)[0]);
            v_colors_.push_back(myColor);
        }
    }
    log(ALWAYS)<<"colormap size = "<<v_colors_.size()<<endLog();

    int vertexNumber, edgeNumber, faceNumber;
    res = fscanf(file,"%d %d %d\n", &vertexNumber, &faceNumber, &edgeNumber);

    log(ALWAYS)<<"Loading off file: "<<vertexNumber<<" vertices, "<<faceNumber<<" faces."<<endLog();
    //read vertices
    for(int vNum=0;vNum<vertexNumber;++vNum)
    {
        isfile_readable = true;
        Vector3f vertex;
        int vertexColorIndex;
        matches = fscanf(file, "%f %f %f %d\n", &vertex[0], &vertex[1], &vertex[2], &vertexColorIndex );
        if (matches == 4)
        {
            if(vertex(0) != vertex(0) || vertex(1) != vertex(1) || vertex(2) != vertex(2))
            {
                nan_points.push_back(v_points_.size());
                old_indices.push_back(-1);
            }
            else
            {
                v_points_.push_back(vertex);
                old_indices.push_back(count++);
            }
        }else
            isfile_readable = false;
    }

    log(ALWAYS)<<"vertices read. "<<v_points_.size()<<" points added, "<<nan_points.size()<<" nan points"<<endLog();

    for(int fNum=0; fNum<faceNumber;++fNum)
    {
        int faceVertexNumber;
        int faceRes;
        Vector3li edgeIndices;
        long faceIndex;
        matches = fscanf(file, " %d %d %d %d %d %ld %ld %ld %ld\n", &faceVertexNumber, &vertexIndex[0], &vertexIndex[1], &vertexIndex[2], &faceRes, &edgeIndices[0], &edgeIndices[1], &edgeIndices[2], &faceIndex);
        if (matches == 9)
        {
            //first part copied from below, with small addition for face texture indices
            if (    (unsigned int)(vertexIndex[0] + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                    (unsigned int)(vertexIndex[1] + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ||
                    (unsigned int)(vertexIndex[2] + v_points_separator_[v_points_separator_.size()-1]) > v_points_.size()+nan_points.size() ){
                std::cout<<"Error! Vertex index out of range: nb_points "<<v_points_.size()<<std::endl;

                std::cout<<"Indices: "<<vertexIndex[0] <<" "<< vertexIndex[1] <<" "<<vertexIndex[2]<<std::endl;
                return false;
            }
            else{ //TBD: add non triangular faces management
                bool is_triangle_safe = true;
                const unsigned long long int temp_index_ref = vertexIndex[0] + v_points_separator_[v_points_separator_.size()-1]; //From 1 to 0 starting point
                const unsigned long long int temp_index_edge1 = vertexIndex[1] + v_points_separator_[v_points_separator_.size()-1];
                const unsigned long long int temp_index_edge2 = vertexIndex[2] + v_points_separator_[v_points_separator_.size()-1];
                for(int32_t j = 0; j < nan_points.size(); ++j)
                {
                    if(nan_points[j] == temp_index_ref || nan_points[j] == temp_index_edge1  || nan_points[j] == temp_index_edge2  )
                    {
                        is_triangle_safe = false;
                        break;
                    }
                }
                if(is_triangle_safe)
                {
                    MyTriangle triangle;
                    triangle.ref = old_indices[vertexIndex[0]]   + v_points_separator_[v_points_separator_.size()-1];
                    triangle.edge1 = old_indices[vertexIndex[1]] + v_points_separator_[v_points_separator_.size()-1];
                    triangle.edge2 = old_indices[vertexIndex[2]] + v_points_separator_[v_points_separator_.size()-1];
                    v_faces_.push_back(triangle);
                    v_face_res_.push_back(faceRes);
                    //v_edge_color_ind_.push_back(edgeIndices);
                    v_face_color_ind_.push_back(faceIndex);
                    Vector3li triangleEdgeIndices;
                    for(int e=0;e<3;++e)    //for each edge
                    {
                        std::vector<size_t>::iterator it;
                        it = std::find(v_edge_real_color_ind_.begin(),v_edge_real_color_ind_.end(), std::abs(edgeIndices[e]));
                        if(it!=v_edge_real_color_ind_.end())    //index already saved
                        {
                            if(edgeIndices[e]<0)
                            {
                                triangleEdgeIndices[e]=-(it-v_edge_real_color_ind_.begin());
                            }
                            else    //should never happen (at least with manifold triangles)
                            {
                                triangleEdgeIndices[e]=it-v_edge_real_color_ind_.begin();
                            }
                        }
                        else
                        {
                            if(edgeIndices[e]<0)
                            {
                                triangleEdgeIndices[e]=-v_edge_real_color_ind_.size();
                            }
                            else
                            {
                                triangleEdgeIndices[e]=v_edge_real_color_ind_.size();
                            }
                            v_edge_real_color_ind_.push_back(std::abs(edgeIndices[e]));
                        }
                    }
                    v_edge_color_ind_.push_back(triangleEdgeIndices);
                }
            }
        }
        else{
            std::cout<<"File can't be read by this simple parser : ( Try exporting with other options, or check if faces are triangles... others not managed yet)"<<std::endl;
            return false;
        }
    }

    std::cout<<"[MySpecialMesh] Loaded "<<v_points_.size()<<" points, "<< v_faces_.size()<<" triangles."<<std::endl;
    if(!nan_points.empty())
        std::cout<<"[MySpecialMesh] Warning: Found "<< nan_points.size() <<" NAN coordinates, cleaned faces and points accordingly."<<std::endl<<std::endl;

    return isfile_readable;
}

void MySpecialMesh::exportAsOBJ(std::string filename){

    if(filename.empty())
        filename = s_file_name_;

    std::ofstream outFile;
    outFile.open(filename);

    if(outFile.is_open())
    {
        outFile << "o OutMesh"<<std::endl;
        for(std::vector<Vector3f>::iterator point_it = v_points_.begin() ; point_it != v_points_.end() ; ++point_it)
            outFile<<"v "<< (*point_it)[0] <<" "<< (*point_it)[1] <<" "<< (*point_it)[2] <<std::endl;
        for(std::vector<MyTriangle>::iterator v_faces_it = v_faces_.begin(); v_faces_it != v_faces_.end(); ++v_faces_it)
            outFile<<"f "<< v_faces_it->ref+1 <<" "<< v_faces_it->edge1+1 <<" "<< v_faces_it->edge2+1 <<std::endl;
    }
    else{
        std::cout<<"Error, could not open file..."<<std::endl;
        return;
    }
    outFile.close();
}

void MySpecialMesh::exportAsCOFF(std::string filename) const{
    if(filename.empty())
        filename = s_file_name_;
    if(v_colors_.size() != v_points_.size())
        std::cout<<"[MySpecialMesh::exportAsCOFF()] Color Error."<<std::endl;

    std::ofstream outFile;
    outFile.open(filename);

    if(outFile.is_open())
    {
        outFile << "COFF"<<std::endl;
        outFile << v_points_.size()<<" "<<v_faces_.size()<<" 0"<<std::endl;
        for(int32_t point_it = 0 ; (unsigned long int)point_it <v_points_.size() ; ++point_it)
            outFile<< v_points_[point_it](0)
                      <<" "<< v_points_[point_it](1)
                        <<" "<< v_points_[point_it](2)
                          <<" "<< v_colors_[point_it](0)/255.0
                            <<" "<< v_colors_[point_it](1)/255.0
                              <<" "<< v_colors_[point_it](2)/255.0
                                <<" 1.0"<< std::fixed <<std::endl;
        for(std::vector<MyTriangle>::const_iterator v_faces_it = v_faces_.begin(); v_faces_it != v_faces_.end(); ++v_faces_it)
            outFile<<"3 "<< v_faces_it->ref <<" "<< v_faces_it->edge1 <<" "<< v_faces_it->edge2 <<std::endl;
    }
    else{
        std::cout<<"Error, could not open file..."<<std::endl;
        return;
    }
    outFile.close();
}


void MySpecialMesh::exportAsMOFF(std::string filename) const{
	std::string rootname = filename.substr(0,filename.length()-4);

	std::string pngfilename = rootname + ".png";
	std::string ppmfilename = rootname + ".ppm";

    //extract filename of png file from path
    std::size_t lastSlash = pngfilename.rfind("/");
    if (lastSlash!=std::string::npos)
        pngfilename = pngfilename.substr(lastSlash);

	if(filename.empty())
        filename = s_file_name_;

    std::ofstream outFile;
    outFile.open(filename);

    if(outFile.is_open())
    {
        outFile << "MOFF"<<std::endl;
        outFile << pngfilename<<std::endl;
        outFile << v_points_.size()<<" "<<v_faces_.size()<<" 0"<<std::endl;
        //write vertices and their color coordinates (i.e. their line number...)
        for(int32_t point_it = 0 ; (unsigned long int)point_it <v_points_.size() ; ++point_it)
        {
            outFile<< v_points_[point_it](0)
                      <<" "<< v_points_[point_it](1)
                        <<" "<< v_points_[point_it](2)
						  <<" "<<point_it<< std::fixed <<std::endl;
        }
        //write faces: vertices number, vertices index, face resolution, edge color indices, and face color index
        for(int myRes=64;myRes>0;myRes--)
        {
            int face_it=0;
            for(std::vector<MyTriangle>::const_iterator v_faces_it = v_faces_.begin(); v_faces_it != v_faces_.end(); ++v_faces_it)
            {
                if(v_face_res_[face_it]==myRes)
                {
                	outFile<<"3 "<< v_faces_it->ref <<" "<< v_faces_it->edge1 <<" "<< v_faces_it->edge2
                      <<" "<< v_face_res_[face_it]
        				<<" "<< v_edge_color_ind_[face_it][0] <<" "<< v_edge_color_ind_[face_it][1] <<" "<< v_edge_color_ind_[face_it][2]
        				  <<" "<< v_face_color_ind_[face_it]
                    <<std::endl;
                }
            	face_it++;
            }

        }
        
    }
    else{
        log(ERROR)<<"Error, could not open file..."<<endLog();
        return;
    }
    outFile.close();
    //saving png file as ppm
	outFile.open(ppmfilename);
	int img_width = int(sqrt(v_colors_.size()))+2;
    int img_height=img_width;
    while((img_width*(img_height-1))>=v_colors_.size())
    {
        img_height-=1;
    }
	if(outFile.is_open())
	{
		outFile << "P3"<<std::endl;
		outFile << img_width<<" "<<img_height<<" 255"<<std::endl;
		outFile << "# Image Data"<<std::endl;
		for(int32_t point_it = 0 ; (unsigned long int)point_it < v_colors_.size() ; ++point_it)
			outFile << v_colors_[point_it](0) <<" "<< v_colors_[point_it](1) <<" "<< v_colors_[point_it](2) <<std::endl;
	}
	else{
		log(ERROR)<<"Error, could not open file..."<<endLog();
		return;
	}
	outFile.close();
	log(ALWAYS)<<"[SpaceTimeSampler] : Writing done!"<<endLog();
}

void MySpecialMesh::exportAsNMOFF(std::string filename) const{
    std::string rootname = filename.substr(0,filename.length()-4);

    std::string pngfilename = rootname + ".png";
    std::string ppmfilename = rootname + ".ppm";

    //extract filename of png file from path
    std::size_t lastSlash = pngfilename.rfind("/");
    if (lastSlash!=std::string::npos)
        pngfilename = pngfilename.substr(lastSlash);

    if(filename.empty())
        filename = s_file_name_;

    std::ofstream outFile;
    outFile.open(filename);

    if(outFile.is_open())
    {
        outFile << "MOFF"<<std::endl;
        outFile << pngfilename<<std::endl;
        outFile << v_points_.size()<<" "<<v_faces_.size()<<" 0"<<std::endl;
        //write vertices
        for(int32_t point_it = 0 ; (unsigned long int)point_it <v_points_.size() ; ++point_it)
        {
            short pv1, pv2, pv3;
            pv1 = short(v_points_[point_it](0)*65536/4);
            pv2 = short(v_points_[point_it](1)*65536/4);
            pv3 = short(v_points_[point_it](2)*65536/4);
            outFile<< char(pv1/256) << char(pv1 - pv1/256) << char(pv2/256) << char(pv2 - pv2/256) << char(pv3/256) << char(pv3 - pv3/256) << std::endl;
        }
        //write faces: vertices number, vertices index, face resolution, edge color indices, and face color index
        int color_ind=0;
        for(int myRes=1;myRes<=16;myRes*=2)     //TODO: pass max res as parameter
        {
            int face_it=0;
            
            outFile<<"r "<<myRes<<std::endl;
            for(std::vector<MyTriangle>::const_iterator v_faces_it = v_faces_.begin(); v_faces_it != v_faces_.end(); ++v_faces_it)
            {
                if(v_face_res_[face_it]==myRes)
                {
                    //outFile<< short(v_faces_it->ref) <<" "<< short(v_faces_it->edge1) <<" "<< short(v_faces_it->edge2);
                    
                    outFile<< char(short(v_faces_it->ref)/256) << char(short(v_faces_it->ref) - short(v_faces_it->ref)/256)
                      << char(short(v_faces_it->edge1)/256) << char(short(v_faces_it->edge1) - short(v_faces_it->edge1)/256)
                        << char(short(v_faces_it->edge2)/256) << char(short(v_faces_it->edge2) - short(v_faces_it->edge2)/256);

                    if(myRes>=1)
                    {
                        //outFile<<" "<< v_edge_color_ind_[face_it][0] <<" "<< v_edge_color_ind_[face_it][1] <<" "<< v_edge_color_ind_[face_it][2];   //edges color indices
                        std::vector<unsigned char> myColorBytes = v_colors_bitarray[color_ind].getWrittenBytes();
                        for(int ch=0;ch<myColorBytes.size();++ch)
                        {
                            outFile<<myColorBytes[ch];
                            /*
                            unsigned int rmdr=(unsigned int)(myColorBytes[ch]);
                            if(rmdr>256)
                            {
                                log(ALWAYS)<<"rmdr = "<<rmdr<<", char: "<<int(myColorBytes[ch])<<", color_ind = "<<color_ind<<endLog();
                            }
                            outFile<<" ";
                            for(int b=0;b<8;++b)
                            {
                                unsigned int myBit = (unsigned int)(rmdr/std::pow(2,7-b));
                                outFile<<myBit;
                                rmdr-= myBit*std::pow(2,7-b);
                            }
                            */
                            
                        }
                        ++color_ind;
                    }
                    outFile<<std::endl;
                }
                face_it++;
            }

        }
        log(ALWAYS)<<"Finished writing, color_ind = "<<color_ind<<endLog();

        /*
        for(int face_it=0; face_it<v_faces_.size();++face_it)
            outFile<<"3 "<< v_faces_[face_it]->ref <<" "<<v_faces_[face_it]->edge1 <<" "<< v_faces_[face_it]->edge2
                      <<" "<< v_face_res_[face_it]
                        <<" "<< v_edge_color_ind[face_it][0] <<" "<< v_edge_color_ind[face_it][1] <<" "<< v_edge_color_ind[face_it][2]
                          <<" "<< v_face_color_ind[face_it] <<std::endl;
        */
    }
    else{
        std::cout<<"Error, could not open file..."<<std::endl;
        return;
    }
    outFile.close();
    //saving png file as ppm
    //std::ofstream outFile;
    outFile.open(ppmfilename);
    int img_width = int(sqrt(v_colors_.size()))+2;
    int img_height=img_width;
    while((img_width*(img_height-1))>=v_colors_.size())
    {
        img_height-=1;
    }
    if(outFile.is_open())
    {
        outFile << "P3"<<std::endl;
        outFile << img_width<<" "<<img_height<<" 255"<<std::endl;
        outFile << "# Image Data"<<std::endl;
        for(int32_t point_it = 0 ; (unsigned long int)point_it < v_colors_.size() ; ++point_it)
            outFile << v_colors_[point_it](0) <<" "<< v_colors_[point_it](1) <<" "<< v_colors_[point_it](2) <<std::endl;
    }
    else{
        std::cout<<"Error, could not open file..."<<std::endl;
        return;
    }
    outFile.close();
    log(ALWAYS)<<"[SpaceTimeSampler] : Writing done!"<<endLog();
}


void MySpecialMesh::exportAsMPLY(std::string filename) const{
    std::string rootname = filename.substr(0,filename.length()-4);

    std::string pngfilename = rootname + ".png";
    std::string ppmfilename = rootname + ".ppm";

    //extract filename of png file from path
    std::size_t lastSlash = pngfilename.rfind("/");
    if (lastSlash!=std::string::npos)
        pngfilename = pngfilename.substr(lastSlash);

    if(filename.empty())
        filename = s_file_name_;

    std::ofstream outFile;
    outFile.open(filename);

    if(outFile.is_open())
    {
        //header
        outFile << "ply"<<std::endl;
        outFile << "format ascii 1.0"<<std::endl;
        //putting path to colormap as a comment for now
        outFile << "comment " << pngfilename<<std::endl;
        outFile << "element vertex " << v_points_.size()<<std::endl;
        outFile << "property float32 x"<<std::endl;
        outFile << "property float32 y"<<std::endl;
        outFile << "property float32 z"<<std::endl;
        outFile << "property uint16 color_index"<<std::endl;
        outFile << "element face "<<v_faces_.size()<<std::endl;
        outFile << "property list uint8 int32 vertex_index"<<std::endl;
        outFile << "property uint8 faceRes"<<std::endl;
        outFile << "property int32 ei1"<<std::endl;
        outFile << "property int32 ei2"<<std::endl;
        outFile << "property int32 ei3"<<std::endl;
        outFile << "property int32 faceIndex"<<std::endl;
        outFile << "end_header"<<std::endl;
        //end header
        //The rest of the file is just like MOFF
        //write vertices and their color coordinates (i.e. their line number...)
        for(int32_t point_it = 0 ; (unsigned long int)point_it <v_points_.size() ; ++point_it)
            outFile<< v_points_[point_it](0)
                      <<" "<< v_points_[point_it](1)
                        <<" "<< v_points_[point_it](2)
                          <<" "<<point_it<< std::fixed <<std::endl;
        //write faces: vertices number, vertices index, face resolution, edge color indices, and face color index
        int face_it=0;
        for(std::vector<MyTriangle>::const_iterator v_faces_it = v_faces_.begin(); v_faces_it != v_faces_.end(); ++v_faces_it)
        {
            outFile<<"3 "<< v_faces_it->ref <<" "<< v_faces_it->edge1 <<" "<< v_faces_it->edge2
              <<" "<< v_face_res_[face_it]
                <<" "<< v_edge_color_ind_[face_it][0] <<" "<< v_edge_color_ind_[face_it][1] <<" "<< v_edge_color_ind_[face_it][2]
                  <<" "<< v_face_color_ind_[face_it] <<std::endl;
            face_it++;
        }

        /*
        for(int face_it=0; face_it<v_faces_.size();++face_it)
            outFile<<"3 "<< v_faces_[face_it]->ref <<" "<<v_faces_[face_it]->edge1 <<" "<< v_faces_[face_it]->edge2
                      <<" "<< v_face_res_[face_it]
                        <<" "<< v_edge_color_ind[face_it][0] <<" "<< v_edge_color_ind[face_it][1] <<" "<< v_edge_color_ind[face_it][2]
                          <<" "<< v_face_color_ind[face_it] <<std::endl;
        */
    }
    else{
        std::cout<<"Error, could not open file..."<<std::endl;
        return;
    }
    outFile.close();
    //saving png file as ppm
    //std::ofstream outFile;
    outFile.open(ppmfilename);
    int img_width = int(sqrt(v_colors_.size()))+2;
    int img_height=img_width;
    while((img_width*(img_height-1))>=v_colors_.size())
    {
        img_height-=1;
    }
    if(outFile.is_open())
    {
        outFile << "P3"<<std::endl;
        outFile << img_width<<" "<<img_height<<" 255"<<std::endl;
        outFile << "# Image Data"<<std::endl;
        for(int32_t point_it = 0 ; (unsigned long int)point_it < v_colors_.size() ; ++point_it)
            outFile << v_colors_[point_it](0) <<" "<< v_colors_[point_it](1) <<" "<< v_colors_[point_it](2) <<std::endl;
    }
    else{
        std::cout<<"Error, could not open file..."<<std::endl;
        return;
    }
    outFile.close();
    log(ALWAYS)<<"[SpaceTimeSampler] : Writing done!"<<endLog();
}


void MySpecialMesh::exportAsMinPLY(std::string filename) const{
    std::string rootname = filename.substr(0,filename.length()-4);

    std::string pngfilename = rootname + ".png";
    std::string ppmfilename = rootname + ".ppm";

    //extract filename of png file from path
    std::size_t lastSlash = pngfilename.rfind("/");
    if (lastSlash!=std::string::npos)
        pngfilename = pngfilename.substr(lastSlash);

    if(filename.empty())
        filename = s_file_name_;

    std::ofstream outFile;
    outFile.open(filename);

    if(outFile.is_open())
    {
        //header
        outFile << "ply"<<std::endl;
        outFile << "format ascii 1.0"<<std::endl;
        //putting path to colormap as a comment for now
        outFile << "comment " << pngfilename<<std::endl;
        outFile << "element vertex " << v_points_.size()<<std::endl;
        outFile << "property float32 x"<<std::endl;
        outFile << "property float32 y"<<std::endl;
        outFile << "property float32 z"<<std::endl;
        outFile << "property uint8 r"<<std::endl;
        outFile << "property uint8 g"<<std::endl;
        outFile << "property uint8 b"<<std::endl;
        outFile << "element face "<<v_faces_.size()<<std::endl;
        outFile << "property list uint8 int32 vertex_index"<<std::endl;
        outFile << "end_header"<<std::endl;
        //end header
        //write vertices and their colors
        for(int32_t point_it = 0 ; (unsigned long int)point_it <v_points_.size() ; ++point_it)
            outFile<< v_points_[point_it](0)
                      <<" "<< v_points_[point_it](1)
                        <<" "<< v_points_[point_it](2)
                          <<" "<<v_colors_[point_it](0) <<" "<< v_colors_[point_it](1) <<" "<< v_colors_[point_it](2)<<std::endl;
        //write faces: vertices number, vertices index
        int face_it=0;
        for(std::vector<MyTriangle>::const_iterator v_faces_it = v_faces_.begin(); v_faces_it != v_faces_.end(); ++v_faces_it)
        {
            outFile<<"3 "<< v_faces_it->ref <<" "<< v_faces_it->edge1 <<" "<< v_faces_it->edge2
              <<std::endl;
            face_it++;
        }

    }
    else{
        std::cout<<"Error, could not open file..."<<std::endl;
        return;
    }
    outFile.close();
    // //saving png file as ppm
    // //std::ofstream outFile;
    // outFile.open(ppmfilename);
    // int img_width = int(sqrt(v_colors_.size()))+2;
    // int img_height=img_width;
    // while((img_width*(img_height-1))>=v_colors_.size())
    // {
    //     img_height-=1;
    // }
    // if(outFile.is_open())
    // {
    //     outFile << "P3"<<std::endl;
    //     outFile << img_width<<" "<<img_height<<" 255"<<std::endl;
    //     outFile << "# Image Data"<<std::endl;
    //     for(int32_t point_it = 0 ; (unsigned long int)point_it < v_colors_.size() ; ++point_it)
    //         outFile << v_colors_[point_it](0) <<" "<< v_colors_[point_it](1) <<" "<< v_colors_[point_it](2) <<std::endl;
    // }
    // else{
    //     std::cout<<"Error, could not open file..."<<std::endl;
    //     return;
    // }
    // outFile.close();
    log(ALWAYS)<<"[SpaceTimeSampler] : Writing done!"<<endLog();
}


void MySpecialMesh::exportAsFullPLY(std::string filename) const{
    std::string rootname = filename.substr(0,filename.length()-4);

    std::string pngfilename = rootname + ".png";
    std::string ppmfilename = rootname + ".ppm";

    //extract filename of png file from path
    std::size_t lastSlash = pngfilename.rfind("/");
    if (lastSlash!=std::string::npos)
        pngfilename = pngfilename.substr(lastSlash);

    if(filename.empty())
        filename = s_file_name_;

    std::ofstream outFile;
    outFile.open(filename);

    if(outFile.is_open())
    {
        //header
        outFile << "ply"<<std::endl;
        outFile << "format ascii 1.0"<<std::endl;
        //putting path to colormap as a comment for now
        outFile << "comment " << pngfilename<<std::endl;
        outFile << "element vertex " << v_points_.size()<<std::endl;
        outFile << "property float32 x"<<std::endl;
        outFile << "property float32 y"<<std::endl;
        outFile << "property float32 z"<<std::endl;
        outFile << "property uint8 r"<<std::endl;
        outFile << "property uint8 g"<<std::endl;
        outFile << "property uint8 b"<<std::endl;

        outFile << "element face "<<v_faces_.size()<<std::endl;
        outFile << "property list uint8 int32 vertex_index"<<std::endl;
        outFile << "property uint8 faceRes"<<std::endl;
        outFile << "property uint8 edgeRes1"<<std::endl;
        outFile << "property uint8 edgeRes2"<<std::endl;
        outFile << "property uint8 edgeRes3"<<std::endl;

        //edge samples (max res = 8 for now)
        for(int e=1;e<=3;++e)
        {
            for(int cI=1;cI<16;++cI)
            {
                outFile << "property uint8 re"<<e<<"_"<<cI<<std::endl;
                outFile << "property uint8 ge"<<e<<"_"<<cI<<std::endl;
                outFile << "property uint8 be"<<e<<"_"<<cI<<std::endl;
            }
        }
        //face samples
        for(int cI=0;cI<105;++cI)
        {
            outFile << "property uint8 rf"<<cI<<std::endl;
            outFile << "property uint8 gf"<<cI<<std::endl;
            outFile << "property uint8 bf"<<cI<<std::endl;
        }
        outFile << "end_header"<<std::endl;
        //end header
        //The rest of the file is just like MOFF
        //write vertices and their color (coordinates is their line number...)
        for(int32_t point_it = 0 ; (unsigned long int)point_it <v_points_.size() ; ++point_it)
            outFile<< v_points_[point_it](0)
                      <<" "<< v_points_[point_it](1)
                        <<" "<< v_points_[point_it](2)
                          <<" "<< v_colors_[point_it](0) <<" "<< v_colors_[point_it](1) <<" "<< v_colors_[point_it](2) <<std::endl;
                          //<<" "<<point_it<< std::fixed <<std::endl;
        //write faces: vertices number, vertices index, face resolution, edge color samples (pad with 0), and face color samples (pad with 0)
        int face_it=0;
        for(std::vector<MyTriangle>::const_iterator v_faces_it = v_faces_.begin(); v_faces_it != v_faces_.end(); ++v_faces_it)
        {
            unsigned short faceRes = v_face_res_[face_it];
            outFile<<"3 "<< v_faces_it->ref <<" "<< v_faces_it->edge1 <<" "<< v_faces_it->edge2
              <<" "<< faceRes;
                //edges
                for(int e=0;e<3;++e)
                {
                    long edgeInd = std::abs(v_edge_color_ind_[face_it][e]);
                    //edge Res
                    int edgeRes = v_colors_[edgeInd](0);
                    outFile<<" "<<edgeRes;
                    for(int samp=1;samp<edgeRes;++samp)
                    {
                        long curInd = edgeInd+samp;
                        outFile<<" "<<v_colors_[curInd](0)<<" "<<v_colors_[curInd](1)<<" "<<v_colors_[curInd](2);
                    }
                    //pad with zeros
                    for(int i=0;i<16-edgeRes;++i)
                    {
                        outFile<<" "<<0<<" "<<0<<" "<<0;
                    }
                }
                //face samples
                unsigned long faceInd = v_face_color_ind_[face_it];
                int sampNum = (faceRes-1)*(faceRes-2)/2;
                for(int samp=0;samp<sampNum;++samp)
                {
                    outFile<<" "<<v_colors_[faceInd+samp](0)<<" "<<v_colors_[faceInd+samp](1)<<" "<<v_colors_[faceInd+samp](2);
                }
                for(int i=0;i<(105-sampNum);++i)
                {
                    outFile<<" "<<0<<" "<<0<<" "<<0;
                }
                outFile<<std::endl;

            face_it++;
        }
    }
    else{
        std::cout<<"Error, could not open file..."<<std::endl;
        return;
    }
    outFile.close();
    //saving png file as ppm
    //std::ofstream outFile;
    outFile.open(ppmfilename);
    int img_width = int(sqrt(v_colors_.size()))+2;
    int img_height=img_width;
    while((img_width*(img_height-1))>=v_colors_.size())
    {
        img_height-=1;
    }
    if(outFile.is_open())
    {
        outFile << "P3"<<std::endl;
        outFile << img_width<<" "<<img_height<<" 255"<<std::endl;
        outFile << "# Image Data"<<std::endl;
        for(int32_t point_it = 0 ; (unsigned long int)point_it < v_colors_.size() ; ++point_it)
            outFile << v_colors_[point_it](0) <<" "<< v_colors_[point_it](1) <<" "<< v_colors_[point_it](2) <<std::endl;
    }
    else{
        std::cout<<"Error, could not open file..."<<std::endl;
        return;
    }
    outFile.close();
    log(ALWAYS)<<"[SpaceTimeSampler] : Writing done!"<<endLog();
}





