#include <iostream>
#include <string>
#include "stdio.h"
#include "stdlib.h"


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "mesh.h"

///TBD: set this as a parameter?
std::string output_name_format;


// void removeDupliVertexCloseHolesAndSmooth(const std::string &input_name, const std::string &output_name){
//     log(ALWAYS) << "[Main] : Removing Duplicated Vertices, Closing holes and Smoothing..." <<endLog();
//     system(std::string("meshlabserver -i "+input_name+" -o "+output_name+" -s ./clean_HC_LaplacianSmoothing.mlx -om vc").c_str());
// }


int main(int argc, char **argv)
{

    ///***************** Implicit function setup *************************
    /* Read the command line options and store the parameters.
           If the user requests the help page, print it to screen and quit.
    */
    OptionManager om;
    if (om.parseOptions(argc,argv) == HELP){
        return 1;
    }
    om.displayOptions();

    /*Initialise the logger parameters:
      - enableColors allows to use a color for the different login levels.
      - setLogLevel allows to set the log level.
    */
    enableColors(om.use_colors_);
    setLogLevel(static_cast<LogLevel>(om.log_level_));

    /* Initialize Space Time (4D) point container according to user input
      */

    SpaceTimeSampler *hyper_volume = new SpaceTimeSampler(om);
    output_name_format = om.get_output_folder()+"%06i.";
    std::string temp_out_folder = om.get_output_folder();
    std::string input_mesh_format = om.input_mesh_;

    /// 0 -> first frame
    for(int frame = 0; frame < hyper_volume->getNumberOfFrames() ; ++frame)
    {

        std::string input_name = (boost::format(input_mesh_format) % hyper_volume->getFrameNumber(frame)).str();    //Mesh file to be loaded (OBJ, or OFF)

        if (om.mode_ == 'C') //Color obj frames contained in Temp folder
        {
            
            
            MySpecialMesh mesh_to_be_cleaned(hyper_volume->getFrameNumber(frame),input_name);
            hyper_volume->loadImages(frame);                           //Only load images and silhouettes in cameras

            int totalCamNum = hyper_volume->getNumCams();

            int fr, dst;

            
            // std::vector<int> frValues = {0,5,20,40};
            std::vector<int> frValues = {5};
            // std::vector<int> dstValues = {0,50,80,100,150,180};
            // std::vector<int> dstValues = {0, 20, 30, 50};
            std::vector<int> dstValues = {50};
            
            for(int frInd=0;frInd<frValues.size();++frInd)
            {
                fr = frValues[frInd];
                for(int dstInd=0;dstInd<dstValues.size();++dstInd)
                {
                    dst = dstValues[dstInd];
                    // std::string colored_output_name = (boost::format(output_name_format+std::to_string(fr)+"_"+std::to_string(dst)+".off") % hyper_volume->getFrameNumber(frame)).str();
                    std::string colored_output_name = (boost::format(output_name_format+"off") % hyper_volume->getFrameNumber(frame)).str();
            

                    // if (boost::filesystem::exists(colored_output_name))
                    // {
                    //     log(ALWAYS)<<"Skipping file "<<colored_output_name<<endLog();
                    //     continue;
                    // }
                    log(ALWAYS)<<"Generating file "<<colored_output_name<<endLog();
                    mesh_to_be_cleaned.cleanAndColor(hyper_volume, fr, dst);
        
                    mesh_to_be_cleaned.exportAsMOFF(colored_output_name);
                    log(ALWAYS)<<"exported as "<<colored_output_name<<endLog();
                }
            }



        }
        else if(om.mode_ == 'Z')
        {
            
            int quantFactor = 16384;
            // float quantMatCoefs[] = {1,0.1,0.02};

            // std::vector<int> quantFactorValues = {4096,6144,8192,12288,16384, 24576, 32768};
            // std::vector<int> quantFactorValues = {24576, 32768};
            // std::vector<int> quantFactorValues = {4096, 8192, 16384};
            std::vector<int> quantFactorValues = {4096};
            // for(quantFactor=8192;quantFactor<=16384;quantFactor+=4096)
            for(int quantFactorInd=0;quantFactorInd<quantFactorValues.size();++quantFactorInd)
            {
                quantFactor = quantFactorValues[quantFactorInd];
                for(float sF=1.0;sF<=1.0;sF+=0.3)
                    for(float sF2=0.01;sF2<=0.01;sF2+=0.04)
                    {
                        float quantMatCoefs[] = {1,sF,sF2};
                        MySpecialMesh mesh_to_be_cleaned(hyper_volume->getFrameNumber(frame),input_name);
                        log(ALWAYS)<<"mesh loaded."<<endLog();
                        //std::string compressed_output_name = (boost::format(output_name_format+"comp_"+std::to_string(quantFactor)+"_"+std::to_string(10*quantMatCoefs[1])+"_"+std::to_string(100*quantMatCoefs[2])+".off") % hyper_volume->getFrameNumber(frame)).str();
                        // std::string compressed_output_name = (boost::format(output_name_format+"comp0m_%i_%.1f_%.1f.off") % hyper_volume->getFrameNumber(frame) % quantFactor % (10*quantMatCoefs[1]) %  (100*quantMatCoefs[2])).str();
                        std::string compressed_output_name = (boost::format(output_name_format+"zoff") % hyper_volume->getFrameNumber(frame)).str();
                        
                        mesh_to_be_cleaned.compressColoredMesh(hyper_volume,quantFactor,quantMatCoefs);
                        
                        mesh_to_be_cleaned.exportAsZOFF(compressed_output_name);
                    }
            }
        }

        else if(om.mode_ == 'X')    //compress. test. Matt
        {
            
            int quantFactor = 16384;
            // float quantMatCoefs[] = {1,0.1,0.02};

            // std::vector<int> quantFactorValues = {4096,6144,8192,12288,16384, 24576, 32768};
            // std::vector<int> quantFactorValues = {24576, 32768};
            // std::vector<int> quantFactorValues = {4096, 8192, 16384};
            std::vector<int> quantFactorValues = {4096};
            for(int quantFactorInd=0;quantFactorInd<quantFactorValues.size();++quantFactorInd)
            {
                quantFactor = quantFactorValues[quantFactorInd];
                for(float sF=1.0;sF<=1.0;sF+=0.3)
                    for(float sF2=0.01;sF2<=0.01;sF2+=0.04)
                    {
                        float quantMatCoefs[] = {1,sF,sF2};
                        MySpecialMesh mesh_to_be_cleaned(hyper_volume->getFrameNumber(frame),input_name);
                        log(ALWAYS)<<"mesh loaded."<<endLog();
                        //std::string compressed_output_name = (boost::format(output_name_format+"comp_"+std::to_string(quantFactor)+"_"+std::to_string(10*quantMatCoefs[1])+"_"+std::to_string(100*quantMatCoefs[2])+".off") % hyper_volume->getFrameNumber(frame)).str();
                        std::string compressed_output_name = (boost::format(output_name_format+"comp0m_%i_%.1f_%.1f.off") % hyper_volume->getFrameNumber(frame) % quantFactor % (10*quantMatCoefs[1]) %  (100*quantMatCoefs[2])).str();
                        mesh_to_be_cleaned.uncompressColoredMesh(hyper_volume,quantFactor,quantMatCoefs);
                        mesh_to_be_cleaned.exportAsMOFF(compressed_output_name);
                    }
            }
        }
    }
    return 0;
}

