#include <iostream>
#include <string>
#include "stdio.h"
#include "stdlib.h"


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "mesh.h"

std::string output_name_format;

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
            fr = FACE_RES_RATIO;
            dst = DOWNSAMPLE_THRESHOLD;
            std::string colored_output_name = (boost::format(output_name_format+"moff") % hyper_volume->getFrameNumber(frame)).str();

            // if (boost::filesystem::exists(colored_output_name))
            // {
            //     log(WARN)<<"Skipping file "<<colored_output_name<<": already exists."<<endLog();
            //     continue;
            // }

            log(ALWAYS)<<"Generating file "<<colored_output_name<<endLog();
            mesh_to_be_cleaned.cleanAndColor(hyper_volume, fr, dst);

        }
        else if(om.mode_ == 'Z')
        {
            int quantFactor = QUANT_FACTOR;
            float quantMatCoefs[] = {1,QUANT_MAT_L1,QUANT_MAT_L2};
            MySpecialMesh mesh_to_be_cleaned(hyper_volume->getFrameNumber(frame),input_name);
            log(ALWAYS)<<"mesh loaded."<<endLog();

            std::string compressed_output_name = (boost::format(output_name_format+"zoff") % hyper_volume->getFrameNumber(frame)).str();
            mesh_to_be_cleaned.compressColoredMesh(hyper_volume,quantFactor,quantMatCoefs);
            mesh_to_be_cleaned.exportAsZOFF(compressed_output_name);
        }

        else if(om.mode_ == 'X')    //compress. test. Matt
        {
            int quantFactor = QUANT_FACTOR;
            float quantMatCoefs[] = {1,QUANT_MAT_L1,QUANT_MAT_L2};
            MySpecialMesh mesh_to_be_cleaned(hyper_volume->getFrameNumber(frame),input_name);
            log(ALWAYS)<<"mesh loaded."<<endLog();
            std::string compressed_output_name = (boost::format(output_name_format+"_extracted.moff") % hyper_volume->getFrameNumber(frame)).str();
            mesh_to_be_cleaned.uncompressColoredMesh(hyper_volume,quantFactor,quantMatCoefs);
            mesh_to_be_cleaned.exportAsMOFF(compressed_output_name);
        }
    }
    return 0;
}

