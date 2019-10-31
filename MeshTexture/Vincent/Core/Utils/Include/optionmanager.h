/* *************************************************
 * Copyright (C) 2014 INRIA
 * *************************************************/
#ifndef OPTIONMANAGER_H
#define OPTIONMANAGER_H

#include "Logger.h"
#include <boost/program_options.hpp>
#include "boost/filesystem.hpp"

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/optional.hpp>
#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <fstream>

/*! \file optionmanager.h
    \author Vincent LEROY
    \brief A class to handle program options and input variables.
*/


namespace po =boost::program_options;
using namespace boost::filesystem;

namespace {
const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;
const size_t HELP = 3;
}


class OptionManager
{
public:
    OptionManager();
    ~OptionManager();


    void bindOptions();
    int  writeConfigurationFile();
    void displayOptions();
    bool writeInConfigurationFile(const std::string& message);
    int parseOptions(int argc,char** argv);

    void getCamerasPath(const std::string& filename) const;

    //! Recover Camera Ids according to the image sequence argument
    /*! Extract folder name from image sequence, containing all the camera folders
        To do so, we search the first instance of "%0"
        from there we scan the folder and recover every int values in respect to camera name format.

        Designed to be used as:
        if(boost::optional< std::vector<size_t> > o_vector = getCamIds(...))
            myresultingvector = o_vector*;

            or
        try {
          std::vector<size_t> vector = getCamIds(...).value();
        }
        catch (const boost::bad_optional_access&) {
          // deal with it
        }
    */
    boost::optional< std::vector< size_t > > getCamIds() const;

    std::string get_output_folder() const;

private:
    //! A method that returns the generic options description.
    /*! Returns the generic options description.

    \return the generic options description.
    */
    po::options_description generic_options_description();
    po::options_description config_options_description();
    po::options_description advanced_options_description();
    po::options_description debug_options_description();
    po::options_description optional_options_description();

    bool createConfigurationFile(const std::string& filename) const;

    std::string get_output_path(const std::string& filename) const;

    /*! Every public member refers to an option.
    The descriptions stand for the option and the variable.
*/
public:

    /* Here the members are public because they need to be easily get and set.
      It is not very clean but avoid using some getters/setters.
    */

    //! Mode of the reconstruction (can be 'S' (Static) 'D' (Dynamic) 'C' (Cleaning)
    //! mode_
    //!

    char mode_;


    //! The base name of the output sequence.
    /*! The base name of the output sequence.
        We will build the filename of each output frame using this string.
        example : /path/to/output/folder/output_mesh_%05d.off where %05d is the frame number.
    */
    std::string output_folder_;

    //! The base name of the image sequence          We assume the first %0ni to be the camera ID and the second to be the frame number.
    /*! The base name of the image sequence
        We will build the filename of each input frame using this string.
        example :  /rootpath/camera%03i/path/to/input/images/images_sequences_%05i.jpg
    */
    std::string images_sequences_;

    //! The base name for Projection Matrices.
    /*! The base name for Projection Matrices.
        We will build the projection matrix filename of each camera using this string.
        example :  /rootpath/camera%03i/projmat.txt. where the %0ni is the camera ID
    */
    std::string projection_matrices_;

    //! Path to the configuration file.
    /*! Path to the configuration file.
      This string is optional considering the configuration file is optional.
    */
    std::string config_file_;

    //! The base name of the input OFF or OBJ mesh file.
    /*! The name of the input OFF or OBJ mesh file.
        Used in all three modes, to load a precolored mesh.
        We will build the mesh filename of each frame using this string.
        example: /path/to/input/folder/input_mesh_%05d.off where %05d is the frame number.
    */
    std::string input_mesh_;

    //! The first frame of the input sequence.
    int first_frame_;

    //! The last frame of the input sequence.
    int last_frame_;

    //! process the sequence from the last frame to the first frame.
    bool backward_processing_;


    bool create_config_file_;
    bool use_colors_;
    int log_level_;


private:
    //! The path to the output. We will store the output meshes and the summary file on this folder.
    std::string output_path_;


};

//! Turn a number into a string
/*! \relates OptionManager
    \param T the type of the number (int, float...)
    \param number the number we want to convert
    \return a string
*/
template<typename T> std::string num_to_string(T number)
{
    std::ostringstream stream;
    stream << number;
    return stream.str();
}

#endif // OPTIONMANAGER_H



