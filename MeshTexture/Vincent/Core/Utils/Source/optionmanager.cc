#include "../Include/optionmanager.h"

/**
 * @brief OptionManager::OptionManager
 */
OptionManager::OptionManager()
{
    mode_ = 'S';
    output_folder_ = "";
    images_sequences_ = "";
    silhouettes_sequences_ = "";
    projection_matrices_ = "";

    nb_interest_centroids_ = 50;
    reconstruct_cameras_ = false;

    config_file_ = "";
    first_frame_ = 0;
    last_frame_ = 100;

    backward_processing_ = false;

    output_path_ = "";
    create_config_file_ = false;

    use_colors_ = true;
    log_level_ = 0;
}


/**
 * @brief OptionManager::~OptionManager
 */
OptionManager::~OptionManager(){

}


/**
 * @brief OptionManager::get_output_path
 * @param filename
 * @return
 */
std::string OptionManager::get_output_path(const std::string& filename) const{
    size_t found = filename.find_last_of("/");
    return filename.substr(0,found);
}



/**
 * @brief OptionManager::getCamIds
 * @return
 */
boost::optional< std::vector<size_t> > OptionManager::getCamIds() const{

    path cameras_folder = images_sequences_.substr(0,images_sequences_.substr(0,images_sequences_.find_first_of("%")).find_last_of("/"));
    std::string camera_names = images_sequences_.substr(
              images_sequences_.substr(0,images_sequences_.find_first_of("%")).find_last_of("/") + 1,
              images_sequences_.substr(images_sequences_.find_first_of("%")).find_first_of("/") + images_sequences_.find_first_of("%") - 1 -
                                                                                    images_sequences_.substr(0,images_sequences_.find_first_of("%")).find_last_of("/")
                );
    std::vector<size_t> v_out ;

    if(!exists(cameras_folder)){
        log(ERROR)<< " [Option Manager] : Error, "<< cameras_folder.c_str() <<" does not exists, no camera detected."<<endLog();
        return boost::none;
    }

    //Get every folder and extract integer IDs from camera names
    directory_iterator end_itr;
      for ( directory_iterator itr( cameras_folder );itr != end_itr;++itr ){
        if ( is_directory(itr->status()) ){
            std::string folder_name = itr->path().string();
            boost::regex expr{"[^0-9]"};
            v_out.push_back(std::atoi(boost::regex_replace(folder_name.substr(folder_name.find_last_of("/")),expr,"").c_str()));

            if(std::strcmp(folder_name.substr(folder_name.find_last_of("/")+1).c_str(),(boost::format(camera_names) % v_out[v_out.size()-1]).str().c_str()) != 0)
                log(ERROR)<< "[ Option Manager ] : Camera Name/Index unhandled mismatch, please investigate. "<< endLog();




        }
      }
      if(v_out.empty())
          return boost::none;
      else
          return v_out;

}
/**
 * @brief OptionManager::createConfigurationFile
 * @param filename
 * @return
 */
bool OptionManager::createConfigurationFile(const std::string &filename) const{
    //First we check if the file already exists.
    if (exists(filename) && is_regular_file(filename)) return true;

    //The file doesn't exist, we create create it's containing directory if needed.
    std::string target_dir = get_output_path(filename);

    if(exists(target_dir)){
        if (is_directory(target_dir)) return true;
        else{
            log(ERROR) << "[Option Manager] : invalid file name " + filename << endLog();
            return false;
        }
    }
    //We try to create the target folder.
    else if(create_directories(target_dir)){
        log(ALWAYS) << "[Option Manager] : creating folder " + target_dir << endLog();
        return false;
    }

    return false;
}


/**
 * @brief OptionManager::writeConfigurationFile
 * @return
 */
int OptionManager::writeConfigurationFile(){
    log(ALWAYS) << "[Option Manager] : generating sequence summary file." << endLog();


    std::string seq_config_path = get_output_path(output_folder_) + "/sequence_configuration.txt";

    createConfigurationFile(seq_config_path);

    std::ofstream seq_config(seq_config_path.c_str());

    if (seq_config.is_open()){
        seq_config << "#" <<    boost::posix_time::second_clock::local_time() << std::endl;
        seq_config << "Mode                                     = " << mode_ << std::endl;
        seq_config << "Output Folder                            = " << output_folder_ << std::endl;
        seq_config << "Images Sequence                          = " << images_sequences_ << std::endl;
        seq_config << "Silhouettes Sequence                     = " << silhouettes_sequences_ << std::endl;
        seq_config << "Projection Matrices                      = " << projection_matrices_ << std::endl;
        seq_config << "firstFrame                               = " << first_frame_ << std::endl;
        seq_config << "lastFrame                                = " << last_frame_ << std::endl;
        seq_config << "backwardProcessing                       = " << backward_processing_ << std::endl;
        seq_config << "Number of interest Points                = " << nb_interest_centroids_ << std::endl;
        seq_config << "Enable Camera Reconstruction             = " << reconstruct_cameras_ << std::endl;
        seq_config.close();
    }

    else{
        log(ERROR) << "[Option Manager] : unable to write sequence summary file." << endLog();
    }
}


/**
 * @brief OptionManager::generic_options_description
 * @return
 */
po::options_description OptionManager::generic_options_description(){

    po::options_description generic("Generic options");
    generic.add_options()
            ("version,v", "print version string")
            ("help,h", "produce help message");

    return generic;
}


/**
 * @brief OptionManager::config_options_description
 * @return
 */
po::options_description OptionManager::config_options_description(){
    po::options_description config("Configuration");
    config.add_options()
            ("configuration_file", po::value<std::string>(&config_file_),"Configuration file")
            ("mode,m", po::value<char>(&mode_), "(required) Running mode (S (static), D (dynamic) or C (cleaning).")
            ("output_mesh_seq,o", po::value<std::string>(&output_folder_), "(required) Full path of the output mesh sequence.")
            ("images_sequences,i", po::value<std::string>(&images_sequences_),"(required) Path to images sequences")
            ("silhouettes_sequences,s", po::value<std::string>(&silhouettes_sequences_)," Path to silhouettes sequences")
            ("projection_matrices,p", po::value<std::string>(&projection_matrices_)," Path to projection matrices ")
            ("first_frame,f",po::value<int>(&first_frame_)->default_value(0),"First frame of the mesh sequence.")
            ("last_frame,l",po::value<int>(&last_frame_)->default_value(100),"Last frame of the mesh sequence.")
            ("number_centroids,k",po::value<unsigned int>(&nb_interest_centroids_)->default_value(50),"Number of interest points to be detected.")
            ("enable_cam_reconstruction,c",po::value<bool>(&reconstruct_cameras_)->default_value(false),"Enable Camera Reconstruction if visible.");
    return config;

}


/**
 * @brief OptionManager::advanced_options_description
 * @return
 */
po::options_description OptionManager::advanced_options_description(){
    po::options_description config("Advanced Options : these options are for experimented users");
    //config.add_options()
    //        ("probabilistic", po::value<bool>(&probabilistic)->default_value(true),"Run the algorithm in probabilistic mode");
    return config;

}


/**
 * @brief OptionManager::debug_options_description
 * @return
 */
po::options_description OptionManager::debug_options_description(){
    po::options_description config("Debug Options : Logger configuration");
    config.add_options()
            ("log_level",po::value<int>(&log_level_)->default_value(0),"Log Level")
            ("use_colors",po::value<bool>(&use_colors_)->default_value(true),"use colors to higlight the log levels : NODEBUG=-4, ALWAYS=-3, ERROR=-2, WARN=-1, NOTICE=0, DEBUG=1, DEBUG_EXTRA=2, DEBUG_MAX=3");
    return config;

}


/**
 * @brief OptionManager::optional_options_description
 * @return
 */
po::options_description OptionManager::optional_options_description(){
    po::options_description config("Facultative Options : usefull options to improve the processing");
    config.add_options()
            ("create_config_file",po::value<bool>(&create_config_file_)->default_value(false),"Optional : Save a file with the current configuration")
            ("backward_processing",po::value<bool>(&backward_processing_)->default_value(false),"backward processing");

    return config;

}


/**
 * @brief OptionManager::displayOptions
 */
void OptionManager::displayOptions(){
    log(ALWAYS) << "[Option Manager] : program options. " << endLog();
    log(ALWAYS) << "--------- " << endLog();
    log(ALWAYS) << " ................. running mode                  = " << mode_ << endLog();
    log(ALWAYS) << " ................. images sequences              = " << images_sequences_ << endLog();
    log(ALWAYS) << " ................. silhouettes sequences         = " << silhouettes_sequences_ << endLog();
    log(ALWAYS) << " ................. projection matrices           = " << projection_matrices_ << endLog();
    log(ALWAYS) << " ................. output folder                 = " << output_folder_ << endLog();
    log(ALWAYS) << " ................. first frame                   = " << first_frame_ << endLog();
    log(ALWAYS) << " ................. last frame                    = " << last_frame_ << endLog();
    log(ALWAYS) << " ................. backward processing           = " << backward_processing_ << endLog();
    log(ALWAYS) << " ................. number of centroids           = " << nb_interest_centroids_ << endLog();
    log(ALWAYS) << " ................. enable camera reconstruction  = " << reconstruct_cameras_ << endLog();
    log(ALWAYS) << std::endl;
}


/**
 * @brief OptionManager::parseOptions
 * @param argc
 * @param argv
 * @return
 */
int OptionManager::parseOptions(int argc, char **argv){

    po::options_description generic  = generic_options_description();
    po::options_description config   = config_options_description();
    po::options_description advanced = advanced_options_description();
    po::options_description debug    = debug_options_description();
    po::options_description optional = optional_options_description();

    po::options_description cmdline_options;
    cmdline_options.add(generic).add(config).add(advanced).add(optional).add(debug);

    po::options_description config_file_options;
    config_file_options.add(config).add(generic).add(advanced).add(optional).add(debug);

    po::positional_options_description p;
    p.add("configuration-file", -1);

    po::variables_map vm;

    po::store(po::command_line_parser(argc, argv).options(cmdline_options).positional(p).run(), vm);

    if (vm.count("help")) {
        log(ALWAYS) << "Optional arguments are not used if not set" << endLog();
        log(ALWAYS) << cmdline_options << endLog();
        return HELP;
    }

    po::notify(vm);

    //Check if a configuration file is passed as an argument.
    if (vm.count("configuration-file")){

        std::ifstream ifs(config_file_.c_str());

        if(!ifs){
            log(ERROR) << "[Option Manager] : invalid Configuration File" << endLog();
        }

        else{
            log(DEBUG) << "[Option Manager] : valid Configuration File" << endLog();
            po::store(po::parse_config_file(ifs, config_file_options), vm);
            po::notify(vm);
        }

    }

    /*There are N mandatory option :
     * - cameras_folders : camera folder name
      - output_mesh_seq : the output sequence name
      - images_sequences_ : image sequence relative to a camera
      If one of these options is not set, the program exits.
    */

    if (!vm.count("output_mesh_seq")){
        log(ERROR) << "[Option Manager] : missing output sequence name argument." << endLog();
        log(ERROR) << cmdline_options << endLog();
        throw std::runtime_error("[Fatal Error] : missing argument.");
    }

    if (!vm.count("images_sequences")){
        log(ERROR) << "[Option Manager] : missing images sequence argument." << endLog();
        log(ERROR) << cmdline_options << endLog();
        throw std::runtime_error("[Fatal Error] : missing argument.");
    }

    /*This option allows to create a file that will contain the current configuration and extra information.
    This file can be used as a configuration file to run the program again.
    */
    if (vm.count("createConfigFile_")){
        writeConfigurationFile();
    }

}

/**
 * @brief OptionManager::writeInConfigurationFile
 * @param message
 * @return
 */
bool OptionManager::writeInConfigurationFile(const std::string &message){
    std::string file_path = get_output_path(output_folder_) + "/sequence_configuration.txt";
    if(!exists(file_path)){
        createConfigurationFile(file_path);
    }

    std::ofstream config_file(file_path.c_str(),std::ofstream::app);
    if (!config_file.is_open()){
        log(ERROR) << "[Option Manager] : unable to open the sequence summary file." << endLog();
        return false;
    }

    config_file << "# " << message << std::endl;
    config_file.close();

}


