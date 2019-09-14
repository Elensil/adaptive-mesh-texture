#include "../Include/framemanager.h"
#include "algorithm"


/**
 * @brief FrameManager::FrameManager
 */
FrameManager::FrameManager()
{
    active_frame_ = -1;
}


/**
 * @brief FrameManager::FrameManager
 * @param first_frame
 * @param last_frame
 * @param backward_processing
 * @param start_frame
 */
FrameManager::FrameManager(int first_frame, int last_frame ,bool backward_processing, int start_frame ){
    build_frame_manager(first_frame,last_frame,start_frame,backward_processing);
    active_frame_ = (backward_processing)?last_frame:first_frame;
}


/**
 * @brief FrameManager::build_frame_manager
 * @param first_frame
 * @param last_frame
 * @param start_frame
 * @param backward_processing
 * @return
 */
bool FrameManager::build_frame_manager(int first_frame, int last_frame, int start_frame,bool backward_processing){

    if (start_frame == 0) start_frame = first_frame;

    if (last_frame < first_frame){
        log(ERROR) << "[Frame Manager] : The first frame index should be lower than the last frame." << endLog();
        return false;
    }

    if (start_frame < first_frame){
        log(ERROR) << "[Frame Manager] : The start frame index should be lower than the first frame." << endLog();
        return false;
    }

    int delta_start = start_frame - first_frame;

    if (start_frame == first_frame){
        for (int i = start_frame; i <= last_frame; i++){
            frame_sequence_.push_back(i);
        }

        if (backward_processing) std::reverse(frame_sequence_.begin()+delta_start,frame_sequence_.end());
    }

    else{
        if (!backward_processing){
            for (int i = start_frame; i <= last_frame; i++){
                frame_sequence_.push_back(i);
            }
            for(int i = start_frame-1;i >= first_frame; i--){
                frame_sequence_.push_back(i);
            }
        }
        if (backward_processing){
            for(int i = start_frame;i >= first_frame; i--){
                frame_sequence_.push_back(i);
            }
            for (int i = start_frame+1; i <= last_frame; i++){
                frame_sequence_.push_back(i);
            }
        }

    }
    log(ALWAYS) << "[Frame Manager] : successfully generated the frame list for the following mesh sequence" << endLog();
    return true;
}
