/* *************************************************
 * Copyright (C) 2014 INRIA
 * *************************************************/
#ifndef FRAMEMANAGER_H
#define FRAMEMANAGER_H

#include <vector>
#include <algorithm>
#include "Logger.h"

/*! \file framemanager.h
    \brief Contains the frame manager class that generates frame sequences.
*/

/*! \class FrameManager
    \ingroup GMorpher
    \brief The frame manager class that generates a frame sequence.
*/
class FrameManager
{
public:
    FrameManager();
    FrameManager(int first_frame,int last_frame,bool backward_processing=false,int start_frame=0);

    //! Generate the frame sequence following the right parameters.
    /*! \param first_frame the frame with the lower index.
        \param last_frame the frame with the higher index.
        \param backward_processing invert the sequence order.
        \param start_frame the first frame of the sequence. It can be different than the first frame.
        \return true if the sequence is generated, false if not.
    */
    bool build_frame_manager(int first_frame,int last_frame,int start_frame, bool backward_processing);

    bool buid_frame_manager(int first_frame, int last_frame);
    //! Get the frame sequence vector.
    /*! \return a constant reference to the frame sequence vector.
    */
    inline const std::vector<int>& frame_sequence() const{
        return frame_sequence_;
    }

    /**
     * @brief setActiveFrameNumber
     * @param frame
     */
    inline void setActiveFrameNumber(int frame){
        if(frame>=*std::min_element(frame_sequence_.begin(),frame_sequence_.end())&&frame<=*std::max_element(frame_sequence_.begin(),frame_sequence_.end()))
            active_frame_ = frame;
        else log(ERROR)<<"[FrameManager::setActiveFrame()] Error: requested frame number outside boudaries"<<endLog();}

    /**
     * @brief getActiveFrameNumber
     * @return active_frame
     */
    inline int getActiveFrameNumber()const{return active_frame_;}

    //! Get a specific frame in the frame vector.
    /*! \param index the index of the target frame.
        \return the frame at the index position
    */
    inline int get_frame_number(unsigned int index) const{
        if (index < frame_sequence_.size())
            return frame_sequence_[index];
        return -1;
    }

    inline int get_first_frame() const{
        return frame_sequence_[0];
    }

    //! Get the number of frames.
    /*! \return the number of frames
    */
    inline unsigned int num_frames() const{
        return frame_sequence_.size();
    }

    //TBD: Add/Remove frame functions

private:
    //! the frame sequence vector.
    std::vector<int> frame_sequence_;
    int active_frame_;
};

#endif // FRAMEMANAGER_H
