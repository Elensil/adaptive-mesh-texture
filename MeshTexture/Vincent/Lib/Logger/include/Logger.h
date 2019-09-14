/* *************************************************
 * Copyright (C) 2014 INRIA
 * *************************************************/
#ifndef _LOGGER
#define _LOGGER

#include <fstream>
#include <sys/time.h>

#include "DualStream.h"


enum LogLevel { NODEBUG=-4, ALWAYS=-3, ERROR=-2, WARN=-1, NOTICE=0, DEBUG=1, DEBUG_EXTRA=2, DEBUG_MAX=3 } ;
enum LogColor { NORMAL=0, BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, NONE_FORCED } ;


// *********************************************************************
//  Timing management
// *********************************************************************

enum TimeUnit { SEC, MSEC, USEC, AUTO } ;

// Return a human-friendly time difference string
std::string timeDiffStr(struct timeval& tv1, struct timeval& tv0, const TimeUnit unit=SEC) ;


// *********************************************************************
//  Simple log class (singleton)
// *********************************************************************
class Logger
{
private:
    // Constructor
    Logger(const LogLevel coutLevel=NOTICE, const LogLevel fileLevel=NOTICE) ;

    // Not implemented - protection against copy
    Logger(const Logger&) ;
    Logger& operator=(const Logger&) ;

    //static Logger* _logger ; // actual instance

public:

    // Destructor
    ~Logger() ;

    // To call when finished (todo: issue with destructor)
    void close() ;

    // Return an instance of the Logger
    static Logger& instance(){
        static Logger logger;
        return logger;
    }

    // Open a log file
    bool open(const std::string& fileName) ;

    // Set the export log level
    void setLogLevel(const LogLevel coutLevel, const LogLevel fileLevel) ;

    // Return true if the message will be print out (cout of file) at the given level
    bool willLog(const LogLevel level) ;

    // Add an entry in the logs
    std::ostream& log(const LogLevel level=ALWAYS) ;

    // Add an entry in the logs, force the color
    std::ostream& log(const LogLevel level, const LogColor color) ;

    // Initialise the relative time (ms)
    void initTime() ;

    // Log the relative time
    void logTime(const LogLevel level, const TimeUnit unit=SEC) ;

    // Clear the color on std::cout
    void clearColor() ;

    std::string endLog();

    void enableColors(bool enableColor){
        this->enableColor = enableColor;
    }

    LogLevel coutLevel ; // maximum level to export to std::cout
    LogLevel fileLevel ; // maximum level to export to file



protected:

    // Return the stream to be used given the log level
    std::ostream& getStream(const LogLevel level) ;

    std::ofstream file ;
    DualStream dualStream ;

    struct timeval tv0, tv1 ;
    bool enableColor ; // whether to log in colours (std::cout only)
} ;


// *********************************************************************
//  Convenience functions
// *********************************************************************

// Convenience function to set the export log level
inline void setLogLevel(const LogLevel coutLevel, const LogLevel fileLevel=NOTICE)
{
    Logger::instance().setLogLevel(coutLevel,fileLevel) ;
}

inline void setLogLevel(const int coutLevel, const LogLevel fileLevel=NOTICE)
{
    Logger::instance().setLogLevel(static_cast<LogLevel>(coutLevel),fileLevel) ;
}
// Convenience function to enable colors
inline void enableColors(bool enable_colors)
{

    Logger::instance().enableColors(enable_colors);
}

// Convenience function to log a message
inline std::string endLog()
{

    return Logger::instance().endLog();
}

// Convenience function to log a message
inline std::ostream& log(const LogLevel level=ALWAYS)
{
    std::ostream& out = Logger::instance().log(level) ;
    return out;
}

// Convenience function to log a message with color
inline std::ostream& log(const LogLevel level, const LogColor color)
{
    return Logger::instance().log(level,color) ;
}

// Log the relative time
inline void logTime(const LogLevel level, const TimeUnit unit=SEC)
{
    Logger::instance().logTime(level,unit) ;
}

// Convenience function to see whether the message will print out at the given level
inline bool willLog(const LogLevel level)
{
    return Logger::instance().willLog(level) ;
}

#endif // #ifndef _LOGGER

