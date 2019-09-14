#include <iostream>
#include <sstream>

#include "Logger.h"

// *********************************************************************
//  Color management
// *********************************************************************

// unix terminal color definitions
const static std::string terminalColors[] =
{
    "\x1B[0m",  // NORMAL=0
    "\x1B[30m", // BLACK
    "\x1B[31m", // RED
    "\x1B[32m", // GREEN
    "\x1B[33m", // YELLOW
    "\x1B[34m", // BLUE
    "\x1B[35m", // MAGENTA
    "\x1B[36m", // CYAN
    "\x1B[37m", // WHITE
} ;

// level to color associations
const static LogColor levelColors[] =
{
    NORMAL, // ALWAYS=-3
    RED,    // ERROR=-2
    YELLOW, // WARN=-1
    NORMAL, // NOTICE=0
    BLUE,   // DEBUG=1
    BLUE,   // DEBUG_EXTRA=2
    BLUE,   // DEBUG_MAX=3
} ;

inline void setColor(const LogLevel level)
{
    std::cout << terminalColors[levelColors[(int)(level)-(int)ALWAYS]] ;
}


// *********************************************************************
//  Timing management
// *********************************************************************

const static std::string timeUnits[] =
{
    "s",
    "ms",
    "µs",
    "?",
} ;

// Return a human-friendly time difference string
std::string timeDiffStr(struct timeval& tv1, struct timeval& tv0, const TimeUnit unit)
{
    const int diffs  = (int)tv1.tv_sec-(int)tv0.tv_sec ;
    const int diffus = (int)tv1.tv_usec-(int)tv0.tv_usec ;
    TimeUnit unit_ = unit ;
    int t = 0 ;
    switch(unit)
    {
    case SEC:
    {
        t = diffs + diffus/1000000 ;
        break ;
    }
    case MSEC:
    {
        t = diffs*1000 + diffus/1000 ;
        break ;
    }
    case USEC:
    {
        t = diffs*1000000 + diffus ;
        break ;
    }
    case AUTO:
    {
        if(diffs>10)
        {
            unit_ = SEC ;
            t = diffs + diffus/1000000 ;
        }
        else if(diffus>10000)
        {
            unit_ = MSEC ;
            t = diffs*1000 + diffus/1000 ;
        }
        else
        {
            unit_ = USEC ;
            t = diffs*1000000 + diffus ;
        }
        break ;
    }
    }

    std::ostringstream oss ;
    oss << t <<  timeUnits[(int)unit_] ;
    return oss.str() ;
}

// *********************************************************************
//  Simple log class (singleton)
// *********************************************************************

static std::ostream nullStream(0) ;

// Constructor
Logger::Logger(const LogLevel coutLevel, const LogLevel fileLevel) :
    coutLevel(coutLevel), fileLevel(fileLevel), enableColor(true)
{
    dualStream.link(0,std::cout) ;
    initTime() ;
}

// Destructor
Logger::~Logger()
{
    close() ;
}

// To call when finished (todo: issue with destructor)
void Logger::close()
{
    clearColor() ;
    if(file.is_open())
        file.close() ;
}

// Return an instance of the Logger
//static Logger& Logger::instance(void)
//{
//    static Logger logger;
//    return logger ;
//}

// Open a log file
bool Logger::open(const std::string& fileName)
{
    if(file.is_open())
    {
        std::cout /*<< terminalColors[(int)RED]*/ << "Log file already open, switching to: " << fileName << std::endl ;
        file.close() ;
    }
    file.open(fileName.c_str()) ;
    dualStream.link(1,file) ;
    return file.is_open() ;
}

// Set the export log level
void Logger::setLogLevel(const LogLevel coutLevel, const LogLevel fileLevel)
{
    this->coutLevel = coutLevel ;
    this->fileLevel = fileLevel ;
}

// Return true if the message will be print out (cout of file) at the given level
bool Logger::willLog(const LogLevel level)
{
    return (level<=coutLevel) || (level<=this->fileLevel && file.is_open()) ;
}

// Add an entry in the logs
std::ostream& Logger::log(const LogLevel level)
{
#pragma omp critical
    {
        if(level<=this->coutLevel && enableColor){
            setColor(level) ;
        }

        return getStream(level) ;
    }
}

// Add an entry in the logs, force the color
std::ostream& Logger::log(const LogLevel level, const LogColor color)
{
#pragma omp critical
    {
        if(level<=this->coutLevel && enableColor)
            std::cout << terminalColors[(int)color] ;
        return getStream(level) ;
    }
}

// Initialise the relative time
void Logger::initTime()
{
    gettimeofday(&tv0,0) ;
    tv1 = tv0 ;
}

// Log the relative time
void Logger::logTime(const LogLevel level, const TimeUnit unit)
{
    struct timeval tv2 ;
    gettimeofday(&tv2,0) ;
    log(level) << "@ t=" << timeDiffStr(tv2,tv0,unit) << " dt=" << timeDiffStr(tv2,tv1,unit) << std::endl ;
    tv1 = tv2 ;
}

std::string Logger::endLog(){
    std::string out = "\x1B[0m\n";
    return out;
}

// Clear the color on std::cout
void Logger::clearColor()
{
    std::cout << terminalColors[(int)NORMAL] << std::endl;
}

// Return the stream to be used given the log level
std::ostream& Logger::getStream(const LogLevel level)
{
    if(level<=this->coutLevel)
    {
        if(level<=this->fileLevel && file.is_open())
            // both streams
            return dualStream ;
        else
            // std::cout only
            return std::cout ;
    }

    if(level<=this->fileLevel && file.is_open())
        // file only
        return file ;

    // no output stream
    return nullStream ;
}


