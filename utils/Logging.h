//
// Created by evaletov on 1/21/24.
//

#ifndef LOGGING_H
#define LOGGING_H

#include "Utilities.h"

// Define your log level here or via compiler flags
#define DEBUG_LEVEL_DEBUG 3
#define DEBUG_LEVEL_INFO 2
#define DEBUG_LEVEL_WARN 1
#define DEBUG_LEVEL_ERROR 0

// Compile-time debug level
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL DEBUG_LEVEL_INFO
#endif

// Optional flag for including MPI and thread info in logs
// Define this flag in your compile-time settings if you want to include this info
#define INCLUDE_MPI_THREAD_DEBUG

// Function to get current date-time
std::string currentDateTime();

std::string createLogPrefix();

// Macros for different levels of logging
#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
#define DEBUG_MSG std::cout << createLogPrefix() << "DEBUG: "
#else
#define DEBUG_MSG if(false) std::cout
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_INFO
#define INFO_MSG std::cout << createLogPrefix() << "INFO: "
#else
#define INFO_MSG if(false) std::cout
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_WARN
#define WARN_MSG std::cerr << createLogPrefix() << "WARN: "
#else
#define WARN_MSG if(false) std::cerr
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_ERROR
#define ERROR_MSG std::cerr << createLogPrefix() << "ERROR: "
#else
#define ERROR_MSG if(false) std::cerr
#endif

#endif // LOGGING_H
