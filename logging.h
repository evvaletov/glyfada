//
// Created by evaletov on 1/21/24.
//

#ifndef LOGGING_H

#include <iostream>

// Define your log level here or via compiler flags
#define DEBUG_LEVEL_DEBUG 3
#define DEBUG_LEVEL_INFO 2
#define DEBUG_LEVEL_WARN 1
#define DEBUG_LEVEL_ERROR 0

// Compile-time debug level
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL DEBUG_LEVEL_INFO
#endif

// Macros for different levels of logging
#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
#define DEBUG_MSG std::cout << "DEBUG: "
#else
#define DEBUG_MSG if(false) std::cout
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_INFO
#define INFO_MSG std::cout << "INFO: "
#else
#define INFO_MSG if(false) std::cout
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_WARN
#define WARN_MSG std::cerr << "WARN: "
#else
#define WARN_MSG if(false) std::cerr
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_ERROR
#define ERROR_MSG std::cerr << "ERROR: "
#else
#define ERROR_MSG if(false) std::cerr
#endif

#define LOGGING_H

#endif //LOGGING_H
