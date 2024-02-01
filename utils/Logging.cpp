//
// Created by evaletov on 2/1/24.
//

#include "Logging.h"

// Helper function to create log prefix
std::string createLogPrefix() {
    std::ostringstream ss;
#ifdef INCLUDE_MPI_THREAD_INFO
    ss << "[MPI Rank: " << getCurrentMpiRank() << ", Thread ID: " << getCurrentThreadId() << "] ";
#endif
    ss << "[" << currentDateTime() << "] ";
    return ss.str();
}