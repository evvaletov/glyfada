//
// Created by evaletov on 1/28/24.
//

#include "Utilities.h"
#include <iomanip>
#include <sstream>
#include <chrono>
#include <mpi.h>
#include <thread>

void writeParametersToCsv(std::ofstream &file, const std::map<std::string, double> &parameters,
                          const std::vector<std::string> &parameter_names,
                          const std::string &single_category_parameters) {
    // Write parameters to the header of the CSV file
    for (const auto &[param, value]: parameters) {
        file << "# " << param << " = " << value << "\n";
    }
    file << "# " << single_category_parameters << "\n";

    // Write parameter names in one last header row
    file << "# ";
    for (size_t i = 0; i < parameter_names.size(); ++i) {
        file << parameter_names[i];
        if (i != parameter_names.size() - 1) {
            file << " ";
        }
    }
    file << "\n";
}

nlohmann::json read_json_file(const std::string &filename) {
    std::ifstream input_file(filename);
    if (!input_file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }
    nlohmann::json json_data;
    input_file >> json_data;
    return json_data;
}

std::vector<std::string> read_dependency_files(const nlohmann::json &json) {
    std::vector<std::string> dependency_files;
    if (json.contains("dependency_files") && !json.at("dependency_files").is_null() && json.at("dependency_files").is_array()) {
        for (const auto &file: json.at("dependency_files")) {
            dependency_files.push_back(file.get<std::string>());
        }
    }
    return dependency_files;
}

bool isNumber(const std::string &str) {
    return !str.empty() && str.find_first_not_of("-.0123456789") == std::string::npos;
}

std::string trim(const std::string &str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

std::vector<std::string> parseArguments(const std::string &s) {
    std::vector<std::string> arguments;
    bool inParentheses = false;
    std::string currentArg;

    for (char c: s) {
        if (c == '(') {
            inParentheses = true;
            currentArg += c;
        } else if (c == ')') {
            inParentheses = false;
            currentArg += c;
        } else if (c == ',' && !inParentheses) {
            arguments.push_back(currentArg);
            currentArg.clear();
        } else {
            currentArg += c;
        }
    }

    if (!currentArg.empty()) {
        arguments.push_back(currentArg);
    }

    return arguments;
}

std::string currentDateTime() {
    auto now = std::chrono::system_clock::now();
    auto now_as_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm now_tm;
    localtime_r(&now_as_time_t, &now_tm); // Use localtime_s on Windows

    std::ostringstream stream;
    stream << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
    stream << ':' << std::setfill('0') << std::setw(3) << now_ms.count();

    return stream.str();
}

int getCurrentMpiRank() {
    int initialized;
    MPI_Initialized(&initialized);
    if (initialized) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }
    return 0; // Return 0 if MPI is not initialized
}

std::string getCurrentThreadId() {
    auto id = std::this_thread::get_id();
    std::stringstream ss;
    ss << id;
    return ss.str();
}

std::string getFilenameSuffix() {
    std::string jobId;
    const char* envVar = nullptr;

    // Check for SLURM job ID
    envVar = std::getenv("SLURM_JOB_ID");
    if (envVar) {
        jobId = "SLURM_" + std::string(envVar);
    }

    // Check for other job management systems if SLURM job ID is not found
    if (jobId.empty()) {
        envVar = std::getenv("ClusterId"); // HTCondor
        if (envVar) jobId = "HTCondor_" + std::string(envVar);
    }

    if (jobId.empty()) {
        envVar = std::getenv("PBS_JOBID"); // PBS and Torque
        if (envVar) jobId = "PBS_" + std::string(envVar);
    }

    if (jobId.empty()) {
        envVar = std::getenv("JOB_ID"); // SGE/UGE and OpenLava
        if (envVar) jobId = "SGE_" + std::string(envVar); // 'SGE' used for both SGE/UGE and OpenLava
    }

    if (jobId.empty()) {
        envVar = std::getenv("LSB_JOBID"); // LSF
        if (envVar) jobId = "LSF_" + std::string(envVar);
    }

    if (jobId.empty()) {
        envVar = std::getenv("COBALT_JOBID"); // Cobalt
        if (envVar) jobId = "Cobalt_" + std::string(envVar);
    }

    // If a job ID is found, return it with prefix
    if (!jobId.empty()) {
        return jobId;
    } else {
        // Otherwise, use the current date and time
        std::time_t t = std::time(nullptr);
        std::tm *tm = std::localtime(&t);
        char dateStr[100];
        std::strftime(dateStr, sizeof(dateStr), "%Y-%m-%d_%H-%M-%S", tm);
        return std::string(dateStr);
    }
}

