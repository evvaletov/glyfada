//
// Created by evaletov on 1/28/24.
//

#include "Utilities.h"
#include <iomanip>
#include <sstream>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <thread>
#include <filesystem>

namespace fs = std::filesystem;

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


bool isNumber(const std::string& str) {
    // Copy the input string to modify it.
    std::string s = str;

    // Erase trailing whitespace
    s.erase(s.find_last_not_of(" \f\n\r\t\v") + 1);

    if (s.empty()) {
        return false;
    }

    std::istringstream iss(s);
    double num;
    iss >> num;

    // Check for a successful parse and no characters remaining.
    return iss.eof() && !iss.fail();
}

// Function to split a string by a delimiter
std::vector<std::string> splitString(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string parseDefaultValue(const std::string& arg) {
    size_t equalsIndex = arg.find("default_value=");
    if (equalsIndex == std::string::npos) {
        return "";
    }

    std::string defaultValue = arg.substr(equalsIndex + strlen("default_value="));

    // Remove leading and trailing whitespace
    defaultValue = trim(defaultValue);

    // Remove trailing closing parenthesis if present
    if (defaultValue.back() == ')') {
        defaultValue.pop_back();
    }

    return defaultValue;
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return (first == std::string::npos) ? "" : str.substr(first, (last - first + 1));
}

bool starts_with(const std::string& str, const std::string& prefix) {
    if (str.length() < prefix.length()) {
        return false;
    }
    return str.substr(0, prefix.length()) == prefix;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.substr(str.length() - suffix.length(), suffix.length()) == suffix;
}


std::vector<std::string> parseArguments(const std::string &s) {
        std::vector<std::string> arguments;
        bool inParentheses = false;
        bool inBrackets = false;
        bool inQuotes = false;
        char quoteChar = '\0';
        std::string currentArg;

        for (char c : s) {
            if (!inQuotes) {
                if (c == '(') {
                    inParentheses = true;
                } else if (c == ')') {
                    inParentheses = false;
                } else if (c == '[') {
                    inBrackets = true;
                } else if (c == ']') {
                    inBrackets = false;
                } else if ((c == '"' || c == '\'') && !inParentheses && !inBrackets) {
                    inQuotes = true;
                    quoteChar = c;
                } else if (c == ',' && !inParentheses && !inBrackets) {
                    arguments.push_back(trim(currentArg));
                    currentArg.clear();
                    continue;
                }
            } else if (c == quoteChar) {
                inQuotes = false;
            }

            currentArg += c;
        }

        if (!currentArg.empty()) {
            arguments.push_back(trim(currentArg));
        }

        // Clean up arguments
        for (std::string& arg : arguments) {
            // Remove unbalanced closing parentheses or brackets at the end
            while (!arg.empty() && (arg.back() == ')' || arg.back() == ']')) {
                char closing = arg.back();
                char opening = (closing == ')') ? '(' : '[';
                if (std::count(arg.begin(), arg.end(), opening) < std::count(arg.begin(), arg.end(), closing)) {
                    arg.pop_back();
                } else {
                    break;
                }
            }

            // Remove surrounding double parentheses or mixed parentheses/brackets
            if ((arg.size() >= 4) &&
                ((arg[0] == '(' && arg[1] == '(' && arg[arg.size()-2] == ')' && arg[arg.size()-1] == ')') ||
                 (arg[0] == '(' && arg[1] == '[' && arg[arg.size()-2] == ']' && arg[arg.size()-1] == ')'))) {
                arg = arg.substr(1, arg.size() - 2);
            }
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

std::vector<int> calculate_mpi_partitions(const std::vector<std::string>& partition_data, int num_mpi_ranks) {
    std::vector<int> partitions;
    int total_assigned = 0;

    for (const auto& part : partition_data) {
        if (part == "remaining") {
            // This will be calculated after knowing all other partitions
            partitions.push_back(0);
        } else if (part.back() == '%') {
            double percentage = std::stod(part.substr(0, part.size() - 1)) / 100.0;
            int count = static_cast<int>(std::round(percentage * num_mpi_ranks));
            partitions.push_back(count);
            total_assigned += count;
        } else {
            int count = std::stoi(part);
            partitions.push_back(count);
            total_assigned += count;
        }
    }

    // Assign remaining processes to the 'remaining' partition, if specified
    for (auto& part : partitions) {
        if (part == 0) { // the placeholder for 'remaining'
            part = num_mpi_ranks - total_assigned;
        }
    }

    return partitions;
}

// Function to determine this_partition based on MPI rank
int get_this_partition(const std::vector<int>& partitions, int rank) {
    int cumulative = 0;
    for (size_t i = 0; i < partitions.size(); ++i) {
        cumulative += partitions[i];
        if (rank < cumulative) {
            return i;
        }
    }
    return -1;
}

template<typename T>
T get_json_value(const nlohmann::json& json, const std::string& key, int this_partition, const T& default_value) {
    std::string partition_key = "partition_" + std::to_string(this_partition);

    // First try to get value from the specific partition
    if (json.contains(partition_key) && json[partition_key].is_object()) {
        if (json[partition_key].contains(key)) {
            return json[partition_key][key].get<T>();
        }
    }

    // If not found in the specific partition, fall back to the root level
    return json.value(key, default_value);
}