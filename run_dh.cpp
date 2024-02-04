#include "run_dh.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <boost/filesystem.hpp>
#include <mutex>
#include <codecvt>
#include <locale>
#include <json.hpp>
#include "utils/Logging.h"

constexpr int TIMEOUT_SECONDS = 1800;  // 30 minutes

using json = nlohmann::json;

std::string escapeForShell(const std::string &jsonString) {
    std::string escapedString;
    for (char c: jsonString) {
        if (c == '"') {
            escapedString += '\\'; // Add a backslash to escape double quotes
        }
        escapedString += c;
    }
    return escapedString;
}

/**
 * Executes a shell command and parses its JSON output to extract objective function values.
 *
 * This function runs a specified shell command using popen and captures its stdout output.
 * The output is expected to be a JSON object, which is then parsed to extract objective function values.
 *
 * The function looks for specific keys in the JSON object: 'objective' and 'objective_n' (where n is a
 * sequence number starting from 1). If any 'objective_n' keys are found, the 'objective' key is disregarded.
 * The function also checks for 'objective_n' keys within sub-structures of the JSON object.
 *
 * Parameters:
 *   cmd (std::string): The shell command to be executed. This command should output a JSON object.
 *   n (int): The number of objectives expected in the JSON output. This guides the extraction of 'objective_n' keys.
 *
 * Returns:
 *   std::vector<double>: A vector of extracted objective function values. If the expected number of objectives
 *   (indicated by 'n') is not found, or if JSON parsing fails, the function returns a vector of size n with all
 *   elements set to -10000.0. If the output starts with 'F', indicating a failed evaluation, the function also
 *   returns a vector of size n with all elements set to -10000.0.
 *
 * Throws:
 *   std::runtime_error: If the popen call fails to execute the command.
 *
 * Note:
 *   The function includes detailed error logging. If JSON parsing fails, it outputs the error details.
 *   If the number of objectives found in the JSON output does not match 'n', it logs an error message
 *   along with the expected and found number of objectives, and the JSON output for further inspection.
 */
std::vector<double> exec(const std::string &cmd, int n) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    auto startTime = std::chrono::steady_clock::now();

    while (true) {
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();

        if (elapsedTime >= TIMEOUT_SECONDS) {
            // Timeout reached, terminate the process
            ERROR_MSG<< "Timeout " << TIMEOUT_SECONDS/60 << " min reached. Terminating the process." << std::endl;
            pclose(pipe.get());  // Terminate the process
            return std::vector<double>(n, -10000.0);
        }

        if (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        } else {
            // Process finished
            break;
        }
    }

    // Debug print to check the complete output
    //std::cerr << "Complete output: " << result << std::endl;
    //result = filterJSONString(result);
    //std::cerr << "Filtered JSON string: " << result << std::endl;

    // Check for errors and return an appropriate status code
    if (!result.empty()) {
        char firstChar = result[0];
        if (firstChar == 'F') {
            // Handle failed evaluation
            ERROR_MSG << "Failed evaluation (unknown reason) - " << result << std::endl;
            return std::vector<double>(n, -10000.0);
        } else if (firstChar == 'Q') {
            // Handle COSY INFINITY program QUIT
            ERROR_MSG << "Failed evaluation (COSY INFINITY program QUIT) - " << result << std::endl;
            return std::vector<double>(n, -10000.0);
        } else if (firstChar == 'S') {
            // Handle missing summary file
            ERROR_MSG << "Missing summary file - " << result << std::endl;
            return std::vector<double>(n, -10000.0);
        }
    }

    json j;
    bool parseSuccess = false;

    try {
        j = json::parse(result);
        parseSuccess = true;
    } catch (json::parse_error &e) {
        ERROR_MSG << "JSON Parsing Error: " << e.what() << "\n"
                << "Exception id: " << e.id << "\n"
                << "Byte position of error: " << e.byte << std::endl;
    }

    if (parseSuccess && j.is_object()) {
        std::vector<double> results;
        bool objectiveKeyIgnored = false;
        for (int i = 1; i <= n; ++i) {
            std::string key = "objective_" + std::to_string(i);
            if (j.contains(key) && j[key].is_number()) {
                results.push_back(j[key]);
                objectiveKeyIgnored = true;
            } else {
                // Search in sub-structures
                for (auto &el: j.items()) {
                    if (el.value().is_object() && el.value().contains(key) && el.value()[key].is_number()) {
                        results.push_back(el.value()[key]);
                        objectiveKeyIgnored = true;
                        break;
                    }
                }
            }
        }

        if (!objectiveKeyIgnored && j.contains("objective") && j["objective"].is_number()) {
            results.insert(results.begin(), j["objective"].get<double>()); // Explicitly convert to double
        }

        if (results.size() == n) {
            return results;
        } else {
            ERROR_MSG << "Error: Incorrect number of objectives in JSON output.\n"
                    << "Expected number of objectives: " << n << "\n"
                    << "Found number of objectives: " << results.size() << "\n"
                    << "JSON output: " << j.dump(4) << std::endl; // Pretty print JSON
            return std::vector<double>(n, -10000.0);
        }
    }

    // If output does not meet criteria, return error vector
    ERROR_MSG << "Invalid output: " << result << std::endl;
    return std::vector<double>(n, -10000.0);
}

std::vector<double> run_dh(const std::string &source_command, const std::string &program_directory,
                           const std::string &program_file, const std::string &config_file,
                           const std::vector<std::string> &dependency_files,
                           const std::vector<std::string> &parameter_names,
                           const std::vector<double> &parameter_values, const std::string &single_category_parameters) {
    bool debug = false;
    int max_retries = 3; // Maximum number of retries if the file is modified during evaluation
    int retry_count = 0;

    // Prepare the parameters as a JSON-like string
    std::ostringstream params_stream;
    params_stream << "{";
    for (size_t i = 0; i < parameter_names.size(); ++i) {
        params_stream << "\"" << parameter_names[i] << "\": ";
        if (parameter_names[i].front() == 'B' || parameter_names[i].front() == 'N') {
            params_stream << "\"" << parameter_values[i] << "\"";
        } else {
            params_stream << parameter_values[i];
        }
        if (i < parameter_names.size() - 1) params_stream << ", ";
    }
    params_stream << "}";

    std::string escapedJSONParams = escapeForShell(params_stream.str());
    std::string python_command = "/bin/zsh -c '";
    if (!source_command.empty()) {
        python_command += source_command + " && ";
    }
    python_command += "cd " + program_directory + " && ";
    python_command += "python " + program_directory + "/" + program_file +
            " \"" + escapedJSONParams + "\"'";

    //std::cout << "Constructed Python command: " << python_command << std::endl;

    // Execute the Python command and get the results
    std::vector<double> results;

    // Function to get the last modification time of the file
    auto getLastWriteTime = [&](const std::string &path) -> std::filesystem::file_time_type {
        return std::filesystem::last_write_time(path);
    };

    std::string fullPath = program_directory + "/" + program_file;
    std::filesystem::file_time_type lastWriteTimeBefore = getLastWriteTime(fullPath);


    // Function to perform the evaluation
    auto evaluate = [&]() {
        results = exec(python_command, 3);

        // Check the last modification time after the execution
        std::filesystem::file_time_type lastWriteTimeAfter = getLastWriteTime(fullPath);

        if (lastWriteTimeBefore != lastWriteTimeAfter) {
            throw std::runtime_error("Program file was modified during evaluation.");
        }
    };

    if (debug) {
        std::cout << "Debug mode: " << python_command << std::endl;
        results = exec(python_command, 3);

        // Print the entire results vector
        std::cout << "Results vector: [";
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << results[i];
            if (i < results.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        // Exit if debug is true
        exit(0);
    } else {
        try {
            do {
                try {
                    evaluate(); // Perform the evaluation
                    break; // Break the loop if evaluation succeeds without file modification
                } catch (const std::runtime_error &e) {
                    if (std::string(e.what()) == "FileModifiedDuringEvaluation") {
                        DEBUG_MSG << "Program file was modified during evaluation." << std::endl;
                        if (++retry_count >= max_retries) {
                            ERROR_MSG << "Program file is being modified continuously. Maximum retries reached. Aborting." << std::endl;
                            return std::vector<double>(3, -10000.0);
                        }
                        // Update last write time before retrying
                        lastWriteTimeBefore = getLastWriteTime(fullPath);
                    } else {
                        // Handle other std::runtime_error exceptions
                        throw; // Re-throw the exception to be caught by the outer catch
                    }
                }
            } while (retry_count < max_retries);

            // Print the entire results vector
            std::stringstream msgStream;
            msgStream << "Results vector: [";

            for (size_t i = 0; i < results.size(); ++i) {
                msgStream << results[i];
                if (i < results.size() - 1) {
                    msgStream << ", ";
                }
            }

            msgStream << "]";

            DEBUG_MSG << msgStream.str() << std::endl;
        } catch (const std::exception &e) {
            ERROR_MSG << "Error executing command: " << e.what() << std::endl;
            return std::vector<double>(3, -10000.0);
        }
    }

    return results;
}
