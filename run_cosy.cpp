#include "run_cosy.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <boost/filesystem.hpp>
#include <mutex>

namespace fs = boost::filesystem; // Use Boost filesystem namespace

std::mutex mtx; // Mutex to make the writing process thread-safe

void write_to_log(const std::string &log_file, const std::vector<double> &parameter_values, double result) {
    std::unique_lock<std::mutex> lock(mtx);

    std::ofstream log(log_file, std::ios::app);
    if (log.is_open()) {
        for (const auto &value: parameter_values) {
            log << value << ",";
        }
        log << result << std::endl;
        log.close();
    } else {
        std::cerr << "Unable to open log file." << std::endl;
    }

    lock.unlock();
}

double read_result_from_file(const fs::path& output_file) {
    std::ifstream infile(output_file.string());
    double result;

    if (!infile) {
        std::cerr << "Warning: Cannot open the file for reading." << std::endl;
        std::vector<double> results(3, -10000.0);
        return -10000.0;
    }

    // Read the value in scientific notation format
    infile >> std::scientific >> result;

    // Check if the reading process failed
    if (infile.fail()) {
        std::cerr << "Error: Failed to read the objective function value correctly from the file." << std::endl;

        // Use "cat" to print the contents of the file
        std::string cat_command = "cat " + output_file.string();
        system(cat_command.c_str());

        // Abort the program execution
        std::abort();
    } else {
        result = -std::abs(result);
        std::cout << "Result: " << result << std::endl;
    }

    return result;
}

std::vector<double> run_cosy(const std::string &source_command, const std::string &program_directory,
                             const std::string &program_file, const std::string &config_file,
                             const std::vector<std::string> &dependency_files, const std::vector<std::string> &parameter_names,
                             const std::vector<double> &parameter_values,  const std::string &single_category_parameters,
                             int timeout_seconds) {
// Debug setting
    bool debug = false;

    // Create a temporary directory
    fs::path temp_dir = fs::temp_directory_path() / fs::unique_path();
    fs::create_directories(temp_dir);

    // Copy the program_file from the program_directory to the temporary directory
    fs::path program_file_path = fs::path(program_directory) / program_file;
    fs::path temp_program_file = temp_dir / program_file_path.filename();
    fs::copy_file(program_file_path, temp_program_file);

    // Copy the config_file from the program_directory to the temporary directory
    fs::path config_file_path = fs::path(program_directory) / config_file;
    fs::path temp_config_file = temp_dir / config_file_path.filename();
    if(fs::exists(config_file_path) && !fs::exists(temp_config_file)) {
        // Copy file only if config_file_path exists and temp_config_file does not exist
        fs::copy_file(config_file_path, temp_config_file);
    }

    // Make links to dependency_files from the program_directory to the temporary directory
    for (const auto &dep_file: dependency_files) {
        fs::path dep_file_path = fs::path(program_directory) / dep_file;
        fs::path temp_dep_file = temp_dir / dep_file_path.filename();
        fs::create_symlink(dep_file_path, temp_dep_file);
    }

    // Store a copy of the original file for comparison if debug mode is enabled
    fs::path original_temp_config_file;
    if (debug) {
        original_temp_config_file = temp_dir / ("original_" + config_file_path.filename().string());
        fs::copy_file(temp_config_file, original_temp_config_file);
    }

    // Update parameter values in the copied config_file using sed
    for (size_t i = 0; i < parameter_names.size(); ++i) {
        std::string sed_command = "sed -i 's/" + parameter_names[i] + " := .*/" + parameter_names[i] + " := " +
                                  std::to_string(parameter_values[i]) + " ;/g' " + temp_config_file.string();
        system(sed_command.c_str());
    }

    // If debug mode is enabled, show the diff and wait for user input
    if (debug) {
        std::string diff_command = "diff " + original_temp_config_file.string() + " " + temp_config_file.string();
        system(diff_command.c_str());
        std::cout << "Press enter to continue..." << std::endl;
        std::cin.get();
    }

    // Build the command string
    std::string command = "./cosy " + temp_program_file.filename().string();

    // Run the cosy command in the temporary directory
    command = "cd " + temp_dir.string() + " && " + command;
    int cosy_status = system(command.c_str());

    if (cosy_status != 0) {
        // Clean up the temporary directory
        fs::remove_all(temp_dir);

        // Return -10000 if cosy command fails
        std::vector<double> results(1, -10000.0);
        return results;
    }

// Read the result from the "fort.10000" file
    fs::path output_file1 = temp_dir / "fort.10000";
    double result1;
    result1 = read_result_from_file(output_file1);

// Read the result from the "fort.10001" file
    fs::path output_file2 = temp_dir / "fort.10001";
    double result2;
    result2 = read_result_from_file(output_file2);

    // Read the result from the "fort.10001" file
    fs::path output_file3 = temp_dir / "fort.10002";
    double result3;
    result3 = read_result_from_file(output_file3);

// Clean up the temporary directory
    fs::remove_all(temp_dir);

// Write parameter values and objective function value to log file if debug is true
    if (debug) {
        std::string log_file = "cosy_debug.log";
        write_to_log(log_file, parameter_values, result1);
        write_to_log(log_file, parameter_values, result2);
        write_to_log(log_file, parameter_values, result3);
    }

// Return the results
    std::vector<double> results{result1, result2, result3};
    return results;
}
