#include "run_g4bl.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <boost/filesystem.hpp>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace fs = boost::filesystem;

extern std::mutex mtx;
extern void write_to_log(const std::string &log_file, const std::vector<double> &parameter_values, double result);

std::vector<std::string> split_string(const std::string& str) {
    std::istringstream iss(str);
    std::vector<std::string> split(std::istream_iterator<std::string>{iss},
                                   std::istream_iterator<std::string>());
    return split;
}

void print_file_contents(const fs::path& file_path) {
    std::ifstream infile(file_path);
    if (!infile) {
        std::cerr << "Could not open file: " << file_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::cout << line << std::endl;
    }
    infile.close();
}



double process_output_file(const std::vector<fs::path>& output_files, const std::vector<std::string> &parameter_names, const std::vector<double> &parameter_values) {
    if (output_files.size() != 2) {
        std::cerr << "Error: process_output_file function requires exactly 2 output files." << std::endl;
        return -10000.0;
    }

    const fs::path& output_file_tail = output_files[0];
    const fs::path& output_file_awk = output_files[1];

    // Print the contents of the file
    // print_file_contents(output_file_tail);

    std::string tail_command = "tail -n 1 " + output_file_tail.string();
    FILE* pipe = popen(tail_command.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error: Failed to open the file with tail command." << std::endl;
        return -10000.0;
    }

    char buffer[256];
    std::string last_line;
    try {
        while(fgets(buffer, 256, pipe) != NULL) {
            last_line = buffer;
        }
    } catch(const std::exception& e) {
        std::cerr << "Error: Fatal exception occurred while running tail command. " << e.what() << std::endl;
        pclose(pipe);
        return -10000.0;
    }

    pclose(pipe);

    if(last_line.empty()) {
        std::cerr << "Error: File is empty, no lines found." << std::endl;
        return -10000.0;
    }

    // std::cout << "Last line: >>>" << last_line << "<<<" << std::endl;
    std::vector<std::string> profile = split_string(last_line);

    double sigmax;
    double sigmay;
    try {
        sigmax = std::stod(profile[3]);
        sigmay = std::stod(profile[5]);

        // Add these lines to print the sigmax and sigmay values
        std::cout << "Sigmax: " << sigmax << std::endl;
        std::cout << "Sigmay: " << sigmay << std::endl;
    } catch(const std::invalid_argument& e) {
        std::cerr << "Error: Failed to convert string to double. " << e.what() << std::endl;
        return -10000.0;
    }

    double meanx = 0;
    double meany = 0;

    std::string awk_command = "awk 'NR!=1 {r=sqrt(($1-" + std::to_string(meanx) + ")*($1-" + std::to_string(meanx) + ")+($2-" + std::to_string(meany) + ")*($2-" + std::to_string(meany) + ")); sum += (r<4)?1:1/(r-3); } END { print sum; }' " + output_file_awk.string();
    FILE* awk_pipe = popen(awk_command.c_str(), "r");
    if (!awk_pipe) {
        std::cerr << "Error: Failed to open the file with awk command." << std::endl;
        return -10000.0;
    }
    char awk_buffer[128];
    std::string result_str = "";
    try {
        while(!feof(awk_pipe)) {
            if(fgets(awk_buffer, 128, awk_pipe) != NULL)
                result_str += awk_buffer;
        }
    } catch(const std::exception& e) {
        std::cerr << "Error: Fatal exception occurred while running awk command. " << e.what() << std::endl;
        pclose(awk_pipe);
        return -10000.0;
    }
    pclose(awk_pipe);
    double result = std::stod(result_str);

    auto round_beam_coeff_iter = std::find(parameter_names.begin(), parameter_names.end(), "RoundBeamCoeff");
    if (round_beam_coeff_iter != parameter_names.end()) {
        size_t index = std::distance(parameter_names.begin(), round_beam_coeff_iter);
        double round_beam_coeff = parameter_values[index];
        if (round_beam_coeff == 1.0) {
            double resultcoeff = 0.5 + 0.5 * exp(-pow(sigmax - sigmay, 2) / 25);
            if (resultcoeff < 0.8) {
                result *= resultcoeff;
            }
        }
    }

    return result;
}



std::vector<double> run_g4bl(const std::string &source_command, const std::string &program_directory, const std::string &program_file, const std::string &config_file,
                             const std::vector<std::string> &dependency_files,
                             const std::vector<std::string> &parameter_names,
                             const std::vector<double> &parameter_values, const std::string &single_category_parameters, int n_objectives, int timeout_seconds) {
    // Currently, config_file is not implemented for g4bl
    bool debug = false;
    fs::path temp_dir = fs::temp_directory_path() / fs::unique_path();
    fs::create_directories(temp_dir);

    fs::path program_file_path = fs::path(program_directory) / program_file;
    fs::path temp_program_file = temp_dir / program_file_path.filename();
    fs::copy_file(program_file_path, temp_program_file);

    for (const auto &dep_file: dependency_files) {
        fs::path dep_file_path = fs::path(program_directory) / dep_file;
        fs::path temp_dep_file = temp_dir / dep_file_path.filename();
        fs::create_symlink(dep_file_path, temp_dep_file);
    }

    std::string parameters;
    for (size_t i = 0; i < parameter_names.size(); ++i) {
        parameters += " " + parameter_names[i] + "=" + std::to_string(parameter_values[i]);
    }

    auto evts_iter = std::find(parameter_names.begin(), parameter_names.end(), "OBJFEvalEvts");
    std::string evts;
    if (evts_iter != parameter_names.end()) {
        size_t index = std::distance(parameter_names.begin(), evts_iter);
        evts = " evts=" + std::to_string(parameter_values[index]);
    }

    std::string command = source_command + " && g4bl " + temp_program_file.filename().string() + parameters + " " + single_category_parameters + " decay=0" + evts;
    command = "cd " + temp_dir.string() + " && " + command;

    int g4bl_status;

    if (debug) {
        g4bl_status = system(command.c_str());
        std::cout << "Debug mode: " << command << std::endl;
    } else {
        g4bl_status = system(command.c_str());
    }

    if (g4bl_status != 0) {
        fs::remove_all(temp_dir);
        std::vector<double> results(1, -10000.0);
        return results;
    }

    fs::path output_file_1 = temp_dir / "profile3.txt";
    fs::path output_file_2 = temp_dir / "VD_Diagnostic_3.txt";
    std::vector<fs::path> output_files = {output_file_1, output_file_2};
    double result = process_output_file(output_files, parameter_names, parameter_values);

    if (debug) {
        std::cout << "Temporary directory: " << temp_dir << "\n";
        std::cout << "Objective function value: " << result << "\n";
        std::cout << "Program paused. Press any key to continue...\n";
        std::cin.get(); // Pauses the program until a key is pressed
    } else {
        fs::remove_all(temp_dir);
    }

    if (debug) {
        std::string log_file = "g4bl_debug.log";
        write_to_log(log_file, parameter_values, result);
    }

    std::vector<double> results{result};
    return results;
}