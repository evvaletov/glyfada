#ifndef RUN_G4BL_H
#define RUN_G4BL_H

#include <vector>
#include <string>
#include <mutex>

// Function prototype for run_cosy
std::vector<double> run_g4bl(const std::string& source_command, const std::string& program_directory,
                             const std::string& program_file, const std::string& config_file, const std::vector<std::string>& dependency_files,
                             const std::vector<std::string>& parameter_names,
                             const std::vector<double>& parameter_values, const std::string& single_category_parameters, int n_objectives, int timeout_seconds);

// Global mutex used for ensuring thread safety during file writing.
extern std::mutex mtx;

// Function to write parameter values and the result to a log file.
void write_to_log(const std::string &log_file, const std::vector<double> &parameter_values, double result);

#endif // RUN_G4BL_H

