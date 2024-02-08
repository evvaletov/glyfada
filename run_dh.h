#ifndef RUN_DH_H
#define RUN_DH_H

#include <vector>
#include <string>

constexpr int TIMEOUT_SECONDS = 60 * 20;

// Function prototype for run_cosy
std::vector<double> run_dh(const std::string& source_command, const std::string& program_directory,
                             const std::string& program_file, const std::string& config_file,
                             const std::vector<std::string>& dependency_files,
                             const std::vector<std::string>& parameter_names,
                             const std::vector<double>& parameter_values,
                             const std::string& single_category_parameters,
                             int timeout_seconds = TIMEOUT_SECONDS);

#endif // RUN_DH_H

