//
// Created by evaletov on 1/28/24.
//

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <json.hpp> // Make sure this include path is correct

// Utility function declarations
void writeParametersToCsv(std::ofstream &file,
                          const std::map<std::string, double> &parameters,
                          const std::vector<std::string> &parameter_names,
                          const std::string &single_category_parameters);

nlohmann::json read_json_file(const std::string &filename);

std::vector<std::string> read_dependency_files(const nlohmann::json &json);

bool isNumber(const std::string &str);

std::string trim(const std::string &str);

std::vector<std::string> parseArguments(const std::string &s);

#endif //UTILS_H
