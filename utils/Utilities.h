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

std::vector<std::string> splitString(const std::string& s, char delimiter);

std::string parseDefaultValue(const std::string& arg);

std::string trim(const std::string &str);

std::vector<std::string> parseArguments(const std::string &s);

std::string currentDateTime();

int getCurrentMpiRank();

std::string getCurrentThreadId();

std::string getFilenameSuffix();

std::vector<int> calculate_mpi_partitions(const std::vector<std::string>& partition_data, int num_mpi_ranks);
int get_this_partition(const std::vector<int>& partitions, int rank);



#endif //UTILS_H
