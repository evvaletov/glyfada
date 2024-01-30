//
// Created by evaletov on 1/28/24.
//

#include "Utilities.h"

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