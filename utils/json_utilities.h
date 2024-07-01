// json_utilities.h
#ifndef JSON_UTILITIES_H
#define JSON_UTILITIES_H

#include <string>
#include <json.hpp>  // Ensure this path is correct based on your project setup

/**
 * Retrieves a value from a JSON object, supporting partition-specific overrides.
 * @param json The JSON object to query.
 * @param key The key whose value is to be retrieved.
 * @param this_partition The partition index to check for a specific override.
 * @param default_value The default value to return if the key is not found.
 * @return The value from the JSON object, or the default value if not found.
 */
    template<typename T>
    T get_json_value(const nlohmann::json& json, const std::string& key, int this_partition, const T& default_value) {
        std::string partition_key = "partition_" + std::to_string(this_partition);

        // First, try to get value from the specific partition
        if (json.contains(partition_key) && json[partition_key].is_object()) {
            if (json[partition_key].contains(key)) {
                return json[partition_key][key].get<T>();
            }
        }

        // If not found in the specific partition, fall back to the root level
        return json.value(key, default_value);
    }


bool validateOptimizeParameters(const nlohmann::json& json_data, const std::vector<std::string>& parameter_names) {
    bool error_occurred = false;
    bool parameter_names_printed = false;

    auto printParameterNames = [&parameter_names, &parameter_names_printed]() {
        if (!parameter_names_printed) {
            std::stringstream ss;
            ss << "Available parameter names: [";
            for (size_t i = 0; i < parameter_names.size(); ++i) {
                ss << "'" << parameter_names[i] << "'";
                if (i < parameter_names.size() - 1) ss << ", ";
            }
            ss << "]";
            INFO_MSG << ss.str() << std::endl;
            parameter_names_printed = true;
        }
    };

    for (auto it = json_data.begin(); it != json_data.end(); ++it) {
        if (it.key().find("partition_") == 0 && it.value().is_object()) {
            if (it.value().contains("optimize_parameters") && it.value()["optimize_parameters"].is_array()) {
                const auto& optimize_params = it.value()["optimize_parameters"];
                bool is_string = false;
                bool is_integer = false;

                for (const auto& item : optimize_params) {
                    if (item.is_string()) {
                        is_string = true;
                        if (std::find(parameter_names.begin(), parameter_names.end(), item.get<std::string>()) == parameter_names.end()) {
                            ERROR_MSG << "Error in " << it.key() << ": Parameter name '" << item.get<std::string>() << "' not found in parameter_names." << std::endl;
                            error_occurred = true;
                        }
                    } else if (item.is_number_integer()) {
                        is_integer = true;
                        int index = item.get<int>();
                        if (index < 0 || index >= parameter_names.size()) {
                            ERROR_MSG << "Error in " << it.key() << ": Index " << index << " is out of bounds." << std::endl;
                            error_occurred = true;
                        }
                    } else {
                        ERROR_MSG << "Error in " << it.key() << ": Invalid type in optimize_parameters. Use only strings or integers." << std::endl;
                        error_occurred = true;
                    }
                }

                if (is_string && is_integer) {
                    ERROR_MSG << "Error in " << it.key() << ": Mixed types in optimize_parameters. Use either all strings or all integers." << std::endl;
                    error_occurred = true;
                }
            }
        }
    }

    // Check global optimize_parameters if present
    if (json_data.contains("optimize_parameters") && json_data["optimize_parameters"].is_array()) {
        const auto& optimize_params = json_data["optimize_parameters"];
        bool is_string = false;
        bool is_integer = false;

        for (const auto& item : optimize_params) {
            if (item.is_string()) {
                is_string = true;
                if (std::find(parameter_names.begin(), parameter_names.end(), item.get<std::string>()) == parameter_names.end()) {
                    ERROR_MSG << "Error in global optimize_parameters: Parameter name '" << item.get<std::string>() << "' not found in parameter_names." << std::endl;
                    error_occurred = true;
                }
            } else if (item.is_number_integer()) {
                is_integer = true;
                int index = item.get<int>();
                if (index < 0 || index >= parameter_names.size()) {
                    ERROR_MSG << "Error in global optimize_parameters: Index " << index << " is out of bounds." << std::endl;
                    error_occurred = true;
                }
            } else {
                ERROR_MSG << "Error in global optimize_parameters: Invalid type. Use only strings or integers." << std::endl;
                error_occurred = true;
            }
        }

        if (is_string && is_integer) {
            ERROR_MSG << "Error in global optimize_parameters: Mixed types. Use either all strings or all integers." << std::endl;
            error_occurred = true;
        }
    }

    if (error_occurred) {
        printParameterNames();
    }

    return !error_occurred;
}

#endif // JSON_UTILITIES_H
