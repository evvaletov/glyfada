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

#endif // JSON_UTILITIES_H
