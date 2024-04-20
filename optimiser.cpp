#include <stdio.h>
#include <moeo>
#include <es/eoRealInitBounded.h>
#include <es/eoRealOp.h>
#include <es/eoNormalMutation.h>

#include <mpi/eoMpi.h>
#include <eoSecondsElapsedTrackGenContinue.h>

#include <neighborhood/moRealVectorNeighbor.h>
#include <utils/eoParser.h>
#include <cstdlib>

#include <algorithm>
#include <stdexcept>
#include <chrono>

#include <vector>
#include <string>
#include <json.hpp>
#include <fstream>
#include <smp>

#include "mpi/implMpi.h"


#include <execinfo.h>
#include <cxxabi.h>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <es/eoSBXcross.h>

#include <filesystem>

#include <serial/eoSerial.h>

#include "utils/Logging.h"
#include "run_cosy.h"
#include "run_g4bl.h"
#include "run_dh.h"
//#include "mpi/SerializableBase.h"
#include "utils/Utilities.h"

#include "mpi/param.h"
#include "mpi/schema.h"

using namespace std;
using namespace paradiseo::smp;
using namespace eo::mpi;
//using namespace glyfada_parallel;
namespace fs = std::filesystem;

constexpr unsigned int N_OBJECTIVES = 3;
constexpr unsigned int N_TRAITS = 8;

typedef std::vector<double> (*EvalFunction)(
    const std::string &,
    const std::string &,
    const std::string &,
    const std::string &,
    const std::vector<std::string> &,
    const std::vector<std::string> &,
    const std::vector<double> &,
    const std::string &,
    int);

// the moeoObjectiveVectorTraits
template<int N_OBJECTIVES>
class GlyfadaMoeoObjectiveVectorTraits : public moeoObjectiveVectorTraits {
public:
    static bool minimizing(int i) {
        return false;
    }

    static bool maximizing(int i) {
        return true;
    }

    static unsigned int nObjectives() {
        return N_OBJECTIVES;
    }
};

// objective vector of real values
template<int N_OBJECTIVES>
using GlyfadaMoeoObjectiveVector = moeoRealObjectiveVector<GlyfadaMoeoObjectiveVectorTraits<N_OBJECTIVES> >;

// multi-objective evolving object for the System problem
template<int N_OBJECTIVES, int N_TRAITS>
class GlyfadaMoeoRealVector final : public moeoRealVector<GlyfadaMoeoObjectiveVector<N_OBJECTIVES> > {
public:
    GlyfadaMoeoRealVector() : moeoRealVector<GlyfadaMoeoObjectiveVector<N_OBJECTIVES> >(N_TRAITS) {
    }

    // Overload the equality operator with tolerance
    bool operator==(const GlyfadaMoeoRealVector& other) const {
        const double TOLERANCE = 1e-6; // Adjust the tolerance as needed

        if (this->size() != other.size()) {
            return false;
        }

        for (std::size_t i = 0; i < this->size(); ++i) {
            if (std::abs((*this)[i] - other[i]) > TOLERANCE) {
                return false;
            }
        }
        return true;
    }
};

// Global vectors to store all evaluated solutions and their corresponding generations
std::vector<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > allEvaluatedSolutions;
std::mutex mutexAllEvaluated; // A mutex for controlling access to the above vectors

// evaluation of objective functions
template<int N_OBJECTIVES, int N_TRAITS>
class SystemEval final : public moeoEvalFunc<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > {
public:
    SystemEval(EvalFunction evaluator, const std::string &source_command,
               const std::vector<std::string> &parameter_names,
               const std::string &single_category_parameters, const std::string &program_directory,
               const std::string &program_file, const std::string &config_file,
               const std::vector<std::string> &dependency_files,
               int islandId = -1, int timeout_seconds = 60 * 30, int evaluation_minimal_time = 15)
        : evaluator(evaluator), source_command(source_command), parameter_names(parameter_names),
          single_category_parameters(single_category_parameters),
          program_directory(program_directory), program_file(program_file), config_file(config_file),
          dependency_files(
              const_cast<vector<std::string> &>(dependency_files)), islandId(islandId), timeout_seconds(timeout_seconds),
          evaluation_minimal_time(evaluation_minimal_time), total_evaluation_time(0), evaluation_count(0) {
    }

    void operator()(GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> &_vec) override {
        if (_vec.invalidObjectiveVector()) {
            GlyfadaMoeoObjectiveVector<N_OBJECTIVES> objVec;
            vector<double> parameter_values(N_TRAITS); // use dimension() here instead of size()
            for (size_t i = 0; i < N_TRAITS; ++i) // use dimension() here instead of size()
            {
                parameter_values[i] = _vec[i];
                // std::cout << "parameter_values[" << i << "] = " << parameter_values[i] << std::endl;
            }

            auto start = std::chrono::steady_clock::now();

            // Call the evaluator function
            std::vector<double> results = evaluator(source_command, program_directory, program_file, config_file,
                                                    dependency_files, parameter_names,
                                                    parameter_values, single_category_parameters, timeout_seconds);

            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();


            // Update evaluation time tracking
            total_evaluation_time += elapsed;
            evaluation_count++;

            // Check if the average evaluation time is below the limit and we have enough evaluations
            if ((total_evaluation_time / evaluation_count < evaluation_minimal_time) && (evaluation_count > 10)) {
                ERROR_MSG << "Error: Average evaluation time below " << evaluation_minimal_time << " seconds after " << evaluation_count << " evaluations." << std::endl;
                exit(999);
            } else if (elapsed < evaluation_minimal_time) {
                std::stringstream warnMsgStream;
                warnMsgStream << "Warning: Evaluation completed in less than " << evaluation_minimal_time << " seconds. Parameters for this evaluation: ";
                for (size_t i = 0; i < N_TRAITS; ++i) {
                    warnMsgStream << parameter_names[i] << ": " << parameter_values[i];
                    if (i < N_TRAITS - 1) warnMsgStream << ", ";
                }
                WARN_MSG << warnMsgStream.str() << std::endl;
            }

            std::stringstream msgStream;
            int mpiRank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
            int ompThreadNum = omp_get_thread_num();
            int numThreads = omp_get_num_threads();
            int isNested = omp_get_nested();
            if (islandId != -1) {
                msgStream << "Island " << islandId << ", ";
            }
            msgStream << "OMP Thread " << ompThreadNum << ", ";
            msgStream << "MPI Rank " << mpiRank << ", Results vector: ";
            for (size_t i = 0; i < results.size(); ++i) {
                msgStream << results[i];
                if (i < results.size() - 1) {
                    msgStream << ", ";
                }
            }

            DEBUG_MSG << msgStream.str() << std::endl;

            DEBUG_MSG << "Current Thread: " << ompThreadNum << ", Total Threads: " << numThreads
                    << ", Nested Parallelism: " << (isNested ? "Enabled" : "Disabled") << std::endl;

            // Set the objectives based on the results from run_g4beamline or run_cosy
            for (size_t i = 0; i < N_OBJECTIVES; ++i) {
                objVec[i] = results[i];
            }

            _vec.objectiveVector(objVec);

            // Critical section begins
            mutexAllEvaluated.lock();
            allEvaluatedSolutions.push_back(_vec);
            mutexAllEvaluated.unlock();
            // Critical section ends
        }
    }

private:
    EvalFunction evaluator;
    std::string source_command;
    std::vector<std::string> parameter_names;
    std::string single_category_parameters;
    std::string program_directory;
    std::string program_file;
    std::string config_file;
    std::vector<std::string> &dependency_files;
    int islandId;
    int timeout_seconds;
    int evaluation_minimal_time;
    long long total_evaluation_time;
    int evaluation_count;
};

// TODO DONE: check if in eoNormalVecMutation the sigma argument scaled by the range: yes
// TODO DONE: implement hybrid island method
// TODO DONE: implement MPI parallelization
// TODO: track statistics for the quality and speed of optimization
// TODO: implement hyperparameter optimization
// TODO: manage loads between islands if running on same MPI rank
// TODO: implement the possibility of using a different number of OMP threads for each island
// TODO DONE: merge regular and island-based codes (set mode from parameter file)
// TODO: implement a Ctrl-C handler that gracefully shuts down the optimisation
// TODO: DeepHyper: add optional hyperparameters
// TODO: implement LS like NSGAII
// TODO: implement checkpointing, file <-> redis saves
// TODO: revise so that redis doesn't fail if get or put is done before auth
// TODO: make the minimum number of scaling attempts non-changeable in ULS
// TODO: implement auto-adjustment of bounds
// TODO: LS -- explore different islands of solutions
// TODO: EO -- explore different islands of solutions
// TODO: LS -- if f looks like const in the neighborhood, stop exploring at that point or increase the delta r
// TODO: LS -- gradually expand LS region if not successful in 2-3 generations
// TODO: LS -- implement fast single-objective search in regions dominated by one objective
// TODO: LS -- implement priorities for objectives (first focus on optimising one, then another, etc.)
// TODO: LS -- implement JSON parameters
// TODO: LS -- if no suitable individual, then random search instead of LS
// TODO: LS -- use different exploreres based on conditions.
// TODO: LS -- uniform exploration using cosines
// TODO: LS -- inertia
// TODO: LS -- ADAM optimiser


int main(int argc, char *argv[]) {
    std::cout << "--------------------------------------------------------\n";
    std::cout << " Glyfada - Island Model Optimiser\n";
    std::cout << " Based on the Paradiseo framework\n";
    std::cout << " Author: Eremey Valetov\n";
    std::cout << " Build date: " << __DATE__ << " " << __TIME__ << "\n";
    std::cout << "--------------------------------------------------------\n";

    eo::mpi::Node::init(argc, argv);
    //loadRMCParameters (argc, argv);
    int rank;
    bmpi::communicator &comm = eo::mpi::Node::comm();
    rank = comm.rank();
    int num_mpi_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_mpi_ranks);
    INFO_MSG << "MPI rank: " << rank << endl;
    INFO_MSG << "MPI ranks: " << num_mpi_ranks << endl;

    std::string redisIP;
    std::string redisPassword = "";
    int redisPort = 15559;
    int redisMaxPopSize = 0;
    std::string redisJobID = "";
    std::string redisInitJobID = "";
    bool redisWriteAll = false;
    bool redisUseInitJob = false;
    bool loadedFromRedisJson = false;


    /*if (my_node != nullptr) {
        // Print MPI Rank at the beginning of each line
        std::cout << "[MPI Rank: " << my_node->rk << "] ";

        // Print information about whether the current node is a scheduler
        std::cout << "Is Scheduler Node: " << (isScheduleNode() ? "Yes" : "No") << std::endl;

        // Print the runner IDs associated with the current MPI rank
        for (RUNNER_ID runner_id : my_node->id_run) {
            std::cout << "[MPI Rank: " << my_node->rk << "] ";
            std::cout << "Runner ID: " << runner_id << std::endl;
        }
    } else {
        std::cerr << "Error: my_node is not initialized." << std::endl;
    }*/


    // Get the current time point
    auto now = std::chrono::high_resolution_clock::now();
    // Convert it to a duration since the epoch
    auto duration = now.time_since_epoch();
    // Narrow it down to microseconds
    auto timeSeed = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    // Get the process ID
    auto pid = getpid();
    std::random_device rd;
    unsigned seed2 = rd();
    // Combine time seed, PID, rank, and random seed device input
    std::size_t hashSeed = std::hash<long long>()(timeSeed) ^ std::hash<int>()(pid) ^ std::hash<int>()(rank) ^ std::hash<int>()(seed2);
    // Seed the random number generator
    eo::rng.reseed(hashSeed);
    INFO_MSG << "Seed: " << hashSeed << std::endl;
    //eo::rng.reseed(2 - rank);

    eoParser parser(argc, argv, "Glyfada"); // for user-parameter reading
    //eoState state; // to keep all things allocated

    eoValueParam<std::string> interactiveParam("", "interactive", "Interactive mode override", 'i', false);
    parser.processParam(interactiveParam, "Execution");

    // Debug information to check the value obtained from the command line
    if (parser.isItThere(interactiveParam)) {
        std::string interactiveCmdValue = interactiveParam.value();
        DEBUG_MSG << "Interactive mode parameter from command line: " << interactiveCmdValue << std::endl;
    } else {
        DEBUG_MSG << "Interactive mode parameter not provided in command line, default value will be used." <<
                std::endl;
    }

    // Define a command-line parameter for configuration file
    eoValueParam<std::string> configFileParam("", "config", "Path to configuration file", 'c', true);
    parser.processParam(configFileParam, "General");
    //make_parallel( parser );
    make_help(parser);

    // Read the configuration file parameter
    std::string config_filename;
    if (parser.isItThere(configFileParam)) {
        config_filename = parser.valueOf<std::string>("config");
    } else {
        ERROR_MSG << "Usage: " << argv[0] << " --config=<config_file_path>\n";
        return EXIT_FAILURE;
    }

    // Read the JSON file
    nlohmann::json json_data = read_json_file(config_filename);

    std::string program_file = json_data["program_file"].get<std::string>();
    std::string config_file = json_data["config_file"].get<std::string>();
    std::string program_directory = json_data.value("program_directory", "cwd");
    // Check if program_directory is set to 'cwd', then get the current working directory
    if (program_directory == "cwd") {
        program_directory = fs::current_path().string();
    } else {
        if (fs::current_path().string() != program_directory) {
            WARN_MSG << "Warning: The specified program directory '" << program_directory
                      << "' is not the same as the current working directory." << std::endl;
        }
    }
    // Check if the program directory exists
    if (!fs::exists(program_directory)) {
        ERROR_MSG << "Program directory '" << program_directory << "' does not exist." << std::endl;
        return EXIT_FAILURE;
    }
    std::vector<std::string> dependency_files = read_dependency_files(json_data);

    // Read parameters from JSON file
    unsigned int POP_SIZE = json_data.value("popSize", 200);
    unsigned int MAX_GEN = json_data.value("maxGen", 50);
    unsigned int MAX_TIME = json_data.value("maxTime", 100);  // New parameter
    std::string RUN_LIMIT_TYPE = json_data.value("runLimitType", "maxGen");  // New parameter
    unsigned int MIGRATION_PERIOD = json_data.value("migrationPeriod", 1);  // New parameter
    unsigned int TOURNAMENT_SIZE = json_data.value("tournamentSize", 15);  // New parameter
    unsigned int SELECTION_NUMBER = json_data.value("selectionNumber", 1);  // New parameter
    double M_EPSILON = json_data.value("mutEpsilon", 0.01);
    double P_CROSS = json_data.value("pCross", 0.25);
    double P_MUT = json_data.value("pMut", 0.35);
    double ETA_C = json_data.value("eta_c", 30.0);
    double SIGMA = json_data.value("sigma", 0.1);
    double P_CHANGE = json_data.value("p_change", 1.0);
    std::string EVALUATOR = json_data.value("evaluator", "cosy");
    std::string ALGORITHM = json_data.value("algorithm", "auto");
    unsigned int TIMEOUT_MINUTES = json_data.value("timeout_minutes", 20);
    unsigned int EVALUATION_MINIMAL_TIME = json_data.value("timein_seconds", 15);
    std::string SOURCE_COMMAND = json_data.value("source_command", "");
    bool PRINT_ALL_RESULTS = json_data.value("print_all_results", false);

    bool interactive_mode = json_data.value("interactive_mode", false);
    // modes: multistart, homogeneous
    std::string mode = json_data.value("mode", "multistart");
    INFO_MSG << "Operation mode set to: " << mode << std::endl;

    std::string ALGORITHM_STR = ALGORITHM;
    if (ALGORITHM=="hybrid1") {
        if (rank==0) {
            ALGORITHM = "ULS";
            ALGORITHM_STR = "hybrid1, rank " + to_string(rank) + " -> ULS";
        } else {
            ALGORITHM = "NSGAII";
            ALGORITHM_STR = "hybrid1, rank " + to_string(rank) + " -> NSGAII";
        }
    }

    if (mode == "redis") {
        try {
            // Attempt to load Redis configuration from redis.json
            nlohmann::json redis_config = read_json_file("redis.json");

            if (!redis_config.empty()) {
                loadedFromRedisJson = true;
            }

            if (redis_config.contains("redis_ip")) { redisIP = redis_config["redis_ip"]; }
            if (redis_config.contains("redis_port")) { redisPort = redis_config["redis_port"]; }
            if (redis_config.contains("redis_password")) { redisPassword = redis_config["redis_password"]; }
            if (redis_config.contains("redis_job_ID")) { redisJobID = redis_config["redis_job_ID"]; }
            if (redis_config.contains("redis_init_job_ID")) { redisInitJobID = redis_config["redis_init_job_ID"]; }
            if (redis_config.contains("redis_max_pop_size")) { redisMaxPopSize = redis_config["redis_max_pop_size"].get<int>(); }
            if (redis_config.contains("redis_write_all")) { redisWriteAll = redis_config["redis_write_all"].get<bool>(); }
            if (redis_config.contains("redis_use_init_job")) { redisUseInitJob = redis_config["redis_use_init_job"].get<bool>(); }
        } catch (std::exception& e) {
            std::cerr << "Notice: redis.json not found or contains errors; attempting to load from main configuration. Error: " << e.what() << std::endl;
        }

        auto overrideConfig = [&loadedFromRedisJson](auto oldValue, const auto& newValue, const std::string& key) {
            if (loadedFromRedisJson && oldValue != newValue) {
                std::cerr << "Warning: Overriding " << key << " from main configuration." << std::endl;
            }
            return newValue; // Return by value
        };

        // Override with values from json_data if present, with warnings for overrides
        if (json_data.contains("redis_ip")) { redisIP = overrideConfig(redisIP, json_data["redis_ip"], "redis_ip"); }
        if (json_data.contains("redis_port")) { redisPort = overrideConfig(redisPort, json_data["redis_port"], "redis_port"); }
        if (json_data.contains("redis_password")) { redisPassword = overrideConfig(redisPassword, json_data["redis_password"], "redis_password"); }
        if (json_data.contains("redis_job_ID")) { redisJobID = overrideConfig(redisJobID, json_data["redis_job_ID"], "redis_job_ID"); }
        if (json_data.contains("redis_init_job_ID")) { redisInitJobID = overrideConfig(redisInitJobID, json_data["redis_init_job_ID"], "redis_init_job_ID"); }
        if (json_data.contains("redis_max_pop_size")) { redisMaxPopSize = overrideConfig(redisMaxPopSize, json_data["redis_max_pop_size"].get<int>(), "redis_max_pop_size"); }
        if (json_data.contains("redis_write_all")) { redisWriteAll = overrideConfig(redisWriteAll, json_data["redis_write_all"].get<bool>(), "redis_write_all"); }
        if (json_data.contains("redis_use_init_job")) { redisUseInitJob = overrideConfig(redisUseInitJob, json_data["redis_use_init_job"].get<bool>(), "redis_use_init_job"); }

        // Adjust redisUseInitJob based on redisInitJobID being empty
        redisUseInitJob = !redisInitJobID.empty() && redisUseInitJob;

        // Check if redisJobID is set
        if (redisJobID.empty()) {
            std::cerr << "Error: Redis job_ID is not set in configuration." << std::endl;
            return EXIT_FAILURE;
        }

        INFO_MSG << "Redis jobID = " << redisJobID << std::endl;
        INFO_MSG << "Redis init jobID = " << (redisInitJobID.empty() ? "Not set" : redisInitJobID) << std::endl;
        INFO_MSG << "Redis use init job: " << (redisUseInitJob ? "Yes" : "No") << std::endl;
        INFO_MSG << "Redis max population size = " << redisMaxPopSize << std::endl;
        INFO_MSG << "Redis write all: " << (redisWriteAll ? "Enabled" : "Disabled") << std::endl;
    }

    // Check if interactive mode parameter was provided in command line
    if (parser.isItThere(interactiveParam)) {
        std::string interactiveCmd = parser.valueOf<std::string>("interactive");
        if (interactiveCmd == "true" || interactiveCmd == "on") {
            interactive_mode = true;
            DEBUG_MSG << "Interactive mode set to ON based on command line argument." << std::endl;
        } else if (interactiveCmd == "false" || interactiveCmd == "off") {
            interactive_mode = false;
            DEBUG_MSG << "Interactive mode set to OFF based on command line argument." << std::endl;
        }
    }

    // Initialize default values
    std::string dh_model_filename = "model.py";
    bool parse_dh_model = false;

    // Check if json_data has the parameter and set the variables accordingly
    auto dh_model_param = json_data.find("get_parameters_from_dh_model");
    if (dh_model_param != json_data.end()) {
        if (dh_model_param->is_boolean() && dh_model_param->get<bool>()) {
            parse_dh_model = true;
        } else if (dh_model_param->is_string()) {
            dh_model_filename = dh_model_param->get<std::string>();
            parse_dh_model = true;
        }
    }

    fs::path full_dh_model_path = fs::path(program_directory) / dh_model_filename;

    int num_threads = json_data.value("omp_num_threads", 4); // Set this to the number of threads you want OpenMP to use
    eo::parallel.setNumThreads(num_threads);
    omp_set_num_threads(eo::parallel.nthreads());

    // Get the maximum number of threads that could be used
    int max_threads = omp_get_max_threads();
    INFO_MSG << "OpenMP max threads: " << max_threads << std::endl;

    // Check the number of MPI ranks and maximal OpenMP threads
    if (comm.size() >= 24 && max_threads < 12) {
        ERROR_MSG << "Error: Number of MPI ranks is >= 24 and maximal OpenMP threads is below 12." << std::endl;
        return EXIT_FAILURE;
    }

    // First, extract the parameters from the JSON data
    bool use_default_values = json_data.value("use_default_values", true);

    // Create empty vectors to hold the parameter names and their bounds
    std::vector<std::string> parameter_names;
    std::vector<double> min_values;
    std::vector<double> max_values;
    std::vector<double> default_values;
    std::vector<std::vector<double>> default_values_vector;

    // Create a string to hold the single-category parameters
    std::string single_category_parameters;

    if (!parse_dh_model) {
        std::vector<nlohmann::json> search_space = json_data["parameters"].get<std::vector<nlohmann::json> >();
        // Iterate through each parameter
        for (const auto &param: search_space) {
            // Check the parameter type
            std::string param_type = param["type"].get<std::string>();

            if (param_type == "continuous") {
                // If it's a continuous parameter, add its name, min_value, and max_value to the appropriate vectors
                parameter_names.push_back(param["name"].get<std::string>());
                min_values.push_back(param["min_value"].get<double>());
                max_values.push_back(param["max_value"].get<double>());
                // Only add default value if use_default_values is true
                if (use_default_values) {
                    default_values.push_back(param["default_value"].get<double>());
                }
            } else if (param_type == "categorical") {
                std::vector<std::string> categories = param["values"].get<std::vector<std::string>>();

                if (categories.size() == 1) {
                    // For a single-category parameter
                    single_category_parameters += param["name"].get<std::string>() + "=" + categories[0] + " ";

                    // Check for a mismatch between the default value and the category
                    if (use_default_values && param.contains("default_value")) {
                        std::string default_value = param["default_value"].get<std::string>();
                        if (default_value != categories[0]) {
                            WARN_MSG << "Warning: Default value '" << default_value
                                      << "' does not match the sole category '" << categories[0]
                                      << "' for parameter '" << param["name"].get<std::string>() << "'." << std::endl;
                        }
                    }
                } else {
                    // For multiple categories, use the default value if available
                    if (param.contains("default_value")) {
                        std::string default_value = param["default_value"].get<std::string>();
                        single_category_parameters += param["name"].get<std::string>() + "=" + default_value + " ";
                        WARN_MSG << "Warning: Categorical parameters with more than one category are set to their defaults." << std::endl;
                    } else {
                        ERROR_MSG << "Warning: Categorical parameters with more than one category are set to their defaults." << std::endl;
                        return EXIT_FAILURE;
                    }
                }
            }
        }
    } else {
        // Open the Python file
        std::ifstream file(full_dh_model_path);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << dh_model_filename << std::endl;
            return 1;
        }

        std::string line;
        while (std::getline(file, line)) {
            //std::cout << "Reading line: " << line << std::endl; // Debug print

            // Skip comment lines
            size_t firstChar = line.find_first_not_of(" \t");
            if (firstChar != std::string::npos && line[firstChar] == '#') {
                //std::cout << "Skipping comment line." << std::endl;
                continue;
            }

            size_t found = line.find("problem.add_hyperparameter");
            if (found != std::string::npos) {
                std::string params = line.substr(found + 27); // Extract parameters substring
                //std::cout << "Parameters substring: " << params << std::endl; // Debug print

                std::vector<std::string> args = parseArguments(params);
                //std::cout << "Parsed arguments count: " << args.size() << std::endl; // Debug print
                //for (const auto& arg : args) {
                //    std::cout << "Argument: " << arg << std::endl; // Debug print
                //}

                if (args.size() >= 2) {
                    size_t openBracket = args[0].find('[');
                    size_t closeBracket = args[0].find(']');
                    //INFO_MSG << "openBracket = " << openBracket << std::endl;
                    //INFO_MSG << "closeBracket = " << closeBracket << std::endl;
                    if (openBracket != std::string::npos && openBracket < closeBracket) {
                        // Confirmed args[0] is likely a categorical parameter specification
                        // Now, determine if it's a single-category categorical parameter
                        size_t commaIndex = args[0].find(',', openBracket);
                        bool hasComma = commaIndex != std::string::npos && commaIndex < closeBracket;
                        //INFO_MSG << "args[0] = " << args[0] << std::endl;
                        //if (args.size()>=2) INFO_MSG << "args[1] = " << args[1] << std::endl;
                        //if (args.size()>=3) INFO_MSG << "args[2] = " << args[2] << std::endl;
                        //INFO_MSG << "hasComma = " << hasComma << std::endl;

                        // Extract parameter name
                        std::string name = args[1].substr(args[1].find_first_of("\"\'") + 1,
                                                          args[1].find_last_of("\"\'") - args[1].find_first_of("\"\'") - 1);

                        if (hasComma) {
                            // Check for the presence of a default_value in the arguments
                            std::string defaultValue;
                            bool foundDefaultValue = false;
                            for (const auto& arg : args) {
                                size_t equalsIndex = arg.find("default_value=");
                                if (equalsIndex != std::string::npos) {
                                    defaultValue = arg.substr(equalsIndex + strlen("default_value="));
                                    foundDefaultValue = true;
                                    break;
                                }
                            }

                            if (!foundDefaultValue) {
                                ERROR_MSG << "Error: Categorical parameter with multiple categories and no default value provided." << std::endl;
                                exit(EXIT_FAILURE);
                            } else {
                                // Treat this as a single-category categorical parameter, using the default value as the category
                                single_category_parameters += name + "=" + defaultValue + " ";
                            }
                        } else {
                            // It's a single-category parameter without a comma, process normally
                            std::string category = args[0].substr(openBracket + 1, closeBracket - openBracket - 1);
                            single_category_parameters += name + "=" + category + " ";
                        }
                    }  else {
                        if (args.size() < 2 || args.size() > 3) continue; // Ensure correct number of arguments

                        std::string range = args[0];
                        //std::cout << "Range string: " << range << std::endl; // Debug print

                        //INFO_MSG << "args[0] = " << args[0] << std::endl;
                        //if (args.size()>=2) INFO_MSG << "args[1] = " << args[1] << std::endl;
                        //if (args.size()>=3) INFO_MSG << "args[2] = " << args[2] << std::endl;

                        size_t openParen = range.find("(");
                        size_t closeParen = range.find(")");
                        if (openParen == std::string::npos || closeParen == std::string::npos) continue;
                        // Ensure parentheses are found

                        std::string minMax = range.substr(openParen + 1, closeParen - openParen - 1);
                        std::istringstream rangeStream(minMax);
                        std::string min_str_raw, max_str_raw;
                        std::getline(rangeStream, min_str_raw, ',');
                        std::getline(rangeStream, max_str_raw);

                        std::string min_str = trim(min_str_raw);
                        std::string max_str = trim(max_str_raw);

                        //std::cout << "Trimmed Min string: " << min_str << ", Trimmed Max string: " << max_str << std::endl; // Debug print

                        if (!isNumber(min_str)) {
                            ERROR_MSG << "Non-numeric lower bound encountered: " << min_str << std::endl;
                            return EXIT_FAILURE;
                        }
                        if (!isNumber(max_str)) {
                            ERROR_MSG << "Non-numeric upper bound encountered: " << max_str << std::endl;
                            return EXIT_FAILURE;
                        }
                        double min_val = std::stod(min_str);
                        double max_val = std::stod(max_str);
                        //std::cout << "Parsed Min: " << min_val << ", Max: " << max_val << std::endl; // Debug print


                        min_values.push_back(min_val);
                        max_values.push_back(max_val);

                        std::string name = args[1];
                        //std::cout << "Name argument: " << name << std::endl; // Debug print

                        size_t nameStart = name.find_first_of("\"\'");
                        size_t nameEnd = name.find_last_of("\"\'");
                        if (nameStart == std::string::npos || nameEnd == std::string::npos || nameStart == nameEnd)
                            continue;

                        parameter_names.push_back(name.substr(nameStart + 1, nameEnd - nameStart - 1));
                        //std::cout << "Parsed Parameter Name: " << parameter_names.back() << std::endl; // Debug print

                        // Extracting the default value (if use_default_values is true)
                        if (use_default_values && args.size() == 3) {
                            std::string defaultValueStr = args[2];
                            size_t defaultStart = defaultValueStr.find("default_value=");
                            if (defaultStart != std::string::npos) {
                                defaultValueStr = defaultValueStr.substr(defaultStart + strlen("default_value="));
                                defaultValueStr = trim(defaultValueStr);
                                // TODO: the following is a workaround for parseArguments leaving a final ")". Fix this.
                                // Remove a single trailing parenthesis if present
                                if (!defaultValueStr.empty() && defaultValueStr.back() == ')') {
                                    defaultValueStr.pop_back();
                                }
                                if (!isNumber(defaultValueStr)) {
                                    ERROR_MSG << "Non-numeric default value encountered: " << defaultValueStr << std::endl;
                                    return EXIT_FAILURE;
                                }
                                double defaultValue = std::stod(defaultValueStr);
                                default_values.push_back(defaultValue);
                            } else {
                                // Handle the case where no default value is specified
                                ERROR_MSG << "No default value specified for parameter: " << parameter_names.back() << std::endl;
                                return EXIT_FAILURE;
                            }
                        }
                    }
                }
            }
        }

        file.close();
    }

    for (size_t i = 0; i < parameter_names.size(); ++i) {
        std::stringstream message;
        message << "Parameter: " << parameter_names[i]
                << ", Min: " << min_values[i]
                << ", Max: " << max_values[i];
        if (use_default_values) {
            message << ", Default: " << default_values[i];
        }
        INFO_MSG << message.str() << std::endl;
    }

    default_values_vector.push_back(default_values);

    INFO_MSG << "Single-category parameters: " << single_category_parameters << std::endl;

    if (N_TRAITS != parameter_names.size()) {
        ERROR_MSG << "ERROR: The number of parameters does not match the number of EO optimizer traits" << std::endl;
        return EXIT_FAILURE;
    }

    EvalFunction evalFunc;
    std::unordered_map<std::string, EvalFunction> evaluatorMap = {
        {"cosy", run_cosy},
        {"g4bl", run_g4bl},
        {"dh", run_dh}
    };
    auto evalFuncIter = evaluatorMap.find(EVALUATOR);
    if (evalFuncIter != evaluatorMap.end()) {
        evalFunc = evalFuncIter->second;
    } else {
        // Handle error condition
        std::cerr << "Unknown evaluator: " << EVALUATOR << "\n";
        return EXIT_FAILURE;
    }

    INFO_MSG << "Interactive mode: " << (interactive_mode ? "true" : "false") << "\n";
    if (interactive_mode) {
        // Print parameters
        std::cout << "Parameters:\n";
        std::cout << "POP_SIZE: " << POP_SIZE << "\n";
        std::cout << "MAX_GEN: " << MAX_GEN << "\n";
        std::cout << "MAX_TIME: " << MAX_TIME << "\n";
        std::cout << "RUN_LIMIT_TYPE: " << RUN_LIMIT_TYPE << "\n";
        std::cout << "NSGAII_M_EPSILON: " << M_EPSILON << "\n";
        std::cout << "NSGAII_P_CROSS: " << P_CROSS << "\n";
        std::cout << "NSGAII_P_MUT: " << P_MUT << "\n";
        std::cout << "NSGAII_ETA_C: " << ETA_C << "\n";
        std::cout << "NSGAII_SIGMA: " << SIGMA << "\n";
        std::cout << "NSGAII_P_CHANGE: " << P_CHANGE << "\n";
        std::cout << "MIGRATION_PERIOD: " << MIGRATION_PERIOD << "\n";
        std::cout << "TOURNAMENT_SIZE: " << TOURNAMENT_SIZE << "\n";
        std::cout << "SELECTION_NUMBER: " << SELECTION_NUMBER << "\n";
        std::cout << "Evaluator: " << EVALUATOR << "\n";
        std::cout << "Algorithm: " << ALGORITHM_STR << "\n";
        std::cout << "Source command: " << SOURCE_COMMAND << "\n";
        std::cout << "Program file: " << program_file << "\n";
        std::cout << "Config file: " << config_file << "\n";
        std::cout << "Parse DeepHyper model: " << parse_dh_model << "\n";
        if (parse_dh_model) std::cout << "DeepHyper model filename: " << dh_model_filename << "\n";
        std::cout << "Print all results: " << PRINT_ALL_RESULTS << "\n";
        std::cout << "Program directory: " << program_directory << "\n";
        for (unsigned i = 0; i < parameter_names.size(); ++i) {
            std::cout << "Parameter " << parameter_names[i] << ": min = " << min_values[i] << ", max = " << max_values[
                i] << "\n";
        }
        std::cout << "Single category parameters: " << single_category_parameters << "\n";

        // Ask for user confirmation
        std::string input;
        std::cout << "Do you want to proceed with these parameters? (yes/no)\n";
        std::getline(std::cin, input);
        if (input != "yes") {
            std::cout << "Aborted.\n";
            return EXIT_FAILURE; // or however you want to handle aborting the program
        }
    }

    // crossover and mutation
    //eoQuadCloneOp<System<N_OBJECTIVES, N_TRAITS> > xover;
    eoRealVectorBounds bounds(min_values, max_values);
    // eoUniformMutation<System<N_OBJECTIVES, N_TRAITS> > mutation(bounds, M_EPSILON);
    //double eta_c = 30.0; // A parameter for SBX, typically chosen between 10 and 30
    //double eta_m = 20.0; // A parameter for Polynomial Mutation, typically chosen between 10 and 100
    eoSBXCrossover<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > xover(bounds, ETA_C);
    //double sigma = 0.1; // You can set the standard deviation here.
    //double p_change = 1.0; // Probability to change a given coordinate, default is 1.0
    eoNormalVecMutation<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > mutation(bounds, SIGMA, P_CHANGE);
    eoRealInitBounded<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > init(bounds);
    eoRealInitBounded<moRealVectorNeighbor<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>> init2(bounds);
    SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > pop;

    if (mode == "multistart") {
        // objective functions evaluation
        SystemEval<N_OBJECTIVES, N_TRAITS> eval(evalFunc, SOURCE_COMMAND, parameter_names, single_category_parameters,
                                                program_directory, program_file, config_file, dependency_files);

        eoRealVectorBounds bounds(min_values, max_values);
        eoSBXCrossover<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > xover(ETA_C);
        eoNormalVecMutation<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > mutation(bounds, SIGMA, P_CHANGE);
        eoRealInitBounded<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > init(bounds);
        eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > pop0(POP_SIZE, init);
        pop = SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > (pop0);
        moeoNSGAII<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > nsgaII(
            MAX_GEN, eval, xover, P_CROSS, mutation, P_MUT);

        nsgaII(pop);
    } else if (mode == "MPI" or mode == "redis")  {
        if (mode == "redis") {
            if (redisUseInitJob) {
//TODO: test this
                std::stringstream msgStream;
                INFO_MSG << "Initialising from redis job ID: " << redisInitJobID << std::endl;
                auto manager0 = RedisManager<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>::getInstance(redisIP, redisPort, redisPassword, redisInitJobID, (redisMaxPopSize > 0 ? redisMaxPopSize : POP_SIZE));
                if(use_default_values) manager0->setParameters(parameter_names, default_values); else manager0->setParameters(parameter_names);

                // Retrieve the entire population from Redis
                auto retrievedPop = manager0->retrieveEntirePopulation();
                INFO_MSG << "Retrieved population size: " << retrievedPop.size() << std::endl;

                // Determine total desired population size considering both retrieved population and local default values
                size_t totalDesiredSize = retrievedPop.size() + default_values_vector.size();
                size_t N = std::min(static_cast<size_t>(POP_SIZE), totalDesiredSize);

                INFO_MSG << "Population to set: N = min(" << POP_SIZE << ", " << totalDesiredSize << ") = " << N << std::endl;

                // Resize default_values_vector to accommodate up to N individuals
                default_values_vector.resize(N, std::vector<double>(N_TRAITS));

                // Determine the start index in default_values_vector where retrieved population values should be placed
                size_t startIndex = std::max(0, static_cast<int>(N) - static_cast<int>(retrievedPop.size()));

                // Fill in values from the retrieved population with the necessary shift
                for (size_t i = 0; i < retrievedPop.size() && (startIndex + i) < N; ++i) {
                    for (size_t j = 0; j < N_TRAITS; ++j) {
                        default_values_vector[startIndex + i][j] = retrievedPop[i][j];
                    }
                }

                msgStream << "Loaded the following default value vectors from Redis:" << std::endl;
                for (size_t i = startIndex; i < N; ++i) {
                    msgStream << "(";
                    for (size_t j = 0; j < default_values_vector[i].size(); ++j) {
                        msgStream << default_values_vector[i][j] << (j < default_values_vector[i].size() - 1 ? ", " : "");
                    }
                    msgStream << ")" << std::endl;
                }

// Separately print any local default vectors that were preserved due to size constraints
                if (startIndex > 0) { // Indicates that there are local default values
                    msgStream << "Pre-existing local default values preserved:" << std::endl;
                    for (size_t i = 0; i < startIndex && i < default_values_vector.size(); ++i) {
                        msgStream << "(";
                        for (size_t j = 0; j < default_values_vector[i].size(); ++j) {
                            msgStream << default_values_vector[i][j] << (j < default_values_vector[i].size() - 1 ? ", " : "");
                        }
                        msgStream << ")" << std::endl;
                    }
                }
            }

            RedisManager<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>* manager =
                    RedisManager<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>::getInstance(redisIP, redisPort, redisPassword,
                                                                                             redisJobID, (redisMaxPopSize > 0 ? redisMaxPopSize : POP_SIZE));
            if(use_default_values) manager->setParameters(parameter_names, default_values); else manager->setParameters(parameter_names);
        }

        SystemEval<N_OBJECTIVES, N_TRAITS> eval1(evalFunc, SOURCE_COMMAND, parameter_names, single_category_parameters,
                                                 program_directory, program_file, config_file, dependency_files, 1, 60 * TIMEOUT_MINUTES, EVALUATION_MINIMAL_TIME);

        // LS
        std::vector<unsigned int> minScalingExplored = {1, 2, 3};
        RealVectorNeighborhoodExplorer<moRealVectorNeighbor<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>> explorer(eval1, bounds, 1e-3,
                                                                                                                     30, false,
                                                                                                                     minScalingExplored,3, 5);
        eoGenContinue<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > continuator(MAX_GEN);
        moeoUnboundedArchive<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>> archLS;
        // Create explicit std::vector<double> objects
        std::vector<double> vec1 = {-5000, 0, 0};
        std::vector<double> vec2 = {0, 0, -5000};
        // Create ObjectiveVectors from the std::vector objects
        GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>::ObjectiveVector objVec1(vec1);
        GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>::ObjectiveVector objVec2(vec2);
        // Now you can use these ObjectiveVectors in your excludeList
        std::vector<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>::ObjectiveVector> excludeList = {objVec1, objVec2};
       // Create an instance of moeoBestUnvisitedSelect with the exclusion list
        moeoBestUnvisitedSelect <GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>> select(2, excludeList);
        //moeoBestUnvisitedSelect <GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>> select(2);
        //moeoUnifiedDominanceBasedLSReal<moRealVectorNeighbor<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>> LSalgo(continuator, eval1, archLS, explorer, select);

        // NSGAII
        // Define a pointer to the base continuator type
        eoContinue<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>* continuatorPtr = nullptr;
        eoGenContinue<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > continuatorGen(MAX_GEN);
        eoSecondsElapsedTrackGenContinue<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>> continuatorTime(MAX_TIME*60);
        // Decide which continuator to use based on RUN_LIMIT_TYPE
        if (RUN_LIMIT_TYPE == "maxGen") {
            continuatorPtr = &continuatorGen;
            INFO_MSG << "Worker " << rank << " (maxGen): continuator: " << *continuatorPtr << std::endl;
        } else if (RUN_LIMIT_TYPE == "maxTime") {
            continuatorPtr = &continuatorTime;
            INFO_MSG << "Worker " << rank << " (maxTime): continuator: " << *continuatorPtr << std::endl;
        }
        eoSGATransform<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > transform(xover, P_CROSS, mutation, P_MUT);
        Topology<Complete> topo;

        if (mode == "MPI") {
            MPI_IslandModel<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > model(topo);
        } else if (mode == "redis") {
            Redis_IslandModel<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > model(topo, 0);
        }

        std::vector<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>> pops;
        //SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > pop2(pop20);
        // // Emigration policy
        // // // Element 1
        eoPeriodicContinue<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > criteria_1(MIGRATION_PERIOD);
        eoDetTournamentSelect<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > selectOne_1(TOURNAMENT_SIZE);
        eoSelectNumber<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > who_1(selectOne_1, SELECTION_NUMBER);
        MigPolicy<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > migPolicy_1;
        migPolicy_1.push_back(PolicyElement<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> >(who_1, criteria_1));
        // // Integration policy
        eoPlusReplacement<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > intPolicy_1;
        // TODO: read https://pixorblog.wordpress.com/2019/08/14/curiously-recurring-template-pattern-crtp-in-depth/
        // TODO: learn about C++ templates

        try
        {
            if (continuatorPtr != nullptr) {
                // Now use continuatorPtr which points to the selected continuator
                if (mode == "MPI") {
                    if (ALGORITHM == "NSGAII" or ALGORITHM == "auto") {
                        pops = IslandModelWrapper<moeoNSGAII, GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>, MPI_IslandModel>(
                                num_mpi_ranks, topo, POP_SIZE, default_values_vector, init,
                                intPolicy_1,  // Integration policy
                                migPolicy_1,  // Migration policy
                                HOMOGENEOUS_ISLAND,
                                *continuatorPtr,  // Stopping criteria
                                eval1,  // Evaluation function
                                transform);
                    } else if (ALGORITHM == "ULS" or ALGORITHM == "UnifiedDominanceBasedLS_Real") {
                        pops = IslandModelWrapper<moeoUnifiedDominanceBasedLSReal, GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>,
                                moRealVectorNeighbor<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>, MPI_IslandModel>(
                                num_mpi_ranks, topo, POP_SIZE, default_values_vector, init,
                                intPolicy_1,  // Integration policy
                                migPolicy_1,  // Migration policy
                                HOMOGENEOUS_ISLAND,
                                *continuatorPtr,  // Stopping criteria
                                eval1,  // Evaluation function
                                archLS,
                                explorer,
                                select);
                    } else {
                        // Handle the error case where no continuator is selected
                        std::cerr << "Error: Invalid algorithm: " << ALGORITHM << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (mode == "redis") {
                    if (ALGORITHM == "NSGAII" or ALGORITHM == "auto") {
                        pops = IslandModelWrapper<moeoNSGAII, GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>, Redis_IslandModel>(
                                1, topo, POP_SIZE, default_values_vector, init,
                                intPolicy_1,  // Integration policy
                                migPolicy_1,  // Migration policy
                                HOMOGENEOUS_ISLAND,
                                *continuatorPtr,  // Stopping criteria
                                eval1,  // Evaluation function
                                transform);
                    } else if (ALGORITHM == "ULS" or ALGORITHM == "UnifiedDominanceBasedLS_Real") {
                        pops = IslandModelWrapper<moeoUnifiedDominanceBasedLSReal, GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>,
                                moRealVectorNeighbor<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>, Redis_IslandModel>(
                                1, topo, POP_SIZE, default_values_vector, init,
                                intPolicy_1,  // Integration policy
                                migPolicy_1,  // Migration policy
                                HOMOGENEOUS_ISLAND,
                                *continuatorPtr,  // Stopping criteria
                                eval1,  // Evaluation function
                                archLS,
                                explorer,
                                select);
                    } else {
                        // Handle the error case where no continuator is selected
                        std::cerr << "Error: Invalid algorithm: " << ALGORITHM << std::endl;
                        return EXIT_FAILURE;
                    }
                }
                cout << "Continuator status on MPI rank " << rank << ": " << *continuatorPtr << endl;
            } else {
                std::cerr << "Error: RUN_LIMIT_TYPE is not correctly specified." << std::endl;
                return EXIT_FAILURE;
            }
        }
        catch(exception& e)
        {
            cout << "Exception: " << e.what() << '\n';
        }

        if (mode == "MPI") {
            moeoUnboundedArchive<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > arch1;
            pop = SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > (pops[rank]);
            cout << "Arhive update on MPI rank " << rank << ": " << arch1(pops[rank]) << endl;
            arch1.sortedPrintOn(cout);

        } else if (mode == "redis") {
            RedisManager<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>* manager =
                    RedisManager<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>::getInstance(redisIP, redisPort, redisPassword,
                                                                                             redisJobID, (redisMaxPopSize > 0 ? redisMaxPopSize : POP_SIZE));
            if (manager->getIsMainInstance() or redisWriteAll) {
                auto retrievedPop = manager->retrieveEntirePopulation();
                //if (!redisWriteAll) manager->clearPopulation();
                std::cout << "Retrieved pop: " << retrievedPop.size() << std::endl;
                eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>> &mainPop = pops[0];
                cout << "Main pop: " << mainPop.size() << endl;
                mainPop.append(retrievedPop);
                pop = SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > >(mainPop);
                moeoUnboundedArchive<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > arch1;
                //cout << "Pop size before archiving: " << pops[0].size() << endl;
                cout << "Archive update: " << arch1(pop) << endl;

                arch1.sortedPrintOn(cout);
            }
        };


        //model.add(nsgaII_1);
        //model.add(nsgaII_2);
        //model();
        std::cout << "Model run complete on mpi rank " << rank << std::endl;

    } else if (mode == "redistest") {
        RedisManager<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>* manager = RedisManager<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>::getInstance(redisIP, redisPort, redisPassword, "test10", (redisMaxPopSize > 0 ? redisMaxPopSize : POP_SIZE));

        // Test key and value
        std::string testKey = "testKey";
        std::string testValue = "Hello, Redis!";

        // Set a value
        manager->setValue(testKey, testValue);
        std::cout << "Value set for key '" << testKey << "': " << testValue << std::endl;

        // Get the value
        std::string retrievedValue = manager->getValue(testKey);
        std::cout << "Retrieved value for key '" << testKey << "': " << retrievedValue << std::endl;

        // Delete the key
        manager->deleteKey(testKey);
        std::cout << "Key '" << testKey << "' deleted." << std::endl;

        // Try to get the value again
        retrievedValue = manager->getValue(testKey);
        if (retrievedValue.empty()) {
            std::cout << "No value found for key '" << testKey << "' after deletion." << std::endl;
        }

        // Test population size management
        size_t popSize = manager->getPopulationSize();
        std::cout << "Current population size: " << popSize << std::endl;

        // Check initial state of isMainInstance
        bool initialMainInstanceStatus = manager->getIsMainInstance();
        std::cout << "Initial isMainInstance status: " << (initialMainInstanceStatus ? "true" : "false") << std::endl;

        // Set isMainInstance to true for testing
        manager->setIsMainInstance(true);
        std::cout << "isMainInstance set to true for debugging." << std::endl;

        // Check the state of isMainInstance after setting it to true
        bool updatedMainInstanceStatus = manager->getIsMainInstance();
        std::cout << "Updated isMainInstance status: " << (updatedMainInstanceStatus ? "true" : "false") << std::endl;

        // Increment population size
        int incrementAmount = 5;
        manager->incrementPopulationSize(incrementAmount);
        std::cout << "Incremented population size by " << incrementAmount << "." << std::endl;

        // Get new population size
        popSize = manager->getPopulationSize();
        std::cout << "New population size after increment: " << popSize << std::endl;

        // Decrement population size
        int decrementAmount = 3;
        manager->decrementPopulationSize(decrementAmount);
        std::cout << "Decremented population size by " << decrementAmount << "." << std::endl;

        // Get final population size
        popSize = manager->getPopulationSize();
        std::cout << "Final population size after decrement: " << popSize << std::endl;

        manager->clearPopulation();

        manager->addTestIndividual("{\"value\": \"1 2 3 8 1 2 3 4 5 6 7 8 \"}");
        auto retrievedPop = manager->retrievePopulation(1);

        manager->clearPopulation();

        manager->setParameters(parameter_names);
        manager->addTestIndividual("{\"value\": \"1 2 3 8 N8:8 B50:1 B51:2 B52:3 B53:4 Z90:5 N6:6 N7:7 \"}");
        retrievedPop = manager->retrievePopulation(1);

        manager->clearPopulation();

        manager->setParameters(parameter_names, default_values);
        manager->addTestIndividual("{\"value\": \"1 2 3 8 B50:1 B51:2 B52:3 B53:4 Z90:5 N6:6 N8:8 N7:7 \"}");
        retrievedPop = manager->retrievePopulation(1);

        manager->clearPopulation();

        // objective functions evaluation
        SystemEval<N_OBJECTIVES, N_TRAITS> eval(evalFunc, SOURCE_COMMAND, parameter_names, single_category_parameters,
                                                program_directory, program_file, config_file, dependency_files);

        eoRealVectorBounds bounds(min_values, max_values);
        eoSBXCrossover<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > xover(ETA_C);
        eoNormalVecMutation<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > mutation(bounds, SIGMA, P_CHANGE);
        eoRealInitBounded<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > init(bounds);
        eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > pop0(POP_SIZE, init);
        pop = SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > (pop0);
        moeoNSGAII<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > nsgaII(
            MAX_GEN, eval, xover, P_CROSS, mutation, P_MUT);

        nsgaII(pop);

        // First call to updatePopulation
        std::cout << "Sending population to Redis DB:" << std::endl;
        eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>& mainPop = pop;
        for (size_t i = 0; i < mainPop.size(); ++i) {
            std::ostringstream oss;
            mainPop[i].printOn(oss);  // Assuming printOn is defined for the individual type
            std::cout << "Individual " << i << ": " << oss.str() << std::endl;
        }

        // First call to updatePopulation
        manager->updatePopulation(pop);
        std::cout << "First call to updatePopulation completed." << std::endl;

        // Second call to updatePopulation with the same 'pop'
        manager->updatePopulation(pop);
        std::cout << "Second call to updatePopulation with the same population completed." << std::endl;

        if (manager->getIsMainInstance()) {
            auto retrievedPop = manager->retrievePopulation(POP_SIZE);
            std::cout << "Retrieved population size: " << retrievedPop.size() << std::endl;
            pop = SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > (retrievedPop);
        }
    }

    moeoUnboundedArchive<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > arch;

    std::map<std::string, double> parameters = {
        {"popSize", POP_SIZE},
        {"maxGen", MAX_GEN},
        {"mutEpsilon", M_EPSILON},
        {"pCross", P_CROSS},
        {"pMut", P_MUT},
        {"eta_c", ETA_C},
        {"sigma", SIGMA},
        {"p_change", P_CHANGE}
    };

    std::string filenameSuffix = getFilenameSuffix();

    if (comm.rank() != DEFAULT_MASTER) {
        // Worker process: Send pop to the master process
        DEBUG_MSG << "Worker " << comm.rank() << ": Sending population to master." << std::endl;
        comm.send(DEFAULT_MASTER, eo::mpi::Channel::Messages, pop);
        DEBUG_MSG << "Worker " << comm.rank() << ": Population sent." << std::endl;
        //arch(pop);
    }

    if (comm.rank() == DEFAULT_MASTER) {
        // Master process: Receive pop from all other processes
        if (mode=="MPI") {
            DEBUG_MSG << "Master: Ready to receive populations from workers." << std::endl;
        } else {
            DEBUG_MSG << "Main instance: Writing the archive to files." << std::endl;
        }
        for (int i = 1; i < comm.size(); ++i) {
            eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > popUnpacked0(POP_SIZE, init);
            SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > popUnpacked(popUnpacked0);
            DEBUG_MSG << "Master: Waiting to receive population from worker " << i << "." << std::endl;
            comm.recv(i, eo::mpi::Channel::Messages, popUnpacked);
            DEBUG_MSG << "Master: Received population from worker " << i << "." << std::endl;
            arch(popUnpacked); // Process the received population
        }
        if (mode=="MPI")DEBUG_MSG << "Master: All populations received and processed." << std::endl;

        //eoserial::unpack( o, "pop", popUnpacked);
        arch(pop);
        //arch(pop);

        // printing of the final archive
        cout << "Final Archive" << endl;
        arch.sortedPrintOn(cout);
        //cout << endl;

        // Save final archive to a CSV file with the date and time in the filename
        std::ofstream csv_file;
        std::ostringstream filenameStream;
        filenameStream << "pareto_frontier_" << filenameSuffix << ".csv";
        std::string filename = filenameStream.str();
        csv_file.open(filename);

        // Write parameters data to file
        writeParametersToCsv(csv_file, parameters, parameter_names, single_category_parameters);
        for (unsigned i = 0; i < arch.size(); ++i) {
            // Writing objective function values
            for (unsigned j = 0; j < N_OBJECTIVES; ++j) {
                csv_file << arch[i].objectiveVector()[j] << ",";
            }
            // Writing solution parameter vectors
            for (unsigned j = 0; j < N_TRAITS; ++j) {
                csv_file << arch[i][j];
                if (j != N_TRAITS - 1) {
                    csv_file << ",";
                }
            }
            csv_file << "\n";
        }
        csv_file.close();

    }

    if (PRINT_ALL_RESULTS) {
        // Save all evaluated solutions to a CSV file
        std::ofstream all_solutions_file;
        std::ostringstream filenameStream2;
        filenameStream2 << "all_evaluated_solutions_" << filenameSuffix << ".csv";
        std::string filename2 = filenameStream2.str();
        all_solutions_file.open(filename2);

        // Write parameters data to file
        writeParametersToCsv(all_solutions_file, parameters, parameter_names, single_category_parameters);
        for (unsigned i = 0; i < allEvaluatedSolutions.size(); ++i) {
            // Writing objective function values
            for (unsigned j = 0; j < N_OBJECTIVES; ++j) {
                all_solutions_file << allEvaluatedSolutions[i].objectiveVector()[j] << ",";
            }
            // Writing solution parameter vectors
            for (unsigned j = 0; j < N_TRAITS; ++j) {
                all_solutions_file << allEvaluatedSolutions[i][j];
                if (j != N_TRAITS - 1) {
                    all_solutions_file << ",";
                }
            }
            all_solutions_file << "\n"; // Also write the generation number
        }
        all_solutions_file.close();
    }

    return EXIT_SUCCESS;
}
