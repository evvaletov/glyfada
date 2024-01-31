#include <stdio.h>
#include <moeo>
#include <es/eoRealInitBounded.h>
#include <es/eoRealOp.h>
#include <es/eoNormalMutation.h>
#include <mpi/eoMpi.h>

#include <utils/eoParser.h>
#include <cstdlib>

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

#include "logging.h"
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
    const std::string &);

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
               int islandId = -1)
        : evaluator(evaluator), source_command(source_command), parameter_names(parameter_names),
          single_category_parameters(single_category_parameters),
          program_directory(program_directory), program_file(program_file), config_file(config_file),
          dependency_files(
              const_cast<vector<std::string> &>(dependency_files)), islandId(islandId) {
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

            // Call run_g4beamline or run_cosy with the appropriate arguments
            std::vector<double> results = evaluator(source_command, program_directory, program_file, config_file,
                                                    dependency_files, parameter_names,
                                                    parameter_values, single_category_parameters);

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
};

// TODO DONE: check if in eoNormalVecMutation the sigma argument scaled by the range: yes
// TODO: implement hybrid island method
// TODO: implement MPI parallelization
// TODO: track statistics for the quality and speed of optimization
// TODO: implement hyperparameter optimization
// TODO: manage loads between islands if running on same MPI rank
// TODO: implement the possibility of using a different number of OMP threads for each island
// TODO: merge regular and island-based codes (set mode from parameter file)


int main(int argc, char *argv[]) {
    eo::mpi::Node::init(argc, argv);
    //loadRMCParameters (argc, argv);
    int rank;
    bmpi::communicator &comm = eo::mpi::Node::comm();
    rank = comm.rank();
    int num_mpi_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_mpi_ranks);
    INFO_MSG << "MPI rank: " << rank << endl;

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
    // Narrow it down to microseconds and convert to an unsigned integer
    unsigned seed = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    // Seed the random number generator
    eo::rng.reseed(seed);
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
        return 1;
    }

    // Read the JSON file
    nlohmann::json json_data = read_json_file(config_filename);

    std::string program_file = json_data["program_file"].get<std::string>();
    std::string config_file = json_data["config_file"].get<std::string>();
    std::string program_directory = json_data.value("program_directory", "cwd");
    // Check if program_directory is set to 'cwd', then get the current working directory
    if (program_directory == "cwd") {
        program_directory = fs::current_path().string();
    }
    std::vector<std::string> dependency_files = read_dependency_files(json_data);

    // Read parameters from JSON file
    unsigned int POP_SIZE = json_data.value("popSize", 200);
    unsigned int MAX_GEN = json_data.value("maxGen", 50);
    //TODO: Implement MAX_TIME
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
    std::string SOURCE_COMMAND = json_data.value("source_command", "");
    bool PRINT_ALL_RESULTS = json_data.value("print_all_results", false);

    bool interactive_mode = json_data.value("interactive_mode", false);
    // modes: multistart, homogeneous
    std::string mode = json_data.value("mode", "multistart");
    INFO_MSG << "Operation mode set to: " << mode << std::endl;

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
    if (comm.size() >= 24 && max_threads < 16) {
        ERROR_MSG << "Error: Number of MPI ranks is >= 24 and maximal OpenMP threads is below 16." << std::endl;
        return EXIT_FAILURE;
    }

    // First, extract the parameters from the JSON data
    std::vector<nlohmann::json> search_space = json_data["parameters"].get<std::vector<nlohmann::json> >();

    // Create empty vectors to hold the parameter names and their bounds
    std::vector<std::string> parameter_names;
    std::vector<double> min_values;
    std::vector<double> max_values;

    // Create a string to hold the single-category parameters
    std::string single_category_parameters;

    if (!parse_dh_model) {
        // Iterate through each parameter
        for (const auto &param: search_space) {
            // Check the parameter type
            std::string param_type = param["type"].get<std::string>();

            if (param_type == "continuous") {
                // If it's a continuous parameter, add its name, min_value, and max_value to the appropriate vectors
                parameter_names.push_back(param["name"].get<std::string>());
                min_values.push_back(param["min_value"].get<double>());
                max_values.push_back(param["max_value"].get<double>());
            } else if (param_type == "categorical") {
                // If it's a categorical parameter, check how many categories it has
                std::vector<std::string> categories = param["values"].get<std::vector<std::string> >();

                if (categories.size() == 1) {
                    // If it has only one category, add it to the single-category parameters string
                    single_category_parameters += param["name"].get<std::string>() + "=" + categories[0] + " ";
                } else {
                    // If it has more than one category, output a "not implemented" message and abort
                    std::cerr << "Categorical parameters with more than one category are not implemented." << std::endl;
                    exit(1);
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

                if (args.size() == 2 && args[0].find('[') != std::string::npos) {
                    // Handle single-category parameters
                    size_t categoryStart = args[0].find_first_of("[\"");
                    size_t categoryEnd = args[0].find_last_of("\"]");
                    if (categoryStart != std::string::npos && categoryEnd != std::string::npos && categoryStart !=
                        categoryEnd) {
                        std::string category = args[0].substr(categoryStart + 1, categoryEnd - categoryStart - 1);
                        std::string name = args[1].substr(args[1].find_first_of("\"\'") + 1,
                                                          args[1].find_last_of("\"\'") - args[1].find_first_of("\"\'") -
                                                          1);
                        single_category_parameters += name + "=" + category + " ";
                    }
                } else {
                    if (args.size() < 2 || args.size() > 3) continue; // Ensure correct number of arguments

                    std::string range = args[0];
                    //std::cout << "Range string: " << range << std::endl; // Debug print

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

                    double min_val = isNumber(min_str) ? std::stod(min_str) : 0.0;
                    double max_val = isNumber(max_str) ? std::stod(max_str) : 0.0;
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
                }
            }
        }

        file.close();
    }

    for (size_t i = 0; i < parameter_names.size(); ++i) {
        INFO_MSG << "Parameter: " << parameter_names[i] << ", Min: " << min_values[i] << ", Max: " << max_values[i] <<
                std::endl;
    }
    INFO_MSG << "Single-category parameters: " << single_category_parameters << std::endl;

    if (N_TRAITS != parameter_names.size()) {
        ERROR_MSG << "ERROR: The number of parameters does not match the number of EO optimizer traits" << std::endl;
        return 1;
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
        std::cout << "M_EPSILON: " << M_EPSILON << "\n";
        std::cout << "P_CROSS: " << P_CROSS << "\n";
        std::cout << "P_MUT: " << P_MUT << "\n";
        std::cout << "ETA_C: " << ETA_C << "\n";
        std::cout << "SIGMA: " << SIGMA << "\n";
        std::cout << "P_CHANGE: " << P_CHANGE << "\n";
        std::cout << "MIGRATION_PERIOD: " << MIGRATION_PERIOD << "\n";
        std::cout << "TOURNAMENT_SIZE: " << TOURNAMENT_SIZE << "\n";
        std::cout << "SELECTION_NUMBER: " << SELECTION_NUMBER << "\n";
        std::cout << "Evaluator: " << EVALUATOR << "\n";
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
    eoSBXCrossover<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > xover(ETA_C);
    //double sigma = 0.1; // You can set the standard deviation here.
    //double p_change = 1.0; // Probability to change a given coordinate, default is 1.0
    eoNormalVecMutation<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > mutation(bounds, SIGMA, P_CHANGE);
    eoRealInitBounded<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > init(bounds);
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
    } else {
        // objective functions evaluation
        SystemEval<N_OBJECTIVES, N_TRAITS> eval1(evalFunc, SOURCE_COMMAND, parameter_names, single_category_parameters,
                                                 program_directory, program_file, config_file, dependency_files, 1);
        SystemEval<N_OBJECTIVES, N_TRAITS> eval2(evalFunc, SOURCE_COMMAND, parameter_names, single_category_parameters,
                                                 program_directory, program_file, config_file, dependency_files, 2);

        eoGenContinue<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > continuator(MAX_GEN);
        eoSGATransform<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > transform(xover, P_CROSS, mutation, P_MUT);
        Topology<Complete> topo;
        MPI_IslandModel<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > model(topo);

        // ISLAND 1
        // generate initial population
        eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > pop1(POP_SIZE, init);
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
        // build NSGA-II
        // TODO: read https://pixorblog.wordpress.com/2019/08/14/curiously-recurring-template-pattern-crtp-in-depth/
        // TODO: learn about C++ templates
        //Island<moeoNSGAII,System<N_OBJECTIVES, N_TRAITS> > nsgaII_2(pop2, intPolicy_2, migPolicy_2, MAX_GEN, eval, xover, P_CROSS, mutation, P_MUT);
        Island<moeoNSGAII, GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > nsgaII_2(
            pop1, // Population
            intPolicy_1, // Integration policy
            migPolicy_1, // Migration policy
            continuator, // Stopping criteria
            eval1, // Evaluation function
            transform // Transformation operator combining crossover and mutation
        );

        /*// ISLAND 2
        // generate initial population
        eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > pop1(POP_SIZE, init);
        //SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > pop1(pop10);
        // // Emigration policy
        // // // Element 1
        eoPeriodicContinue<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > criteria_1(1);
        eoDetTournamentSelect<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > selectOne_1(15);
        eoSelectNumber<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > who_1(selectOne_1, 5);
        MigPolicy<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > migPolicy_1;
        migPolicy_1.push_back(PolicyElement<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> >(who_1, criteria_1));
        // // Integration policy
        eoPlusReplacement<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > intPolicy_1;
        // build NSGA-II
        Island<moeoNSGAII, GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > nsgaII_1(
            pop1, // Population
            intPolicy_1, // Integration policy
            migPolicy_1, // Migration policy
            continuator, // Stopping criteria
            eval1, // Evaluation function
            transform // Transformation operator combining crossover and mutation
        );*/

        try
        {

            std::vector<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>>> pops = IslandModelWrapper<moeoNSGAII,GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS>,MPI_IslandModel>(num_mpi_ranks, topo, POP_SIZE, init,
                intPolicy_1, // Integration policy
                migPolicy_1, // Migration policy
                HOMOGENEOUS_ISLAND,
                continuator, // Stopping criteria
                eval1, // Evaluation function
                transform);

            moeoUnboundedArchive<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > arch1;
            pop = SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > (pops[rank]);
            cout << "Arhive update on MPI rank " << rank << ": " << arch1(pops[rank]) << endl;
            arch1.sortedPrintOn(cout);
        }
        catch(exception& e)
        {
            cout << "Exception: " << e.what() << '\n';
        }


        //model.add(nsgaII_1);
        //model.add(nsgaII_2);

        //model();
        std::cout << "Model run complete on mpi rank " << rank << std::endl;

        /*if (rank==0) {
            moeoUnboundedArchive<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > arch1;
            std::cout << "Archive update island 1 on mpi rank " << rank << std::endl;
            cout << "Arhive update 1: " << arch1(pop1) << endl;
            arch1.sortedPrintOn(cout);
            pop = SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > (pop1);
        }

        if (rank==1) {
            moeoUnboundedArchive<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > arch2;
            std::cout << "Archive update island 2 on mpi rank " << rank << std::endl;
            cout << "Arhive update 2: " << arch2(pop2) << endl;
            arch2.sortedPrintOn(cout);
            pop = SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > (pop2);
        }*/

        //std::cout << "Joining pops on mpi rank " << rank << std::endl;
        ///pop1.append(pop2);
        //pop = SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > (pop1);
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


    if (comm.rank() != DEFAULT_MASTER) {
        // Worker process: Send pop to the master process
        DEBUG_MSG << "Worker " << comm.rank() << ": Sending population to master." << std::endl;
        comm.send(DEFAULT_MASTER, eo::mpi::Channel::Messages, pop);
        DEBUG_MSG << "Worker " << comm.rank() << ": Population sent." << std::endl;
        //arch(pop);
    }

    if (comm.rank() == DEFAULT_MASTER) {
        // Master process: Receive pop from all other processes
        DEBUG_MSG << "Master: Ready to receive populations from workers." << std::endl;
        for (int i = 1; i < comm.size(); ++i) {
            eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > popUnpacked0(POP_SIZE, init);
            SerializableBase<eoPop<GlyfadaMoeoRealVector<N_OBJECTIVES, N_TRAITS> > > popUnpacked(popUnpacked0);
            DEBUG_MSG << "Master: Waiting to receive population from worker " << i << "." << std::endl;
            comm.recv(i, eo::mpi::Channel::Messages, popUnpacked);
            DEBUG_MSG << "Master: Received population from worker " << i << "." << std::endl;
            arch(popUnpacked); // Process the received population
        }
        DEBUG_MSG << "Master: All populations received and processed." << std::endl;

        //eoserial::unpack( o, "pop", popUnpacked);
        arch(pop);
        //arch(pop);

        // printing of the final archive
        cout << "Final Archive" << endl;
        arch.sortedPrintOn(cout);
        //cout << endl;

        // Save final archive to a CSV file
        std::ofstream csv_file;
        std::ostringstream filenameStream;
        filenameStream << "pareto_frontier_" << rank << ".csv";
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
        filenameStream2 << "all_evaluated_solutions_" << rank << ".csv";
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
