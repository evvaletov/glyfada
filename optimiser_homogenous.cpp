#include <stdio.h>
#include <moeo>
#include <es/eoRealInitBounded.h>
#include <es/eoRealOp.h>
#include <es/eoNormalMutation.h>

//#include "run_g4beamline.h"
#include "run_cosy.h"
#include "run_g4bl.h"
#include "run_dh.h"
#include <vector>
#include <string>
#include <json.hpp>
#include <fstream>
#include <smp>

#include <execinfo.h>
#include <cxxabi.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mutex>
#include <es/eoSBXcross.h>

using namespace std;
using namespace paradiseo::smp;

// Global constants
constexpr unsigned int N_OBJECTIVES = 3;
constexpr unsigned int N_TRAITS = 8;

void writeParametersToCsv(std::ofstream& file, const std::map<std::string, double>& parameters, const std::vector<std::string>& parameter_names, const std::string& single_category_parameters) {
    // Write parameters to the header of the CSV file
    for (const auto& [param, value] : parameters) {
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

typedef std::vector<double> (*EvalFunction)(
        const std::string&,
        const std::string&,
        const std::string&,
        const std::string&,
        const std::vector<std::string>&,
        const std::vector<std::string>&,
        const std::vector<double>&,
        const std::string&);

std::vector<std::string> read_dependency_files(const nlohmann::json &json) {
    std::vector<std::string> dependency_files;
    if (json.contains("dependency_files") && !json.at("dependency_files").is_null() && json.at("dependency_files").is_array())  {
        for (const auto &file: json.at("dependency_files")) {
            dependency_files.push_back(file.get<std::string>());
        }
    }
    return dependency_files;
}

// the moeoObjectiveVectorTraits
template<int N_OBJECTIVES>
class Sch1ObjectiveVectorTraits : public moeoObjectiveVectorTraits {
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
using Sch1ObjectiveVector = moeoRealObjectiveVector<Sch1ObjectiveVectorTraits<N_OBJECTIVES>>;

// multi-objective evolving object for the System problem
template<int N_OBJECTIVES, int N_TRAITS>
class System : public moeoRealVector<Sch1ObjectiveVector<N_OBJECTIVES>> {
public:
    System() : moeoRealVector<Sch1ObjectiveVector<N_OBJECTIVES>>(N_TRAITS) {}
};

// Global vectors to store all evaluated solutions and their corresponding generations
std::vector<System<N_OBJECTIVES, N_TRAITS>> allEvaluatedSolutions;
std::mutex mutexAllEvaluated; // A mutex for controlling access to the above vectors

// evaluation of objective functions
template<int N_OBJECTIVES, int N_TRAITS>
class SystemEval : public moeoEvalFunc<System<N_OBJECTIVES, N_TRAITS>> {
public:
    SystemEval(EvalFunction evaluator, const std::string &source_command, const std::vector<std::string> &parameter_names,
               const std::string &single_category_parameters, const std::string &program_directory,
               const std::string &program_file, const std::string &config_file, const std::vector<std::string> &dependency_files)
            : evaluator(evaluator), source_command(source_command), parameter_names(parameter_names), single_category_parameters(single_category_parameters),
            program_directory(program_directory), program_file(program_file), config_file(config_file),
              dependency_files(
                      const_cast<vector<std::string> &>(dependency_files)) {}

    void operator()(System<N_OBJECTIVES, N_TRAITS> &_sch1) {
        if (_sch1.invalidObjectiveVector()) {
            Sch1ObjectiveVector<N_OBJECTIVES> objVec;
            vector<double> parameter_values(N_TRAITS); // use dimension() here instead of size()
            for (size_t i = 0; i < N_TRAITS; ++i) // use dimension() here instead of size()
            {
                parameter_values[i] = _sch1[i];
                // std::cout << "parameter_values[" << i << "] = " << parameter_values[i] << std::endl;
            }

            // Call run_g4beamline or run_cosy with the appropriate arguments
            std::vector<double> results = evaluator(source_command, program_directory, program_file, config_file, dependency_files, parameter_names,
                                                   parameter_values, single_category_parameters);
            std::cout << results[0] << ", " << results[1] << ", " << results[2] << std::endl;

            // Set the objectives based on the results from run_g4beamline or run_cosy
            for (size_t i = 0; i < N_OBJECTIVES; ++i) {
                objVec[i] = results[i];
            }

            _sch1.objectiveVector(objVec);

            // Critical section begins
            mutexAllEvaluated.lock();
            allEvaluatedSolutions.push_back(_sch1);
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
};

// TODO DONE: check if in eoNormalVecMutation the sigma argument scaled by the range: yes
// TODO: implement hybrid island method
// TODO: implement MPI parallelization
// TODO: track statistics for the quality and speed of optimization
// TODO: implement hyperparameter optimization

// main
int main(int argc, char *argv[]) {
    //int num_threads = 1;  // Set this to the number of threads you want OpenMP to use
    //omp_set_num_threads(num_threads);

    eoParser parser(argc, argv);  // for user-parameter reading
    eoState state;                // to keep all things allocated

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file_path>\n";
        return 1;
    }
    std::string config_filename = argv[1];
    // Read the JSON file
    //std::string json_filename = "/home/evaletov/paradiseo/gitclone/build/moeo/g4bl/test/parametersDH.json";
    //std::string json_filename = "parameterslbnl.json";
    nlohmann::json json_data = read_json_file(config_filename);

    // Print the entire JSON data
    //std::cout << json_data.dump(4) << "\n"; // 4 is for indentation

    // Extract parameter names and bounds from the JSON file
    //std::vector<std::string> parameter_names = json_data["parameter_names"].get<std::vector<std::string>>();
    //std::vector<double> min_values = json_data["min_values"].get<std::vector<double>>();
    //std::vector<double> max_values = json_data["max_values"].get<std::vector<double>>();

    // First, extract the parameters from the JSON data
    std::vector<nlohmann::json> search_space = json_data["parameters"].get<std::vector<nlohmann::json>>();

    // Create empty vectors to hold the parameter names and their bounds
    std::vector<std::string> parameter_names;
    std::vector<double> min_values;
    std::vector<double> max_values;

    // Create a string to hold the single-category parameters
    std::string single_category_parameters;

    // Iterate through each parameter
    for (const auto& param : search_space) {
        // Check the parameter type
        std::string param_type = param["type"].get<std::string>();

        if (param_type == "continuous") {
            // If it's a continuous parameter, add its name, min_value, and max_value to the appropriate vectors
            parameter_names.push_back(param["name"].get<std::string>());
            min_values.push_back(param["min_value"].get<double>());
            max_values.push_back(param["max_value"].get<double>());
        }
        else if (param_type == "categorical") {
            // If it's a categorical parameter, check how many categories it has
            std::vector<std::string> categories = param["values"].get<std::vector<std::string>>();

            if (categories.size() == 1) {
                // If it has only one category, add it to the single-category parameters string
                single_category_parameters += param["name"].get<std::string>() + "=" + categories[0] + " ";
            }
            else {
                // If it has more than one category, output a "not implemented" message and abort
                std::cerr << "Categorical parameters with more than one category are not implemented." << std::endl;
                exit(1);
            }
        }
    }

    std::string program_file = json_data["program_file"].get<std::string>();
    std::string config_file = json_data["config_file"].get<std::string>();
    std::string program_directory = json_data["program_directory"].get<std::string>();
    std::vector<std::string> dependency_files = read_dependency_files(json_data);

    // parameters
    std::cout << "Parameters :" << parameter_names.size() << std::endl;
    assert(N_TRAITS == parameter_names.size());

    // Read parameters from JSON file
    unsigned int POP_SIZE = json_data.value("popSize", 200);
    unsigned int MAX_GEN = json_data.value("maxGen", 50);
    double M_EPSILON = json_data.value("mutEpsilon", 0.01);
    double P_CROSS = json_data.value("pCross", 0.25);
    double P_MUT = json_data.value("pMut", 0.35);
    double eta_c = json_data.value("eta_c", 30.0);
    double sigma = json_data.value("sigma", 0.1);
    double p_change = json_data.value("p_change", 1.0);
    std::string evaluator = json_data.value("evaluator", "cosy");
    std::string source_command = json_data.value("source_command", "");

    EvalFunction evalFunc = nullptr;

    if (evaluator == "cosy") {
        evalFunc = run_cosy;
    } else if (evaluator == "g4bl") {
        evalFunc = run_g4bl;
    } else if (evaluator == "dh") {
        evalFunc = run_dh;
    } else {
        // handle error condition
        std::cerr << "Unknown evaluator: " << evaluator << "\n";
        return EXIT_FAILURE;
    }

    // Check for interactive mode
    bool interactive_mode = json_data.value("interactive_mode", false);
    // Check if the 'interactive_mode' key exists in the JSON data
    /* if (json_data.contains("interactive_mode")) {
        bool interactive_mode = json_data["interactive_mode"];
        std::cout << "Interactive mode: " << (interactive_mode ? "true" : "false") << "\n";
    } else {
        std::cout << "'interactive_mode' key not found in JSON data.\n";
    } */
    std::cout << "Interactive mode: " << (interactive_mode ? "true" : "false") << "\n";
    if (interactive_mode) {
        // Print parameters
        std::cout << "Parameters:\n";
        std::cout << "POP_SIZE: " << POP_SIZE << "\n";
        std::cout << "MAX_GEN: " << MAX_GEN << "\n";
        std::cout << "M_EPSILON: " << M_EPSILON << "\n";
        std::cout << "P_CROSS: " << P_CROSS << "\n";
        std::cout << "P_MUT: " << P_MUT << "\n";
        std::cout << "Evaluator: " << evaluator << "\n";
        std::cout << "Source command: " << source_command << "\n";
        std::cout << "Program file: " << program_file << "\n";
        std::cout << "Config file: " << config_file << "\n";
        std::cout << "Program directory: " << program_directory << "\n";
        for (unsigned i = 0; i < parameter_names.size(); ++i) {
            std::cout << "Parameter " << parameter_names[i] << ": min = " << min_values[i] << ", max = " << max_values[i] << "\n";
        }
        std::cout << "Single category parameters: " << single_category_parameters << "\n";

        // Ask for user confirmation
        std::string input;
        std::cout << "Do you want to proceed with these parameters? (yes/no)\n";
        std::getline(std::cin, input);
        if (input != "yes") {
            std::cout << "Aborted.\n";
            return 1;  // or however you want to handle aborting the program
        }
    }

    // objective functions evaluation
    SystemEval<N_OBJECTIVES, N_TRAITS> eval(evalFunc, source_command, parameter_names, single_category_parameters, program_directory, program_file, config_file, dependency_files);
    // crossover and mutation
    //eoQuadCloneOp<System<N_OBJECTIVES, N_TRAITS> > xover;
    eoRealVectorBounds bounds(min_values, max_values);
    // eoUniformMutation<System<N_OBJECTIVES, N_TRAITS> > mutation(bounds, M_EPSILON);
    //double eta_c = 30.0; // A parameter for SBX, typically chosen between 10 and 30
    //double eta_m = 20.0; // A parameter for Polynomial Mutation, typically chosen between 10 and 100
    eoSBXCrossover<System<N_OBJECTIVES, N_TRAITS>> xover(eta_c);
    //double sigma = 0.1; // You can set the standard deviation here.
    //double p_change = 1.0; // Probability to change a given coordinate, default is 1.0
    eoNormalVecMutation<System<N_OBJECTIVES, N_TRAITS> > mutation(bounds, sigma, p_change);
    eoRealInitBounded<System<N_OBJECTIVES, N_TRAITS>> init(bounds);
    eoGenContinue<System<N_OBJECTIVES, N_TRAITS>> continuator(MAX_GEN);
    eoSGATransform<System<N_OBJECTIVES, N_TRAITS>> transform(xover, P_CROSS, mutation, P_MUT);
    Topology<Complete> topo;
    IslandModel<System<N_OBJECTIVES, N_TRAITS>> model(topo);



    // ISLAND 1
    // generate initial population
    eoPop<System<N_OBJECTIVES, N_TRAITS> > pop2(POP_SIZE, init);
    // // Emigration policy
    // // // Element 1
    eoPeriodicContinue<System<N_OBJECTIVES, N_TRAITS>> criteria_2(10);
    eoDetTournamentSelect<System<N_OBJECTIVES, N_TRAITS>> selectOne_2(15);
    eoSelectNumber<System<N_OBJECTIVES, N_TRAITS>> who_2(selectOne_2, 1);
    MigPolicy<System<N_OBJECTIVES, N_TRAITS>> migPolicy_2;
    migPolicy_2.push_back(PolicyElement<System<N_OBJECTIVES, N_TRAITS>>(who_2, criteria_2));
    // // Integration policy
    eoPlusReplacement<System<N_OBJECTIVES, N_TRAITS>> intPolicy_2;
    // build NSGA-II
    // TODO: read https://pixorblog.wordpress.com/2019/08/14/curiously-recurring-template-pattern-crtp-in-depth/
    // TODO: learn about C++ templates
    //Island<moeoNSGAII,System<N_OBJECTIVES, N_TRAITS> > nsgaII_2(pop2, intPolicy_2, migPolicy_2, MAX_GEN, eval, xover, P_CROSS, mutation, P_MUT);
    Island<moeoNSGAII, System<N_OBJECTIVES, N_TRAITS>> nsgaII_2(
            pop2,             // Population
            intPolicy_2,      // Integration policy
            migPolicy_2,       // Migration policy
            continuator,      // Stopping criteria
            eval,             // Evaluation function
            transform         // Transformation operator combining crossover and mutation
    );

    // ISLAND 1
    // generate initial population
    eoPop<System<N_OBJECTIVES, N_TRAITS> > pop1(POP_SIZE, init);
    // // Emigration policy
    // // // Element 1
    eoPeriodicContinue<System<N_OBJECTIVES, N_TRAITS>> criteria_1(5);
    eoDetTournamentSelect<System<N_OBJECTIVES, N_TRAITS>> selectOne_1(25);
    eoSelectNumber<System<N_OBJECTIVES, N_TRAITS>> who_1(selectOne_1, 5);
    MigPolicy<System<N_OBJECTIVES, N_TRAITS>> migPolicy_1;
    migPolicy_1.push_back(PolicyElement<System<N_OBJECTIVES, N_TRAITS>>(who_1, criteria_1));
    // // Integration policy
    eoPlusReplacement<System<N_OBJECTIVES, N_TRAITS>> intPolicy_1;
    // build NSGA-II
    Island<moeoNSGAII, System<N_OBJECTIVES, N_TRAITS>> nsgaII_1(
            pop1,             // Population
            intPolicy_1,      // Integration policy
            migPolicy_1,       // Migration policy
            continuator,      // Stopping criteria
            eval,             // Evaluation function
            transform         // Transformation operator combining crossover and mutation
    );

    // Create the SMP wrapper for NSGA-II
    //unsigned int workersNb = 4; // Set the desired number of workers
    //paradiseo::smp::MWModel<moeoNSGAII, System<N_OBJECTIVES, N_TRAITS>> mw(workersNb, MAX_GEN, eval, xover, P_CROSS, mutation, P_MUT);


    // help
    make_help(parser);

    // Start a parallel evaluation on the population
    //mw.evaluate(pop);
    //nsgaII(pop);
    //std::cout << "Initial population :" << std::endl;
    //std::cout << pop << std::endl;

    model.add(nsgaII_1);
    model.add(nsgaII_2);

    model();

    pop1.sort();
    pop2.sort();

    // run the algo
    //nsgaII(pop); // serial
    //mw(pop);

    eoPop<System<N_OBJECTIVES, N_TRAITS> > pop1a = nsgaII_1.getPop();
    eoPop<System<N_OBJECTIVES, N_TRAITS> > pop2a = nsgaII_2.getPop();
    // extract first front of the final population using an moeoArchive (this is the output of nsgaII)
    moeoUnboundedArchive<System<N_OBJECTIVES, N_TRAITS> > arch;
    cout << "Arhive update 1: " << arch(pop1a) << endl;
    cout << "Arhive update 2: " << arch(pop2a) << endl;

    // printing of the final archive
    cout << "Final Archive" << endl;
    arch.sortedPrintOn(cout);
    cout << endl;

    std::map<std::string, double> parameters = {
            {"popSize", POP_SIZE},
            {"maxGen", MAX_GEN},
            {"mutEpsilon", M_EPSILON},
            {"pCross", P_CROSS},
            {"pMut", P_MUT},
            {"eta_c", eta_c},
            {"sigma", sigma},
            {"p_change", p_change}
    };

    // Save final archive to a CSV file
    std::ofstream csv_file;
    csv_file.open("pareto_frontier.csv");
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

    // Save all evaluated solutions to a CSV file
    std::ofstream all_solutions_file;
    all_solutions_file.open("all_evaluated_solutions.csv");
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

    return EXIT_SUCCESS;
}
