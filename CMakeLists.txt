cmake_minimum_required(VERSION 3.12)
#cmake_policy(SET CMP0144 NEW)
project(glyfada VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fopenmp")

# Detect system to set specific configurations
if(EXISTS "/global/cfs/cdirs")
	# Configuration if /global/cfs/cdirs exists
	# execute_process(COMMAND module load craype-x86-milan PrgEnv-gnu openmpi gcc)
	set(MPI_C_COMPILER "mpicc")
	set(MPI_CXX_COMPILER "mpicxx")
	set(BOOST_ROOT "/global/cfs/cdirs/m509/paradiseo/boost_1_84_0/bin")
	set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
	set(PARADISEO_ROOT "/global/cfs/cdirs/m509/paradiseo/paradiseo")
	set(CPP_REDIS_ROOT "/global/cfs/cdirs/m509/paradiseo/cpp_redis")
	message(STATUS "Configuring for Perlmutter")
elseif(EXISTS "/srv/software/el8/x86_64/eb/OpenMPI/4.1.5-GCC-12.3.0/bin/mpicc")
	# Configuration for wc2.fnal.gov and wc.fnal.gov
	set(MPI_C_COMPILER "/srv/software/el8/x86_64/eb/OpenMPI/4.1.5-GCC-12.3.0/bin/mpicc")
	set(MPI_CXX_COMPILER "/srv/software/el8/x86_64/eb/OpenMPI/4.1.5-GCC-12.3.0/bin/mpicxx")
	set(BOOST_ROOT "/work1/beamopt/paradiseo/boost_1_84_0/bin")
	set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
	set(PARADISEO_ROOT "/work1/beamopt/paradiseo/paradiseo")
	set(CPP_REDIS_ROOT "/work1/beamopt/paradiseo/cpp_redis")
	message(STATUS "Configuring for TeV cluster")
elseif(EXISTS "/psi/home")
	set(MPI_C_COMPILER "mpicc")
	set(MPI_CXX_COMPILER "mpicxx")
	set(BOOST_ROOT "/psi/home/valetov_e/paradiseo/boost")
	set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
	set(PARADISEO_ROOT "/psi/home/valetov_e/paradiseo/paradiseo")
	set(CPP_REDIS_ROOT "/psi/home/valetov_e/paradiseo/cpp_redis")
	message(STATUS "Configuring for Merlin")
else()
	# Default configuration
	set(MPI_C_COMPILER "/opt/openmpi-5.0.0/bin/mpicc")
	set(MPI_CXX_COMPILER "/opt/openmpi-5.0.0/bin/mpic++")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -g -fopenmp")
	set(CMAKE_BUILD_TYPE Debug)
	set(BOOST_ROOT "/opt/boost_1_84_0")
	set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
	if(EXISTS "/home/evaletov/paradiseo/paradiseo")
		set(PARADISEO_ROOT "/home/evaletov/paradiseo/paradiseo")
	elseif(EXISTS "/opt/paradiseo")
		set(PARADISEO_ROOT "/opt/paradiseo")
	else()
		message(FATAL_ERROR "None of the PARADISEO_ROOT directories exist.")
	endif()
	message(STATUS "Configuring for Generic system")
endif()

set(Boost_USE_STATIC_LIBS ON)
set(PARADISEO_LIB_DIR "${PARADISEO_ROOT}/build/lib" CACHE PATH "Path to Paradiseo library directory")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Boost 1.84 REQUIRED COMPONENTS filesystem system mpi serialization HINTS ${BOOST_ROOT})
find_package(OpenSSL REQUIRED)
find_library(EO_LIBRARY eo HINTS ${PARADISEO_LIB_DIR} REQUIRED)
find_library(EO_UTILS_LIBRARY eoutils HINTS ${PARADISEO_LIB_DIR} REQUIRED)
find_package(LibXml2 REQUIRED)
find_library(REDIS_LIBRARY cpp_redis HINTS "${CPP_REDIS_ROOT}/build/lib" REQUIRED)
find_library(TACOPIE_LIBRARY tacopie HINTS "${CPP_REDIS_ROOT}/build/lib" REQUIRED)

set(SOURCE_FILES
		run_cosy.cpp
		run_g4bl.h
		run_g4bl.cpp
		run_dh.cpp
		${PARADISEO_ROOT}/eo/src/utils/eoParallel.h
		${PARADISEO_ROOT}/eo/src/utils/eoParallel.cpp
		${PARADISEO_ROOT}/smp/src/topology/complete.h
		${PARADISEO_ROOT}/smp/src/topology/complete.cpp
		${PARADISEO_ROOT}/eo/src/mpi/eoMpiNode.h
		${PARADISEO_ROOT}/eo/src/mpi/eoMpiNode.cpp
		${PARADISEO_ROOT}/eo/src/mpi/eoMpi.h
		${PARADISEO_ROOT}/eo/src/mpi/eoMpi.cpp
		${PARADISEO_ROOT}/eo/src/mpi/implMpi.h
		${PARADISEO_ROOT}/eo/src/mpi/implMpi.cpp
		${PARADISEO_ROOT}/eo/src/serial/SerialObject.cpp
		${PARADISEO_ROOT}/eo/src/serial/Parser.cpp
		${PARADISEO_ROOT}/eo/src/serial/SerialArray.cpp
		${PARADISEO_ROOT}/eo/src/serial/SerialString.cpp
		${PARADISEO_ROOT}/moeo/src/core/moeoObjectiveVectorTraits.cpp
		mpi/SerializableBase.h
		mpi/schema.cpp
		mpi/xml_parser.cpp
		mpi/param.h
		mpi/param.cpp
		optimiser.cpp
		utils/Logging.h
		utils/Utilities.h
		utils/Utilities.cpp
		utils/Logging.cpp
        utils/json_utilities.h
)

add_executable(optimiser ${SOURCE_FILES})

target_include_directories(optimiser PRIVATE
#		/usr/include
#    	/usr/include/c++/11
#    	/usr/include/x86_64-linux-gnu/c++/11
		${CMAKE_CURRENT_SOURCE_DIR}
		${PARADISEO_ROOT}/eo/src/
		${PARADISEO_ROOT}/smp/src/topology
		${PARADISEO_ROOT}/eo/src/utils
		${PARADISEO_ROOT}/eo/src/serial
		${PARADISEO_ROOT}/eo/src/mpi
		${PARADISEO_ROOT}/mo/src/
		${PARADISEO_ROOT}/moeo/src/
		${PARADISEO_ROOT}/smp/src/
		${BOOST_ROOT}
		${CPP_REDIS_ROOT}/includes
		${CPP_REDIS_ROOT}/tacopie/includes
		${LIBXML2_INCLUDE_DIR}
		${OPENSSL_INCLUDE_DIR}
)

link_directories(${PARADISEO_LIB_DIR}
		${BOOST_LIBRARYDIR})

target_link_libraries(optimiser PRIVATE OpenMP::OpenMP_CXX ${EO_LIBRARY} ${EO_UTILS_LIBRARY} ${Boost_LIBRARIES} ${LIBXML2_LIBRARIES} ${REDIS_LIBRARY} ${TACOPIE_LIBRARY} OpenSSL::Crypto  stdc++fs)

install(TARGETS optimiser DESTINATION bin)

