cmake_minimum_required(VERSION 3.12)
project(glyfada VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fopenmp")

# Detect system to set specific configurations
if(EXISTS "/srv/software/el8/x86_64/eb/OpenMPI/4.1.5-GCC-12.3.0/bin/mpicc")
	# Configuration for wc2.fnal.gov and wc.fnal.gov
	set(MPI_C_COMPILER "/srv/software/el8/x86_64/eb/OpenMPI/4.1.5-GCC-12.3.0/bin/mpicc")
	set(MPI_CXX_COMPILER "/srv/software/el8/x86_64/eb/OpenMPI/4.1.5-GCC-12.3.0/bin/mpicxx")
	set(BOOST_ROOT "/work1/beamopt/paradiseo/boost_1_84_0/bin")
	set(PARADISEO_ROOT "/work1/beamopt/paradiseo/paradiseo")
else()
	# Default configuration
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -g -fopenmp")
	set(CMAKE_BUILD_TYPE Debug)
	set(BOOST_ROOT "/opt/boost_1_84_0")
	set(PARADISEO_ROOT "/home/evaletov/paradiseo/gitclone")
endif()

set(Boost_USE_STATIC_LIBS ON)
set(PARADISEO_LIB_DIR "${PARADISEO_ROOT}/build/lib" CACHE PATH "Path to Paradiseo library directory")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Boost 1.84 REQUIRED COMPONENTS filesystem system mpi serialization)
find_library(EO_LIBRARY eo HINTS ${PARADISEO_LIB_DIR} REQUIRED)
find_library(EO_UTILS_LIBRARY eoutils HINTS ${PARADISEO_LIB_DIR} REQUIRED)
find_package(LibXml2 REQUIRED)

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
		# Add any additional source files here...
)

add_executable(optimiser ${SOURCE_FILES})

target_include_directories(optimiser PRIVATE
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
		${LIBXML2_INCLUDE_DIR}
)

link_directories(${PARADISEO_LIB_DIR}
		${BOOST_ROOT}/bin/lib)

target_link_libraries(optimiser PRIVATE OpenMP::OpenMP_CXX ${EO_LIBRARY} ${EO_UTILS_LIBRARY} ${Boost_LIBRARIES} ${LIBXML2_LIBRARIES} MPI::MPI_CXX stdc++fs)

