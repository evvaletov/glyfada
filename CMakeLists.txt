cmake_minimum_required(VERSION 3.12)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(glyfada VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(MPI_CC_INCLUDE_PATH "/opt/intel/oneapi/mpi/latest/include")
#set(MPI_CC_LIBRARIES "/opt/intel/oneapi/mpi/latest/lib/release/libmpi.so")
#set(ENV{I_MPI_SUBSTITUTE_INSTALLDIR} "/opt/intel/oneapi/mpi/latest")
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/intel/oneapi/mpi/latest")
set(CMAKE_CXX_FLAGS "-fopenmp")
set(BOOST_ROOT "/opt/boost_1_84_0")
set(Boost_USE_STATIC_LIBS ON)

find_package(OpenMP REQUIRED)
find_package(MPI)
find_package(Boost 1.84 REQUIRED COMPONENTS filesystem system mpi serialization)
find_library(EO_LIBRARY eo HINTS /home/evaletov/paradiseo/gitclone/build/lib REQUIRED)
find_library(EO_UTILS_LIBRARY eoutils HINTS /home/evaletov/paradiseo/gitclone/build/lib REQUIRED)

set(PARADISEO_PATH "/home/evaletov/paradiseo/gitclone")

add_executable(optimiser run_cosy.cpp run_g4bl.h run_g4bl.cpp
		run_dh.cpp
		/home/evaletov/paradiseo/gitclone/eo/src/utils/eoParallel.h
		/home/evaletov/paradiseo/gitclone/eo/src/utils/eoParallel.cpp
		/home/evaletov/paradiseo/gitclone/smp/src/topology/complete.cpp
		/home/evaletov/paradiseo/gitclone/eo/src/mpi/eoMpiNode.h
		/home/evaletov/paradiseo/gitclone/eo/src/mpi/eoMpiNode.cpp
		/home/evaletov/paradiseo/gitclone/eo/src/mpi/eoMpi.h
		/home/evaletov/paradiseo/gitclone/eo/src/mpi/eoMpi.cpp
		/home/evaletov/paradiseo/gitclone/eo/src/mpi/implMpi.h
		/home/evaletov/paradiseo/gitclone/eo/src/mpi/implMpi.cpp
		/home/evaletov/paradiseo/gitclone/eo/src/serial/SerialObject.cpp
		/home/evaletov/paradiseo/gitclone/eo/src/serial/Parser.cpp
		/home/evaletov/paradiseo/gitclone/eo/src/serial/SerialArray.cpp
		/home/evaletov/paradiseo/gitclone/eo/src/serial/SerialString.cpp
		optimiser_homogenous_2.cpp
		logging.h)

target_include_directories(optimiser PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
		${PARADISEO_PATH}/eo/src/
		${PARADISEO_PATH}/eo/src/utils
		${PARADISEO_PATH}/eo/src/mpi
		${PARADISEO_PATH}/eo/src/serial
		${PARADISEO_PATH}/mo/src/
		${PARADISEO_PATH}/moeo/src/
		${PARADISEO_PATH}/smp/src/
		${BOOST_ROOT}
)

link_directories(/home/evaletov/paradiseo/gitclone/build/lib
		${BOOST_ROOT}/lib)

#target_link_libraries(optimiser PRIVATE OpenMP::OpenMP_CXX ${EO_LIBRARY} ${EO_UTILS_LIBRARY} pthread ${Boost_LIBRARIES} MPI::MPI_CXX )
target_link_libraries(optimiser PRIVATE OpenMP::OpenMP_CXX ${EO_LIBRARY} ${EO_UTILS_LIBRARY} ${Boost_LIBRARIES} MPI::MPI_CXX stdc++fs)
#target_link_libraries(optimiser PRIVATE OpenMP::OpenMP_CXX ${EO_LIBRARY} ${EO_UTILS_LIBRARY} MPI::MPI_CXX stdc++fs)
