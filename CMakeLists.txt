cmake_minimum_required(VERSION 3.12)
project(test VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(Python3 COMPONENTS Development)

include_directories(
		${CMAKE_CURRENT_SOURCE_DIR}
		/home/evaletov/paradiseo/gitclone/eo/src/
		/home/evaletov/paradiseo/gitclone/mo/src/
		/home/evaletov/paradiseo/gitclone/moeo/src/
		/home/evaletov/paradiseo/gitclone/smp/src/
		${Boost_INCLUDE_DIRS}
		${Python3_INCLUDE_DIRS}
)

link_directories(/home/evaletov/paradiseo/gitclone/build/lib)

add_executable(optimiser optimiser.cpp run_cosy.cpp run_g4bl.h run_g4bl.cpp
		run_dh.cpp)

target_link_libraries(optimiser
		PRIVATE
		OpenMP::OpenMP_CXX
		eo
		eoutils
		pthread
		${Boost_LIBRARIES}
		${Python3_LIBRARIES}
		)
