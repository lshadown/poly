cmake_minimum_required(VERSION 3.15)
project(poly C)

set(CMAKE_C_STANDARD 99)

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()


add_executable(poly main.c "polybench.c" polybench.h cholesky/cholesky.c cholesky/cholesky.h gemver/gemver.c gemver/gemver.h symm/symm.c symm/symm.h syr2k/syr2k.c syr2k/syr2k.h 2mm/2mm.c 2mm/2mm.h)
target_link_libraries(poly m)
