#include <GASPI.h>
#include <GASPI_Ext.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

#ifndef GASPIHELPER
#define GASPIHELPER

#define SUCCESS_OR_DIE(f...)                                            \
do                                                                      \
{                                                                       \
    const gaspi_return_t r = f;                                         \
                         						\
    if (r != GASPI_SUCCESS)                                             \
    {                                          				\
        std::cout << std::getenv("HOSTNAME") <<" "<<r<< " '" << gaspi_error_str(r) << "' ASGD FAIL at " << __FILE__<< " "<<__LINE__<<std::endl;   \
        fflush (stdout);                                                \
                                                                        \
        exit (EXIT_FAILURE);                                            \
    }                                                                   \
}while (0)

#endif


#define SUCCESS_OR_LOG(f...)                                            \
do                                                                      \
{                                                                       \
    const gaspi_return_t r = f;                                         \
                                   					\
    if (r != GASPI_SUCCESS)                                             \
    {                                                                   \
        std::cout << std::getenv("HOSTNAME") <<" "<<r<< " ASGD FAIL at " << __FILE__<< " "<<__LINE__ <<std::endl;       \
    }                                                                   \
}while (0)
