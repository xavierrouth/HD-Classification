#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#define DUMP_HV_DEF(ELEMTY) \
extern "C" void cu_rt_dump_##ELEMTY##_hv(void *hv, size_t row, const char * filename) {\
    std::ofstream file(filename);\
    for (size_t i = 0; i < row; ++i) {\
        file << ((ELEMTY*)hv)[i] << "\n";\
    }\
}

DUMP_HV_DEF(int);
DUMP_HV_DEF(float);
DUMP_HV_DEF(double);

#define DUMP_HM_DEF(ELEMTY) \
extern "C" void cu_rt_dump_##ELEMTY##_hm(void *hm, size_t row, size_t col, const char * filename) {\
    std::ofstream file(filename);\
    for (size_t i = 0; i < row; ++i) {\
        for (size_t j = 0; j < col; ++j) {\
            file << ((ELEMTY*)hm)[i * col + j] << "\n";\
        }\
        file << "\n";\
    }\
}

DUMP_HM_DEF(int);
DUMP_HM_DEF(float);
DUMP_HM_DEF(double);
