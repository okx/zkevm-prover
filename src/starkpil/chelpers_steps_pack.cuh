#ifndef CHELPERS_STEPS_GPU_CUH
#define CHELPERS_STEPS_GPU_CUH

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)

#include "chelpers_steps_pack.hpp"

class CHelpersStepsPackGPU: public CHelpersStepsPack {
public:
    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t rowIni, uint64_t rowEnd);
    void prepareGPU();
    void cleanupGPU();
};

#endif
#endif
