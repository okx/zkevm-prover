#ifndef CHELPERS_STEPS_GPU_CUH
#define CHELPERS_STEPS_GPU_CUH

#include "chelpers_steps_pack.hpp"
#include "gl64_t.cuh"

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
class CHelpersStepsPackGPU: public CHelpersStepsPack {
public:
    uint8 *ops_d;
    uint16_t *args_d;

    gl64_t *challenges_d;
    gl64_t *challenges_ops_d;

    gl64_t *numbers_d;
    gl64_t *publics_d;
    gl64_t *evals_d;

    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t rowIni, uint64_t rowEnd);
    void prepareGPU();
    void cleanupGPU();
};

#endif
#endif
