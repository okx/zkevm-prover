#ifndef CHELPERS_STEPS_GPU_CUH
#define CHELPERS_STEPS_GPU_CUH

#include "chelpers_steps_pack.hpp"

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
class gl64_t;
class CHelpersStepsPackGPU: public CHelpersStepsPack {
public:

    uint64_t *nColsStagesAcc_d;

    uint8_t *ops_d;
    uint16_t *args_d;

    gl64_t *challenges_d;
    gl64_t *challenges_ops_d;
    gl64_t *numbers_d;
    gl64_t *publics_d;
    gl64_t *evals_d;

    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t rowIni, uint64_t rowEnd);
    void prepareGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void cleanupGPU();
};

__global__ void pack_kernel(uint64_t nrowsPack,
                            uint32_t nOps,
                            uint32_t nArgs,
                            gl64_t *tmp1,
                            gl64_t *tmp3,
                            uint64_t *nColsStagesAcc,
                            uint8_t *ops,
                            uint16_t *args,
                            gl64_t *bufferT_,
                            gl64_t *challenges,
                            gl64_t *challenges_ops,
                            gl64_t *numbers,
                            gl64_t *publics,
                            gl64_t *evals);

#endif
#endif
