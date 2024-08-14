#ifndef CHELPERS_STEPS_GPU_CUH
#define CHELPERS_STEPS_GPU_CUH

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
#include "chelpers_steps_pack.hpp"
#include <cuda_runtime.h>
class gl64_t;
class CHelpersStepsPackGPU: public CHelpersStepsPack {
public:

    int64_t nCudaThreads;

    bool domainExtended;
    uint64_t domainSize;
    uint64_t subDomainSize;
    uint64_t nextStride;
    uint32_t nOps;
    uint32_t nArgs;
    uint32_t nStorePols;
    uint64_t nBufferT;
    uint64_t nTemp1;
    uint64_t nTemp3;
    uint64_t nPols;

    vector<uint64_t> offsetsStagesGPU;

    uint64_t *nColsStages_d;
    uint64_t *nColsStagesAcc_d;
    uint64_t *offsetsStages_d;

    uint8_t *ops_d;
    uint16_t *args_d;
    uint8_t *storePols_d;

    gl64_t *challenges_d;
    gl64_t *challenges_ops_d;
    gl64_t *numbers_d;
    gl64_t *publics_d;
    gl64_t *evals_d;

    gl64_t *constPols_d;
    gl64_t *x_d;
    gl64_t *zi_d;
    gl64_t *pols_d;
    gl64_t *xDivXSubXi_d;

    gl64_t *gBufferT_;
    gl64_t *tmp1_d;
    gl64_t *tmp3_d;

    Goldilocks::Element *tmpExp;

    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t rowIni, uint64_t rowEnd);
    void prepareGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void compare(StepsParams &params, uint64_t row);
    void cleanupGPU();

    void loadData(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t stage);
    void storeData(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t stage);
};

__global__ void loadPolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps, uint64_t nConstants, uint64_t stage);
__global__ void storePolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps);
__global__ void pack_kernel(CHelpersStepsPackGPU *cHelpersSteps);

#endif
#endif
