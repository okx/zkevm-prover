#ifndef CHELPERS_STEPS_GPU_CUH
#define CHELPERS_STEPS_GPU_CUH

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
#include "chelpers_steps_pack.hpp"
#include <cuda_runtime.h>
const uint64_t MAX_GPU = 8;
class gl64_t;
class CHelpersStepsPackGPU: public CHelpersStepsPack {
public:

    int nDevices = 0;
    int64_t nCudaThreads = 1;
    int groupIdx = 0;

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

    // ==== same on each device ====
    uint64_t *nColsStages_d[MAX_GPU];
    uint64_t *nColsStagesAcc_d[MAX_GPU];
    uint64_t *offsetsStages_d[MAX_GPU];

    uint8_t *ops_d[MAX_GPU];  //TODO change to uint32_t or uint64_t?
    uint16_t *args_d[MAX_GPU];
    //uint8_t *storePols_d[MAX_GPU];

    gl64_t *challenges_d[MAX_GPU];
    gl64_t *challenges_ops_d[MAX_GPU];
    gl64_t *numbers_d[MAX_GPU];
    gl64_t *publics_d[MAX_GPU];
    gl64_t *evals_d[MAX_GPU];
    // =================================

    // ==== different on each device ====
    gl64_t *constPols_d[MAX_GPU];
    gl64_t *x_d[MAX_GPU];
    gl64_t *zi_d[MAX_GPU];
    gl64_t *pols_d[MAX_GPU];
    gl64_t *xDivXSubXi_d[MAX_GPU];

    gl64_t *gBufferT_[MAX_GPU];
    gl64_t *tmp1_d[MAX_GPU];
    gl64_t *tmp3_d[MAX_GPU];
    // =================================

    // three streams on each device
    // one for copying data from Host to Device
    // one for calculating on Device
    // one for copying data from Device to Host
    cudaStream_t gpu_stream[MAX_GPUS*3];

    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t rowIni, uint64_t rowEnd);
    void prepareGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void compare(StepsParams &params, uint64_t row);
    void cleanupGPU();

    void loadData(StarkInfo &starkInfo, StepsParams &params, uint64_t row);
    void storeData(StepsParams &params, uint64_t row);
};

__global__ void loadPolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps, uint64_t nConstants, uint64_t stage, uint32_t deviceIdx);
__global__ void storePolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps, uint32_t deviceIdx);
__global__ void pack_kernel(CHelpersStepsPackGPU *cHelpersSteps, uint32_t deviceIdx);

#endif
#endif
