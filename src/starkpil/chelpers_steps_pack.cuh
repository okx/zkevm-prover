#ifndef CHELPERS_STEPS_GPU_CUH
#define CHELPERS_STEPS_GPU_CUH

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
#include "chelpers_steps_pack.hpp"
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

    uint8_t* ops2;
    uint16_t* args2;

    vector<uint64_t> nColsStagesAcc2;
    vector<Goldilocks::Element> challenges2;
    vector<Goldilocks::Element> challenges_ops2;
    vector<Goldilocks::Element> numbers_2;
    vector<Goldilocks::Element> publics2;
    vector<Goldilocks::Element> evals2;

    Goldilocks::Element *pBuffer;

    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t rowIni, uint64_t rowEnd);
    void prepareGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void cleanupGPU();
};

#endif
#endif
