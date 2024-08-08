#include "zklog.hpp"
#include <inttypes.h>

//#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE) && defined(__YXQ__)

#include "chelpers_steps_pack.cuh"
#include "goldilocks_cubic_extension.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"

void CHelpersStepsPackGPU::prepareGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {

    Goldilocks::Element challenges[params.challenges.degree()*FIELD_EXTENSION*nrowsPack];
    Goldilocks::Element challenges_ops[params.challenges.degree()*FIELD_EXTENSION*nrowsPack];
    for(uint64_t i = 0; i < params.challenges.degree(); ++i) {
        for(uint64_t j = 0; j < nrowsPack; ++j) {
            challenges[(i*FIELD_EXTENSION)*nrowsPack + j] = params.challenges[i][0];
            challenges[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.challenges[i][1];
            challenges[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.challenges[i][2];
            challenges_ops[(i*FIELD_EXTENSION)*nrowsPack + j] = params.challenges[i][0] + params.challenges[i][1];
            challenges_ops[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.challenges[i][0] + params.challenges[i][2];
            challenges_ops[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.challenges[i][1] + params.challenges[i][2];
        }
    }

    Goldilocks::Element numbers_[parserParams.nNumbers*nrowsPack];
    for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
        for(uint64_t j = 0; j < nrowsPack; ++j) {
            numbers_[i*nrowsPack + j] = Goldilocks::fromU64(numbers[i]);
        }
    }

    Goldilocks::Element publics[starkInfo.nPublics*nrowsPack];
    for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
        for(uint64_t j = 0; j < nrowsPack; ++j) {
            publics[i*nrowsPack + j] = params.publicInputs[i];
        }
    }

    Goldilocks::Element evals[params.evals.degree()*FIELD_EXTENSION*nrowsPack];
    for(uint64_t i = 0; i < params.evals.degree(); ++i) {
        for(uint64_t j = 0; j < nrowsPack; ++j) {
            evals[(i*FIELD_EXTENSION)*nrowsPack + j] = params.evals[i][0];
            evals[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.evals[i][1];
            evals[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.evals[i][2];
        }
    }

    CHECKCUDAERR(cudaMalloc(&ops_d, parserArgs.nOps * sizeof(uint8)));
    CHECKCUDAERR(cudaMemcpy(ops_d, parserArgs.ops, parserArgs.nOps * sizeof(uint8), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&arg_d, parserArgs.nArgs * sizeof(uint16_t)));
    CHECKCUDAERR(cudaMemcpy(arg_d, parserArgs.args, parserArgs.nArgs * sizeof(uint16_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&challenges_d, params.challenges.degree()*FIELD_EXTENSION*nrowsPack * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(challenges_d, challenges, params.challenges.degree()*FIELD_EXTENSION*nrowsPack * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&challenges_ops_d, params.challenges.degree()*FIELD_EXTENSION*nrowsPack * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(challenges_ops_d, challenges_ops, params.challenges.degree()*FIELD_EXTENSION*nrowsPack * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&numbers_d, parserParams.nNumbers*nrowsPack * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(numbers_d, numbers, parserParams.nNumbers*nrowsPack * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&publics_d, starkInfo.nPublics*nrowsPack * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(publics_d, publics, starkInfo.nPublics*nrowsPack * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&evals_d, params.evals.degree()*FIELD_EXTENSION*nrowsPack * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(evals_d, evals, params.challenges.degree()*FIELD_EXTENSION*nrowsPack * sizeof(uint64_t), cudaMemcpyHostToDevice));
}

void CHelpersStepsPackGPU::calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {




    bool domainExtended = parserParams.stage > 3 ? true : false;
    uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    calculateExpressionsRowsGPU(starkInfo, params, parserArgs, parserParams, 0, domainSize);
}

void CHelpersStepsPackGPU::calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams,
    uint64_t rowIni, uint64_t rowEnd){

    bool domainExtended = parserParams.stage > 3 ? true : false;
    uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];
    uint16_t *args = &parserArgs.args[parserParams.argsOffset];
    uint64_t *numbers = &parserArgs.numbers[parserParams.numbersOffset];
    uint8_t *storePol = &parserArgs.storePols[parserParams.storePolsOffset];

    if(rowEnd < rowIni || rowEnd > domainSize) {
        zklog.info("Invalid range for rowIni and rowEnd");
        exitProcess();
    }
    if((rowEnd -rowIni) % nrowsPack != 0) {
       nrowsPack = 1;
    }

    setBufferTInfo(starkInfo, parserParams.stage);

    Goldilocks::Element bufferT_[2*nCols*nrowsPack];



}

__global__ void pack_kernel() {

}

//#endif