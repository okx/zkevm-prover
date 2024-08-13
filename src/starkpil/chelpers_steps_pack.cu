#include "zklog.hpp"
#include <inttypes.h>

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)

#include "chelpers_steps_pack.cuh"
#include "goldilocks_cubic_extension.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "timer.hpp"

void CHelpersStepsPackGPU::prepareGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {

    prepare(starkInfo, params, parserArgs, parserParams);

    CHECKCUDAERR(cudaMalloc(&nColsStages_d, nColsStages.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(nColsStages_d, nColsStages.data(), nColsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&nColsStagesAcc_d, nColsStagesAcc.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(nColsStagesAcc_d, nColsStagesAcc.data(), nColsStagesAcc.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&ops_d, nOps * sizeof(uint8_t)));
    CHECKCUDAERR(cudaMemcpy(ops_d, &parserArgs.ops[parserParams.opsOffset], nOps * sizeof(uint8_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&args_d, nArgs * sizeof(uint16_t)));
    CHECKCUDAERR(cudaMemcpy(args_d, &parserArgs.args[parserParams.argsOffset], nArgs * sizeof(uint16_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&storePols_d, nStorePols * sizeof(uint8_t)));
    CHECKCUDAERR(cudaMemcpy(storePols_d, &parserArgs.storePols[parserParams.storePolsOffset], nStorePols * sizeof(uint8_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&challenges_d, challenges.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(challenges_d, challenges.data(), challenges.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&challenges_ops_d, challenges_ops.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(challenges_ops_d, challenges_ops.data(), challenges_ops.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&numbers_d, numbers_.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(numbers_d, numbers_.data(), numbers_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&publics_d, publics.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(publics_d, publics.data(), publics.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&evals_d, evals.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(evals_d, evals.data(), evals.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    offsetsStagesGPU.resize(12);
    uint64_t total_offsets = 0;
    for (uint64_t s = 1; s < 11; s++) {
        if (s < 4 || (s == 4 && parserParams.stage != 4) || (s == 10 && domainExtended)) {
            offsetsStagesGPU[s] = int64_t(total_offsets);
            total_offsets += nColsStages[s] * nrowsPack * nCudaThreads;
        } else {
            offsetsStagesGPU[s] = -1;
        }
    }

    CHECKCUDAERR(cudaMalloc(&offsetsStages_d, offsetsStagesGPU.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(offsetsStages_d, offsetsStagesGPU.data(), offsetsStagesGPU.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&constPols_d, starkInfo.nConstants * (nrowsPack * nCudaThreads + 2) * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&x_d, nrowsPack * nCudaThreads * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&zi_d, nrowsPack * nCudaThreads * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&pols_d, total_offsets * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&xDivXSubXi_d, 2 * nrowsPack * nCudaThreads * sizeof(uint64_t)));

    CHECKCUDAERR(cudaMalloc(&gBufferT_, nBufferT * nCudaThreads * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&tmp1_d, nTemp1 * nCudaThreads * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&tmp3_d, nTemp3 * nCudaThreads * sizeof(uint64_t)*nCudaThreads));
}

void CHelpersStepsPackGPU::cleanupGPU() {
    cudaFree(nColsStages_d);
    cudaFree(nColsStagesAcc_d);
    cudaFree(offsetsStages_d);

    cudaFree(ops_d);
    cudaFree(args_d);
    cudaFree(storePols_d);

    cudaFree(challenges_d);
    cudaFree(challenges_ops_d);
    cudaFree(numbers_d);
    cudaFree(publics_d);
    cudaFree(evals_d);

    cudaFree(constPols_d);
    cudaFree(x_d);
    cudaFree(zi_d);
    cudaFree(pols_d);
    cudaFree(xDivXSubXi_d);

    cudaFree(gBufferT_);
    cudaFree(tmp1_d);
    cudaFree(tmp3_d);
}

void CHelpersStepsPackGPU::calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {

    nCudaThreads = 1 << 12;
    domainExtended = parserParams.stage > 3 ? true : false;
    domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    nextStride = domainExtended ? 1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;

    nOps = parserParams.nOps;
    nArgs = parserParams.nArgs;
    nStorePols = parserParams.nStorePols;
    nBufferT = 2*nCols*nrowsPack;
    nTemp1 = parserParams.nTemp1*nrowsPack;
    nTemp3 = parserParams.nTemp3*FIELD_EXTENSION*nrowsPack;

    CHECKCUDAERR(cudaSetDevice(0));

    prepareGPU(starkInfo, params, parserArgs, parserParams);
    calculateExpressionsRowsGPU(starkInfo, params, parserArgs, parserParams, 0, domainSize);
    cleanupGPU();
}

void CHelpersStepsPackGPU::calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams,
    uint64_t rowIni, uint64_t rowEnd){

    if(rowEnd < rowIni || rowEnd > domainSize || (rowEnd -rowIni) % nrowsPack != 0) {
        zklog.info("Invalid range for rowIni " + to_string(rowIni) + " and rowEnd " + to_string(rowEnd));
        exitProcess();
    }

    printf("nCols:%lu\n", nCols);
    printf("nrowsPack:%lu\n", nrowsPack);
    printf("buffer:%lu\n", nBufferT);

    CHelpersStepsPackGPU *cHelpersSteps_d;
    CHECKCUDAERR(cudaMalloc((void **)&(cHelpersSteps_d), sizeof(CHelpersStepsPackGPU)));
    CHECKCUDAERR(cudaMemcpy(cHelpersSteps_d, this, sizeof(CHelpersStepsPackGPU), cudaMemcpyHostToDevice));

    for (uint64_t i = rowIni; i < rowEnd; i+= nrowsPack*nCudaThreads) {
        printf("rows:%lu\n", i);
        loadData(starkInfo, params, i, parserParams.stage);
        loadPolinomialsGPU<<<(nCudaThreads+15)/16,16>>>(cHelpersSteps_d, starkInfo.nConstants, i, parserParams.stage);
        return;
    }

    cudaFree(cHelpersSteps_d);
}

void CHelpersStepsPackGPU::loadData(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t stage) {

    ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
    Polinomial &x = domainExtended ? params.x_2ns : params.x_n;

    // TODO may overflow and cycle
    CHECKCUDAERR(cudaMemcpy(constPols_d, (Goldilocks::Element *)constPols->address() + row * starkInfo.nConstants, starkInfo.nConstants * (nrowsPack * nCudaThreads + nextStride) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(x_d, x[row], nrowsPack * nCudaThreads * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(zi_d, params.zi[row], nrowsPack * nCudaThreads * sizeof(uint64_t), cudaMemcpyHostToDevice));

    for (uint64_t s = 1; s < 11; s++) {
        if (offsetsStagesGPU[s] >= 0) {
            CHECKCUDAERR(cudaMemcpy(pols_d + offsetsStagesGPU[s], &params.pols[offsetsStages[s] + row*nColsStages[s]], nrowsPack * nCudaThreads *nColsStages[s] * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
    }

    CHECKCUDAERR(cudaMemcpy(xDivXSubXi_d, params.xDivXSubXi[row], nrowsPack * nCudaThreads *FIELD_EXTENSION * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(xDivXSubXi_d + nrowsPack * nCudaThreads *FIELD_EXTENSION, params.xDivXSubXi[domainSize + row], nrowsPack * nCudaThreads *FIELD_EXTENSION * sizeof(uint64_t), cudaMemcpyHostToDevice));
}

__global__ void loadPolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps, uint64_t nConstants, uint64_t row, uint64_t stage) {

    uint64_t nCudaThreads = cHelpersSteps->nCudaThreads;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCudaThreads) {
        return;
    }

    uint64_t nrowsPack = cHelpersSteps->nrowsPack;
    uint64_t nextStride = cHelpersSteps->nextStride;
    uint64_t domainSize = cHelpersSteps->domainSize;
    uint64_t nBufferT = cHelpersSteps->nBufferT;

    uint64_t *nColsStages = cHelpersSteps->nColsStages_d;
    uint64_t *nColsStagesAcc = cHelpersSteps->nColsStagesAcc_d;
    uint64_t *offsetsStages = cHelpersSteps->offsetsStages_d;

    gl64_t *bufferT_ = cHelpersSteps->gBufferT_ + idx * nBufferT;
    gl64_t *pols_d = cHelpersSteps->pols_d;
    gl64_t *constPols_d = cHelpersSteps->constPols_d;

    row = row % (nrowsPack * nCudaThreads);
    row = row + idx*nrowsPack;
    uint64_t nStages = 3;
    uint64_t nextStrides[2] = {0, nextStride};

    for(uint64_t k = 0; k < nConstants; ++k) {
        for(uint64_t o = 0; o < 2; ++o) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                uint64_t l = (row + j + nextStrides[o]) % domainSize;
                bufferT_[(nColsStagesAcc[5*o] + k)*nrowsPack + j] = constPols_d[l * nConstants + k];
            }
        }
    }

    // Load x and Zi
    for(uint64_t j = 0; j < nrowsPack; ++j) {
        bufferT_[nConstants*nrowsPack + j] = cHelpersSteps->x_d[row + j];
    }
    for(uint64_t j = 0; j < nrowsPack; ++j) {
        bufferT_[(nConstants + 1)*nrowsPack + j] = cHelpersSteps->zi_d[row + j];
    }

    for(uint64_t s = 1; s <= nStages; ++s) {
        for(uint64_t k = 0; k < nColsStages[s]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT_[(nColsStagesAcc[5*o + s] + k)*nrowsPack + j] = pols_d[offsetsStages[s] + l * nColsStages[s] + k];
                }
            }
        }
    }

    if(stage == 5) {
        for(uint64_t k = 0; k < nColsStages[nStages + 1]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT_[(nColsStagesAcc[5*o + nStages + 1] + k)*nrowsPack + j] = pols_d[offsetsStages[nStages + 1] + l * nColsStages[nStages + 1] + k];
                }
            }
        }

       // Load xDivXSubXi & xDivXSubWXi
       for(uint64_t d = 0; d < 2; ++d) {
           for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
               for(uint64_t j = 0; j < nrowsPack; ++j) {
                  bufferT_[(nColsStagesAcc[11] + FIELD_EXTENSION*d + i)*nrowsPack + j] = cHelpersSteps->xDivXSubXi_d[(d*domainSize + row + j) * FIELD_EXTENSION + i];
               }
           }
       }
    }
}


__global__ void storePolinomialsGPU() {

}

__global__ void pack_kernel(uint64_t nrowsPack,
                            uint32_t nOps,
                            uint32_t nArgs,
                            uint64_t nBufferT,
                            uint64_t nTemp1,
                            uint64_t nTemp3,
                            gl64_t *tmp1,
                            gl64_t *tmp3,
                            uint64_t *nColsStagesAcc,
                            uint8_t *ops,
                            uint16_t *args,
                            gl64_t *bufferT_,
                            gl64_t *challenges,
                            gl64_t *challenges_ops,
                            gl64_t *numbers_,
                            gl64_t *publics,
                            gl64_t *evals)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0) {
        return;
    }

    bufferT_ = bufferT_ + nBufferT*idx;
    tmp1 = tmp1 + nTemp1*idx;
    tmp3 = tmp3 + nTemp3*idx;

    uint64_t i_args = 0;

    for (uint64_t kk = 0; kk < nOps; ++kk) {
        switch (ops[kk]) {
            case 0: {
                // COPY commit1 to commit1
                gl64_t::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack]);
                i_args += 4;
                break;
            }
            case 1: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack]);
                i_args += 7;
                break;
            }
            case 2: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &tmp1[args[i_args + 5] * nrowsPack]);
                i_args += 6;
                break;
            }
            case 3: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &publics[args[i_args + 5] * nrowsPack]);
                i_args += 6;
                break;
            }
            case 4: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &numbers_[args[i_args + 5]*nrowsPack]);
                i_args += 6;
                break;
            }
            case 5: {
                // COPY tmp1 to commit1
                gl64_t::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack]);
                i_args += 3;
                break;
            }
            case 6: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 7: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 8: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 9: {
                // COPY public to commit1
                gl64_t::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &publics[args[i_args + 2] * nrowsPack]);
                i_args += 3;
                break;
            }
            case 10: {
                // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &publics[args[i_args + 3] * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 11: {
                // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &publics[args[i_args + 3] * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 12: {
                // COPY number to commit1
                gl64_t::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &numbers_[args[i_args + 2]*nrowsPack]);
                i_args += 3;
                break;
            }
            case 13: {
                // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &numbers_[args[i_args + 3]*nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 14: {
                // COPY commit1 to tmp1
                gl64_t::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack]);
                i_args += 3;
                break;
            }
            case 15: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                i_args += 6;
                break;
            }
            case 16: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 17: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 18: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 19: {
                // COPY tmp1 to tmp1
                gl64_t::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &tmp1[args[i_args + 1] * nrowsPack]);
                i_args += 2;
                break;
            }
            case 20: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 21: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 22: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                i_args += 4;
                break;
            }
            case 23: {
                // COPY public to tmp1
                gl64_t::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &publics[args[i_args + 1] * nrowsPack]);
                i_args += 2;
                break;
            }
            case 24: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2] * nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 25: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2] * nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                i_args += 4;
                break;
            }
            case 26: {
                // COPY number to tmp1
                gl64_t::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &numbers_[args[i_args + 1]*nrowsPack]);
                i_args += 2;
                break;
            }
            case 27: {
                // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &numbers_[args[i_args + 2]*nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                i_args += 4;
                break;
            }
            case 28: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack]);
                i_args += 7;
                break;
            }
            case 29: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &tmp1[args[i_args + 5] * nrowsPack]);
                i_args += 6;
                break;
            }
            case 30: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &publics[args[i_args + 5] * nrowsPack]);
                i_args += 6;
                break;
            }
            case 31: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &numbers_[args[i_args + 5]*nrowsPack]);
                i_args += 6;
                break;
            }
            case 32: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                i_args += 6;
                break;
            }
            case 33: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &tmp1[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 34: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 35: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 36: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                i_args += 6;
                break;
            }
            case 37: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 38: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 39: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 40: {
                // COPY commit3 to commit3
                Goldilocks3GPU::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack]);
                i_args += 4;
                break;
            }
            case 41: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack]);
                i_args += 7;
                break;
            }
            case 42: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &tmp3[args[i_args + 5] * nrowsPack * FIELD_EXTENSION]);
                i_args += 6;
                break;
            }
            case 43: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &challenges[args[i_args + 5]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 5]*FIELD_EXTENSION*nrowsPack]);
                i_args += 6;
                break;
            }
            case 44: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &challenges[args[i_args + 5]*FIELD_EXTENSION*nrowsPack]);
                i_args += 6;
                break;
            }
            case 45: {
                // COPY tmp3 to commit3
                Goldilocks3GPU::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION]);
                i_args += 3;
                break;
            }
            case 46: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 4] * nrowsPack * FIELD_EXTENSION]);
                i_args += 5;
                break;
            }
            case 47: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 48: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 49: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 50: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 51: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                i_args += 6;
                break;
            }
            case 52: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 53: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 54: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 55: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                i_args += 5;
                break;
            }
            case 56: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &tmp1[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 57: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &publics[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 58: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &numbers_[args[i_args + 3]*nrowsPack]);
                i_args += 4;
                break;
            }
            case 59: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                i_args += 5;
                break;
            }
            case 60: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &tmp1[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 61: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 62: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                i_args += 4;
                break;
            }
            case 63: {
                // COPY commit3 to tmp3
                Goldilocks3GPU::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack]);
                i_args += 3;
                break;
            }
            case 64: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                i_args += 6;
                break;
            }
            case 65: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp3[args[i_args + 4] * nrowsPack * FIELD_EXTENSION]);
                i_args += 5;
                break;
            }
            case 66: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 67: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 68: {
                // COPY tmp3 to tmp3
                Goldilocks3GPU::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION]);
                i_args += 2;
                break;
            }
            case 69: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION]);
                i_args += 4;
                break;
            }
            case 70: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 71: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 72: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 73: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 74: {
                // COPY eval to tmp3
                Goldilocks3GPU::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 1]*FIELD_EXTENSION*nrowsPack]);
                i_args += 2;
                break;
            }
            case 75: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 76: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &evals[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 77: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 78: {
                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                i_args += 5;
                break;
            }
            case 79: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &evals[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            default: {
                assert(false);
            }
        }
    }

    assert(i_args == nArgs);

}

#endif