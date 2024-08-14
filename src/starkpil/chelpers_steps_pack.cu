#include "zklog.hpp"
#include <inttypes.h>

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)

#include "chelpers_steps_pack.cuh"
#include "goldilocks_cubic_extension.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "timer.hpp"

bool writeDataToFile(const std::string& filename, const uint64_t* data, size_t size) {
    // 打开文件
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        // 逐行写入数据
        for (size_t i = 0; i < size; i++) {
            file << (data[i] % 18446744069414584321) << std::endl;
        }
        // 关闭文件
        file.close();
        std::cout << "Data written to file successfully!" << std::endl;
        return true;
    } else {
        std::cerr << "Unable to open file." << std::endl;
        return false;
    }
}

const uint64_t MAX_U64 = 0xFFFFFFFFFFFFFFFF;
CHelpersStepsPackGPU *cHelpersSteps_d[MAX_GPUS];

void CHelpersStepsPackGPU::prepareGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {
    printf("into prepareGPU...\n");
    prepare(starkInfo, params, parserArgs, parserParams);

    nCudaThreads = 1 << 10;
    domainExtended = parserParams.stage > 3 ? true : false;
    domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    subDomainSize = nrowsPack * nCudaThreads;
    nextStride = domainExtended ? 1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;

    nOps = parserParams.nOps;
    nArgs = parserParams.nArgs;
    nStorePols = parserParams.nStorePols;
    nBufferT = 2*nCols*nrowsPack;
    nTemp1 = parserParams.nTemp1*nrowsPack;
    nTemp3 = parserParams.nTemp3*FIELD_EXTENSION*nrowsPack;

    printf("nCols:%lu\n", nCols);
    printf("nrowsPack:%lu\n", nrowsPack);
    printf("buffer:%lu\n", nBufferT);

    // The pols have been splited into segments on gpu, so offsetsStages_d is different from offsetsStages
    // Recalculate the offsetStages on GPU
    offsetsStagesGPU.resize(10 + 2);
    uint64_t total_offsets = 0;
    for (uint64_t s = 1; s < 11; s++) {
        if (s < 4 || (s == 4 && parserParams.stage != 4) || (s == 10 && domainExtended)) {
            printf("s:%lu, offsets:%lu\n", s, total_offsets);
            offsetsStagesGPU[s] = total_offsets;
            total_offsets += nColsStages[s] * (nrowsPack * nCudaThreads + nextStride);
        } else {
            offsetsStagesGPU[s] = MAX_U64;
        }
    }

    nPols = total_offsets;

    printf("total_offsets:%lu\n", total_offsets);

    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    printf("nDevices:%d\n", nDevices);

    #pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + nDevices + d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + 2*nDevices + d));

        CHECKCUDAERR(cudaMalloc(&nColsStages_d[d], nColsStages.size() * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&nColsStagesAcc_d[d], nColsStagesAcc.size() * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&offsetsStages_d[d], offsetsStagesGPU.size() * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&ops_d[d], nOps * sizeof(uint8_t)));
        CHECKCUDAERR(cudaMalloc(&args_d[d], nArgs * sizeof(uint16_t)));
        CHECKCUDAERR(cudaMalloc(&challenges_d[d], challenges.size() * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&challenges_ops_d[d], challenges_ops.size() * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&numbers_d[d], numbers_.size() * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&publics_d[d], publics.size() * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&evals_d[d], evals.size() * sizeof(uint64_t)));

        CHECKCUDAERR(cudaMalloc(&constPols_d[d], starkInfo.nConstants * (subDomainSize + nextStride) * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&x_d[d], subDomainSize * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&zi_d[d], subDomainSize * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&pols_d[d], total_offsets * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&xDivXSubXi_d[d], 2 * subDomainSize * FIELD_EXTENSION * sizeof(uint64_t)));

        CHECKCUDAERR(cudaMalloc(&gBufferT_[d], nBufferT * nCudaThreads * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&tmp1_d[d], nTemp1 * nCudaThreads * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&tmp3_d[d], nTemp3 * nCudaThreads * sizeof(uint64_t)));
    }

    #pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaMemcpyAsync(nColsStages_d[d], nColsStages.data(), nColsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(nColsStagesAcc_d[d], nColsStagesAcc.data(), nColsStagesAcc.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(offsetsStages_d[d], offsetsStagesGPU.data(), offsetsStagesGPU.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(ops_d[d], &parserArgs.ops[parserParams.opsOffset], nOps * sizeof(uint8_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(args_d[d], &parserArgs.args[parserParams.argsOffset], nArgs * sizeof(uint16_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(challenges_d[d], challenges.data(), challenges.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(challenges_ops_d[d], challenges_ops.data(), challenges_ops.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(numbers_d[d], numbers_.data(), numbers_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(publics_d[d], publics.data(), publics.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(evals_d[d], evals.data(), evals.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
    }

    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
    }

    //    CHECKCUDAERR(cudaMalloc(&storePols_d, nStorePols * sizeof(uint8_t)));
    //    CHECKCUDAERR(cudaMemcpy(storePols_d, &parserArgs.storePols[parserParams.storePolsOffset], nStorePols * sizeof(uint8_t), cudaMemcpyHostToDevice));

}

void CHelpersStepsPackGPU::cleanupGPU() {

    #pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        cudaFree(nColsStages_d[d]);
        cudaFree(nColsStagesAcc_d[d]);
        cudaFree(offsetsStages_d[d]);

        cudaFree(ops_d[d]);
        cudaFree(args_d[d]);
        //cudaFree(storePols_d[d]);

        cudaFree(challenges_d[d]);
        cudaFree(challenges_ops_d[d]);
        cudaFree(numbers_d[d]);
        cudaFree(publics_d[d]);
        cudaFree(evals_d[d]);

        cudaFree(constPols_d[d]);
        cudaFree(x_d[d]);
        cudaFree(zi_d[d]);
        cudaFree(pols_d[d]);
        cudaFree(xDivXSubXi_d[d]);

        cudaFree(gBufferT_[d]);
        cudaFree(tmp1_d[d]);
        cudaFree(tmp3_d[d]);

        cudaStreamDestroy(gpu_stream[d]);
        cudaStreamDestroy(gpu_stream[nDevices+d]);
        cudaStreamDestroy(gpu_stream[2*nDevices+d]);
    }

}

void CHelpersStepsPackGPU::compare(StepsParams &params, uint64_t row) {
//    for (uint64_t s = 1; s < 11; s++) {
//        if (offsetsStagesGPU[s] != MAX_U64) {
//            printf("compare s:%lu\n", s);
//            Goldilocks::Element *temp = (Goldilocks::Element *)malloc(subDomainSize *nColsStages[s] * sizeof(uint64_t));
//            CHECKCUDAERR(cudaMemcpy(temp, pols_d + offsetsStagesGPU[s], subDomainSize *nColsStages[s] * sizeof(uint64_t), cudaMemcpyDeviceToHost));
//            for (uint64_t i=0; i<subDomainSize *nColsStages[s]; i++) {
//                uint64_t left = Goldilocks::toU64(params.pols[offsetsStages[s] + row*nColsStages[s] + i]);
//                uint64_t right = Goldilocks::toU64(temp[i]);
//                if (left != right) {
//                    printf("compare not equal, s:%lu, i:%lu, left:%lu, right:%lu\n", s, i, left, right);
//                    writeDataToFile("left.txt", (uint64_t *)params.pols +offsetsStages[s] + row*nColsStages[s], subDomainSize *nColsStages[s]);
//                    writeDataToFile("right.txt", (uint64_t *)temp, subDomainSize *nColsStages[s]);
//                    break;
//                }
//            }
//            free(temp);
//        }
//    }

    for (uint64_t s = 1; s < 11; s++) {
        if (offsetsStagesGPU[s] != MAX_U64) {
            printf("write s:%lu\n", s);
            writeDataToFile("gpu.txt", (uint64_t *)params.pols +offsetsStages[s] + row*nColsStages[s], (subDomainSize + nextStride) *nColsStages[s]);
        }
    }

    assert(0);
}

void CHelpersStepsPackGPU::calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {

    prepareGPU(starkInfo, params, parserArgs, parserParams);

    #pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++) {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaMalloc((void **)&(cHelpersSteps_d[d]), sizeof(CHelpersStepsPackGPU)));
        CHECKCUDAERR(cudaMemcpy(cHelpersSteps_d[d], this, sizeof(CHelpersStepsPackGPU), cudaMemcpyHostToDevice));
    }

    calculateExpressionsRowsGPU(starkInfo, params, parserArgs, parserParams, 0, domainSize-nrowsPack*nCudaThreads*nDevices);
    calculateExpressionsRows(starkInfo, params, parserArgs, parserParams, domainSize-nrowsPack*nCudaThreads*nDevices, domainSize);
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[nDevices+d]));
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[2*nDevices+d]));
    }
    #pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++) {
        cudaFree(cHelpersSteps_d[d]);
    }

    cleanupGPU();
    //compare(params, 0);
}

void CHelpersStepsPackGPU::calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams,
    uint64_t rowIni, uint64_t rowEnd){

    if(rowEnd < rowIni || rowEnd > domainSize || (rowEnd -rowIni) % nrowsPack != 0) {
        zklog.info("Invalid range for rowIni " + to_string(rowIni) + " and rowEnd " + to_string(rowEnd));
        exitProcess();
    }

    for (uint64_t i = rowIni; i < rowEnd; i+= nrowsPack*nCudaThreads*nDevices) {
        printf("rows:%lu\n", i);
//        if (i + nrowsPack*nCudaThreads > rowEnd) {
//            nCudaThreads = (rowEnd - i) / nrowsPack;
//            subDomainSize = nrowsPack*nCudaThreads;
//        }

        TimerStart(Memcpy_H_to_D);
        loadData(starkInfo, params, i);
        TimerStopAndLog(Memcpy_H_to_D);

        TimerStart(EXP_Kernel);

        //#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++) {
            cudaStream_t stream = gpu_stream[groupIdx*nDevices+d];
            loadPolinomialsGPU<<<(nCudaThreads+15)/16, 16, 0, stream>>>(cHelpersSteps_d[d], starkInfo.nConstants, parserParams.stage, d);
            pack_kernel<<<(nCudaThreads+15)/16, 16, 0, stream>>>(cHelpersSteps_d[d], d);
            storePolinomialsGPU<<<(nCudaThreads+15)/16, 16, 0, stream>>>(cHelpersSteps_d[d], d);
        }

        // debug
//        uint64_t *temp = (uint64_t *)malloc(nBufferT * nCudaThreads * sizeof(uint64_t));
//        CHECKCUDAERR(cudaMemcpy(temp, gBufferT_, nBufferT * nCudaThreads * sizeof(uint64_t), cudaMemcpyDeviceToHost));
//        writeDataToFile("input2.txt", temp, nBufferT * nCudaThreads);
        //

//        CHECKCUDAERR(cudaMemcpy(temp, gBufferT_, nBufferT * nCudaThreads * sizeof(uint64_t), cudaMemcpyDeviceToHost));
//        writeDataToFile("output2.txt", temp, nBufferT * nCudaThreads);

        TimerStopAndLog(EXP_Kernel);

        TimerStart(Memcpy_D_to_H);
        storeData(params, i);
        TimerStopAndLog(Memcpy_D_to_H);

        groupIdx = (groupIdx + 1) % 3;
    }
}

void CHelpersStepsPackGPU::loadData(StarkInfo &starkInfo, StepsParams &params, uint64_t row) {

    ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
    Polinomial &x = domainExtended ? params.x_2ns : params.x_n;

    //#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++) {
        cudaStream_t stream = gpu_stream[groupIdx*nDevices+d];
        Goldilocks::Element *constPols_h = ((Goldilocks::Element *)constPols->address()) + (row+d*subDomainSize)*starkInfo.nConstants;
        CHECKCUDAERR(cudaMemcpyAsync(constPols_d[d], constPols_h, starkInfo.nConstants * (subDomainSize+nextStride)*sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        CHECKCUDAERR(cudaMemcpyAsync(x_d[d], x[row+d*subDomainSize], subDomainSize * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        CHECKCUDAERR(cudaMemcpyAsync(zi_d[d], params.zi[row+d*subDomainSize], subDomainSize * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

        for (uint64_t s = 1; s < 11; s++) {
            if (offsetsStagesGPU[s] != MAX_U64) {
                Goldilocks::Element *pols_h = &params.pols[offsetsStages[s]+(row+d*subDomainSize)*nColsStages[s]];
                CHECKCUDAERR(cudaMemcpyAsync(pols_d[d] + offsetsStagesGPU[s], pols_h, (subDomainSize+nextStride)*nColsStages[s]*sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
            }
        }

        CHECKCUDAERR(cudaMemcpyAsync(xDivXSubXi_d[d], params.xDivXSubXi[row+d*subDomainSize], subDomainSize*FIELD_EXTENSION*sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        CHECKCUDAERR(cudaMemcpyAsync(xDivXSubXi_d[d] + subDomainSize*FIELD_EXTENSION, params.xDivXSubXi[domainSize + row+d*subDomainSize], subDomainSize*FIELD_EXTENSION*sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    }
}

void CHelpersStepsPackGPU::storeData(StepsParams &params, uint64_t row) {

    //#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++) {
        cudaStream_t stream = gpu_stream[groupIdx*nDevices+d];
        for (uint64_t s = 1; s < 11; s++) {
            if (offsetsStagesGPU[s] != MAX_U64) {
                Goldilocks::Element *pols_h = &params.pols[offsetsStages[s] + (row+d*subDomainSize)*nColsStages[s]];
                CHECKCUDAERR(cudaMemcpyAsync(pols_h, pols_d[d] + offsetsStagesGPU[s], subDomainSize*nColsStages[s]*sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
            }
        }
    }
}

__global__ void loadPolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps, uint64_t nConstants, uint64_t stage, uint32_t d) {

    uint64_t nCudaThreads = cHelpersSteps->nCudaThreads;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCudaThreads) {
        return;
    }

    uint64_t nrowsPack = cHelpersSteps->nrowsPack;
    uint64_t nextStride = cHelpersSteps->nextStride;
    //uint64_t domainSize = cHelpersSteps->domainSize;
    uint64_t subDomainSize = cHelpersSteps->subDomainSize;
    uint64_t nBufferT = cHelpersSteps->nBufferT;

    uint64_t *nColsStages = cHelpersSteps->nColsStages_d[d];
    uint64_t *nColsStagesAcc = cHelpersSteps->nColsStagesAcc_d[d];
    uint64_t *offsetsStages = cHelpersSteps->offsetsStages_d[d];

    gl64_t *bufferT_ = cHelpersSteps->gBufferT_[d] + idx * nBufferT;
//    for (uint64_t i = 0; i < nBufferT; i++) {
//        bufferT_[i] = gl64_t(uint64_t(0));
//    }
    gl64_t *pols = cHelpersSteps->pols_d[d];
    gl64_t *constPols = cHelpersSteps->constPols_d[d];

    uint64_t row = idx*nrowsPack;
    uint64_t nStages = 3;
    uint64_t nextStrides[2] = {0, nextStride};

    for(uint64_t k = 0; k < nConstants; ++k) {
        for(uint64_t o = 0; o < 2; ++o) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                uint64_t l = (row + j + nextStrides[o]); // % domainSize;
                bufferT_[(nColsStagesAcc[5*o] + k)*nrowsPack + j] = constPols[l * nConstants + k];
            }
        }
    }

    // Load x and Zi
    for(uint64_t j = 0; j < nrowsPack; ++j) {
        bufferT_[nConstants*nrowsPack + j] = cHelpersSteps->x_d[d][row + j];
    }
    for(uint64_t j = 0; j < nrowsPack; ++j) {
        bufferT_[(nConstants + 1)*nrowsPack + j] = cHelpersSteps->zi_d[d][row + j];
    }

    for(uint64_t s = 1; s <= nStages; ++s) {
        for(uint64_t k = 0; k < nColsStages[s]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]); // % domainSize;
                    bufferT_[(nColsStagesAcc[5*o + s] + k)*nrowsPack + j] = pols[offsetsStages[s] + l * nColsStages[s] + k];
                }
            }
        }
    }

    if(stage == 5) {
        for(uint64_t k = 0; k < nColsStages[nStages + 1]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]); // % domainSize;
                    bufferT_[(nColsStagesAcc[5*o + nStages + 1] + k)*nrowsPack + j] = pols[offsetsStages[nStages + 1] + l * nColsStages[nStages + 1] + k];
                }
            }
        }

       // Load xDivXSubXi & xDivXSubWXi
       for(uint64_t d = 0; d < 2; ++d) {
           for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
               for(uint64_t j = 0; j < nrowsPack; ++j) {
                  bufferT_[(nColsStagesAcc[11] + FIELD_EXTENSION*d + i)*nrowsPack + j] = cHelpersSteps->xDivXSubXi_d[d][(d*subDomainSize + row + j) * FIELD_EXTENSION + i];
               }
           }
       }
    }
}


__global__ void storePolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps, uint32_t d) {
    uint64_t nCudaThreads = cHelpersSteps->nCudaThreads;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCudaThreads) {
        return;
    }

    bool domainExtended = cHelpersSteps->domainExtended;
    uint64_t nrowsPack = cHelpersSteps->nrowsPack;
    uint64_t nBufferT = cHelpersSteps->nBufferT;

    uint64_t row = idx*nrowsPack;

    uint64_t *nColsStages = cHelpersSteps->nColsStages_d[d];
    uint64_t *nColsStagesAcc = cHelpersSteps->nColsStagesAcc_d[d];
    uint64_t *offsetsStages = cHelpersSteps->offsetsStages_d[d];

    // uint8_t *storePols = cHelpersSteps->storePols_d;

    gl64_t *bufferT_ = cHelpersSteps->gBufferT_[d] + idx * nBufferT;
    gl64_t *pols = cHelpersSteps->pols_d[d];

    if(domainExtended) {
        // Store either polinomial f or polinomial q
        for(uint64_t k = 0; k < nColsStages[10]; ++k) {
            gl64_t *buffT = &bufferT_[(nColsStagesAcc[10] + k)* nrowsPack];
            gl64_t::copy_pack(nrowsPack, &pols[offsetsStages[10] + k + row * nColsStages[10]], nColsStages[10], buffT);
        }
    } else {
        uint64_t nStages = 3;
        for(uint64_t s = 2; s <= nStages + 1; ++s) {
            for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                gl64_t *buffT = &bufferT_[(nColsStagesAcc[s] + k)* nrowsPack];
                gl64_t::copy_pack(nrowsPack, &pols[offsetsStages[s] + k + row * nColsStages[s]], nColsStages[s], buffT);
            }
        }
    }
}

__global__ void pack_kernel(CHelpersStepsPackGPU *cHelpersSteps, uint32_t d)
{
    uint64_t nCudaThreads = cHelpersSteps->nCudaThreads;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCudaThreads) {
        return;
    }

    uint64_t nrowsPack = cHelpersSteps->nrowsPack;
    uint64_t nOps = cHelpersSteps->nOps;
    uint64_t nArgs = cHelpersSteps->nArgs;
    uint64_t nBufferT = cHelpersSteps->nBufferT;
    uint64_t nTemp1 = cHelpersSteps->nTemp1;
    uint64_t nTemp3 = cHelpersSteps->nTemp3;

    uint64_t *nColsStagesAcc = cHelpersSteps->nColsStagesAcc_d[d];
    uint8_t *ops = cHelpersSteps->ops_d[d];
    uint16_t *args = cHelpersSteps->args_d[d];
    gl64_t *challenges = cHelpersSteps->challenges_d[d];
    gl64_t *challenges_ops = cHelpersSteps->challenges_ops_d[d];
    gl64_t *numbers_ = cHelpersSteps->numbers_d[d];
    gl64_t *publics = cHelpersSteps->publics_d[d];
    gl64_t *evals = cHelpersSteps->evals_d[d];

    gl64_t *bufferT_ = cHelpersSteps->gBufferT_[d] + idx * nBufferT;
    gl64_t *tmp1 = cHelpersSteps->tmp1_d[d] + nTemp1*idx;
    gl64_t *tmp3 = cHelpersSteps->tmp3_d[d] + nTemp3*idx;

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