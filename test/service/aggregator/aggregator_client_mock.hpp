#ifndef AGGREGATOR_CLIENT_MOCK_HPP
#define AGGREGATOR_CLIENT_MOCK_HPP

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "aggregator.grpc.pb.h"
#include "config.hpp"
#include "definitions.hpp" 
#include <chrono>
#include <string>
#include <pthread.h>

// 定義請求類型枚舉
enum tProverRequestType {
    prt_none,
    prt_genBatchProof,
    prt_genAggregatedProof,
    prt_genFinalProof
};

class AggregatorClientMock
{
private:
    std::chrono::system_clock::time_point lastAggregatorGenProof;
    std::string lastAggregatorUUID;
    tProverRequestType requestType;

public:
    const Config &config;
    aggregator::v1::AggregatorService::Stub* stub;
    pthread_t t;

public:
    explicit AggregatorClientMock(const Config &config);
    ~AggregatorClientMock();

    void runThread(void);
    void waitForThread(void);
    bool GetStatus(aggregator::v1::GetStatusResponse &getStatusResponse);
    bool GenBatchProof(const aggregator::v1::GenBatchProofRequest &genBatchProofRequest, 
                      aggregator::v1::GenBatchProofResponse &genBatchProofResponse);
    bool GenStatelessBatchProof(const aggregator::v1::GenStatelessBatchProofRequest &genStatelessBatchProofRequest, 
                               aggregator::v1::GenBatchProofResponse &genBatchProofResponse);
    bool GenAggregatedProof(const aggregator::v1::GenAggregatedProofRequest &genAggregatedProofRequest, 
                           aggregator::v1::GenAggregatedProofResponse &genAggregatedProofResponse);
    bool GenFinalProof(const aggregator::v1::GenFinalProofRequest &genFinalProofRequest, 
                      aggregator::v1::GenFinalProofResponse &genFinalProofResponse);
    bool Cancel(const aggregator::v1::CancelRequest &cancelRequest, 
               aggregator::v1::CancelResponse &cancelResponse);
    bool GetProof(const aggregator::v1::GetProofRequest &getProofRequest, 
                 aggregator::v1::GetProofResponse &getProofResponse);

    // Helper functions
    double TimeDiff(const std::chrono::system_clock::time_point& start) const;
    std::string getUUID() const;
};

// 輔助函數聲明
std::string getUUID();
std::string getTimestamp();
std::string string2ba(const std::string& input);
void string2file(const std::string& content, const std::string& filename);
void exitProcess();
uint64_t getNumberOfCores();

struct MemoryInfo {
    uint64_t total;
    uint64_t free;
};
void getMemoryInfo(MemoryInfo& info);

void* aggregatorClientMockThread(void* arg);

#endif