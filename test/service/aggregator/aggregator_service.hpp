#ifndef AGGREGATOR_SERVICE_HPP
#define AGGREGATOR_SERVICE_HPP

#include "aggregator.grpc.pb.h"
#include "config.hpp"

class AggregatorServiceImpl final : public aggregator::v1::AggregatorService::Service {
private:
    Config &config;
    uint64_t messageId;

public:
    explicit AggregatorServiceImpl(Config &config) : config(config), messageId(0) {};
    
    ::grpc::Status Channel(::grpc::ServerContext* context, 
                          ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                          ::aggregator::v1::ProverMessage>* stream) override;

    ::grpc::Status ChannelOld(::grpc::ServerContext* context,
                             ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                             ::aggregator::v1::ProverMessage>* stream);

private:
    ::grpc::Status GetStatus(::grpc::ServerContext* context,
                            ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                            ::aggregator::v1::ProverMessage>* stream);

    ::grpc::Status Cancel(::grpc::ServerContext* context,
                         ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                         ::aggregator::v1::ProverMessage>* stream,
                         const std::string& requestID,
                         aggregator::v1::Result& result);

    ::grpc::Status GenBatchProof(::grpc::ServerContext* context,
                                ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                                ::aggregator::v1::ProverMessage>* stream,
                                const std::string& inputFile,
                                std::string& requestID);

    ::grpc::Status GenAggregatedProof(::grpc::ServerContext* context,
                                     ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                                     ::aggregator::v1::ProverMessage>* stream,
                                     const std::string& inputFileA,
                                     const std::string& inputFileB,
                                     std::string& requestID);

    ::grpc::Status GenFinalProof(::grpc::ServerContext* context,
                                ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                                ::aggregator::v1::ProverMessage>* stream,
                                const std::string& inputFile,
                                std::string& requestID);

    ::grpc::Status GetProof(::grpc::ServerContext* context,
                           ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                           ::aggregator::v1::ProverMessage>* stream,
                           const std::string& requestID,
                           aggregator::v1::GetProofResponse_Result& result,
                           std::string& proof);

    ::grpc::Status GenAndGetBatchProof(::grpc::ServerContext* context,
                                      ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                                      ::aggregator::v1::ProverMessage>* stream,
                                      const std::string& inputFile,
                                      const std::string& outputFile);

    ::grpc::Status GenAndGetAggregatedProof(::grpc::ServerContext* context,
                                           ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                                           ::aggregator::v1::ProverMessage>* stream,
                                           const std::string& inputFileA,
                                           const std::string& inputFileB,
                                           const std::string& outputFile);

    ::grpc::Status GenAndGetFinalProof(::grpc::ServerContext* context,
                                      ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage,
                                      ::aggregator::v1::ProverMessage>* stream,
                                      const std::string& inputFile,
                                      const std::string& outputFile);
};

#endif