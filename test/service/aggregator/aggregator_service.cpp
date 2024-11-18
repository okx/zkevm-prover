#include "aggregator_service.hpp"
#include "utils.hpp"
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using namespace std;
using json = nlohmann::json;

#define AGGREGATOR_SERVER_NUMBER_OF_LOOPS 1

#define AGGREGATOR_SERVER_RETRY_SLEEP 10
#define AGGREGATOR_SERVER_NUMBER_OF_GET_PROOF_RETRIES 600  // 600 retries every 10 seconds = 6000 seconds = 100 minutes

::grpc::Status AggregatorServiceImpl::Channel(::grpc::ServerContext* context, 
    ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage, 
    ::aggregator::v1::ProverMessage>* stream)
{
#ifdef LOG_SERVICE
    cout << "AggregatorServiceImpl::Channel() stream starts" << endl;
#endif
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    aggregator::v1::Result result;
    string uuid;
    ::grpc::Status grpcStatus;
    string requestID;
    string proof;

    // Define file paths
    const string inputFile0  = "testvectors/e2e/fork_" + to_string(PROVER_FORK_ID) + "/input_executor_0.json";
    const string outputFile0 = "testvectors/aggregatedProof/fork_" + to_string(PROVER_FORK_ID) + "/recursive1.zkin.proof_0.json";

    const string inputFile1  = "testvectors/e2e/fork_" + to_string(PROVER_FORK_ID) + "/input_executor_1.json";
    const string outputFile1 = "testvectors/aggregatedProof/fork_" + to_string(PROVER_FORK_ID) + "/recursive1.zkin.proof_1.json";

    const string inputFile01a = outputFile0;
    const string inputFile01b = outputFile1;
    const string outputFile01 = "testvectors/finalProof/fork_" + to_string(PROVER_FORK_ID) + "/recursive2.zkin.proof_01.json";


    const string inputFile2  = "testvectors/e2e/fork_" + to_string(PROVER_FORK_ID) + "/input_executor_2.json";
    const string outputFile2 = "testvectors/aggregatedProof/fork_" + to_string(PROVER_FORK_ID) + "/recursive1.zkin.proof_2.json";
    
    const string inputFile3  = "testvectors/e2e/fork_" + to_string(PROVER_FORK_ID) + "/input_executor_3.json";
    const string outputFile3 = "testvectors/aggregatedProof/fork_" + to_string(PROVER_FORK_ID) + "/recursive1.zkin.proof_3.json";

    const string inputFile23a = outputFile2;
    const string inputFile23b = outputFile3;
    const string outputFile23 = "testvectors/finalProof/fork_" + to_string(PROVER_FORK_ID) + "/recursive2.zkin.proof_23.json";

    const string inputFile03a = outputFile01;
    const string inputFile03b = outputFile23;
    const string outputFile03 = "testvectors/finalProof/fork_" + to_string(PROVER_FORK_ID) + "/recursive2.zkin.proof_03.json";

    const string inputFileFinal  = outputFile03;
    const string outputFileFinal = "testvectors/finalProof/fork_" + to_string(PROVER_FORK_ID) + "/proof.json";


    // Get status
    grpcStatus = GetStatus(context, stream);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }

    // Cancel an invalid request ID and check result
    grpcStatus = Cancel(context, stream, "invalid_id", result);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    if (result != aggregator::v1::Result::RESULT_ERROR)
    {
        cerr << "Error: Channel() got cancel result=" << result 
             << " instead of RESULT_CANCEL_ERROR" << endl;
        return Status::CANCELLED;
    }

    // Main proof generation loop
    for (uint64_t loop = 0; loop < AGGREGATOR_SERVER_NUMBER_OF_LOOPS; loop++)
    {
        // Generate batch proof 0
        grpcStatus = GenAndGetBatchProof(context, stream, inputFile0, outputFile0);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "Channel() called GenAndGetBatchProof(" << inputFile0 
             << ", " << outputFile0 << ")" << endl;

        // Generate batch proof 1
        grpcStatus = GenAndGetBatchProof(context, stream, inputFile1, outputFile1);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "Channel() called GenAndGetBatchProof(" << inputFile1 
             << ", " << outputFile1 << ")" << endl;

        // Generate aggregated proof 01
        grpcStatus = GenAndGetAggregatedProof(context, stream, 
                                            inputFile01a, inputFile01b, outputFile01);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "Channel() called GenAndGetAggregatedProof(" << inputFile01a 
             << ", " << inputFile01b << ", " << outputFile01 << ")" << endl;


        // Generate batch proof 2
        grpcStatus = GenAndGetBatchProof(context, stream, inputFile2, outputFile2);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBatchProof(" << inputFile2 << ", " << outputFile2 << ")" << endl;

        // Generate batch proof 3
        grpcStatus = GenAndGetBatchProof(context, stream, inputFile3, outputFile3);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetBatchProof(" << inputFile3 << ", " << outputFile3 << ")" << endl;

        // Generate aggregated proof 23
        grpcStatus = GenAndGetAggregatedProof(context, stream, inputFile23a, inputFile23b, outputFile23);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetAggregatedProof(" << inputFile23a << ", " << inputFile23b << ", " << outputFile23 << ")" << endl;


        // Generate aggregated proof 03
        grpcStatus = GenAndGetAggregatedProof(context, stream, inputFile03a, inputFile03b, outputFile03);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetAggregatedProof(" << inputFile03a << ", " << inputFile03b << ", " << outputFile03 << ")" << endl;

        // Generate final proof
        grpcStatus = GenAndGetFinalProof(context, stream, inputFileFinal, outputFileFinal);
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }
        cout << "AggregatorServiceImpl::Channel() called GenAndGetFinalProof(" << inputFileFinal << ", " << outputFileFinal << ")" << endl;
    }

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GetStatus(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;

    // Send a get status request message
    aggregatorMessage.Clear();
    aggregator::v1::GetStatusRequest * pGetStatusRequest = new aggregator::v1::GetStatusRequest();
    zkassertpermanent(pGetStatusRequest != NULL);
    aggregatorMessage.set_allocated_get_status_request(pGetStatusRequest);
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GetStatus() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get status response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GetStatus() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGetStatusResponse)
    {
        cerr << "Error: AggregatorServiceImpl::GetStatus() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GET_STATUS_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::GetStatus() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::Cancel(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string &requestID, aggregator::v1::Result &result)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;

    // Send a cancel request message
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregator::v1::CancelRequest * pCancelRequest = new aggregator::v1::CancelRequest();
    zkassertpermanent(pCancelRequest != NULL);
    pCancelRequest->set_id(requestID);
    aggregatorMessage.set_allocated_cancel_request(pCancelRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding cancel response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kCancelResponse)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.response_case=" << proverMessage.response_case() << " instead of CANCEL_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    // Check cancel result
    result = proverMessage.cancel_response().result();

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenBatchProof(::grpc::ServerContext* context, 
    ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage, 
    ::aggregator::v1::ProverMessage>* stream, 
    const string &inputFile, 
    string &requestID)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;

    if (inputFile.empty()) {
        cerr << "Error: GenBatchProof() found inputFile empty" << endl;
        exitProcess();
    }

    // Create mock input data
    auto* pInputProver = new aggregator::v1::InputProver();
    auto* pPublicInputs = new aggregator::v1::PublicInputs();
    
    // Set mock values for testing
    pPublicInputs->set_old_state_root("0x1234");
    pPublicInputs->set_old_acc_input_hash("0x5678");
    pPublicInputs->set_old_batch_num(1);
    pPublicInputs->set_chain_id(1);
    pPublicInputs->set_fork_id(PROVER_FORK_ID);
    pPublicInputs->set_batch_l2_data("0x9abc");
    pPublicInputs->set_timestamp_limit(1000);
    pPublicInputs->set_sequencer_addr("0x1234");
    pPublicInputs->set_aggregator_addr("0x5678");
    
    pInputProver->set_allocated_public_inputs(pPublicInputs);

    // Add some mock db entries
    (*pInputProver->mutable_db())["key1"] = "value1";
    (*pInputProver->mutable_db())["key2"] = "value2";

    // Add some mock contract data
    (*pInputProver->mutable_contracts_bytecode())["contract1"] = "bytecode1";
    
    // Create and send request
    auto* pGenBatchProofRequest = new aggregator::v1::GenBatchProofRequest();
    pGenBatchProofRequest->set_allocated_input(pInputProver);

    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregatorMessage.set_allocated_gen_batch_proof_request(pGenBatchProofRequest);

    bResult = stream->Write(aggregatorMessage);
    if (!bResult) {
        cerr << "Error: GenBatchProof() failed calling stream->Write()" << endl;
        return Status::CANCELLED;
    }

    // Get response
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult) {
        cerr << "Error: GenBatchProof() failed calling stream->Read()" << endl;
        return Status::CANCELLED;
    }
    
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenBatchProofResponse) {
        cerr << "Error: GenBatchProof() got unexpected response type" << endl;
        return Status::CANCELLED;
    }

    if (proverMessage.id() != aggregatorMessage.id()) {
        cerr << "Error: GenBatchProof() got mismatched message ID" << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_batch_proof_response().id();
    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenAggregatedProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string &inputFileA, const string &inputFileB, string &requestID)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;
    string inputFileAContent;
    string inputFileBContent;

    if (inputFileA.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedProof() found inputFileA empty" << endl;
        exitProcess();
    }
    file2string(inputFileA, inputFileAContent);

    if (inputFileB.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedProof() found inputFileB empty" << endl;
        exitProcess();
    }
    file2string(inputFileB, inputFileBContent);

    // Allocate the aggregated batch request
    aggregator::v1::GenAggregatedProofRequest *pGenAggregatedProofRequest = new aggregator::v1::GenAggregatedProofRequest();
    zkassertpermanent(pGenAggregatedProofRequest != NULL );
    pGenAggregatedProofRequest->set_recursive_proof_1(inputFileAContent);
    pGenAggregatedProofRequest->set_recursive_proof_2(inputFileBContent);

    // Send the gen proof request
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregatorMessage.set_allocated_gen_aggregated_proof_request(pGenAggregatedProofRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedProof() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenAggregatedProofResponse)
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_AGGREGATED_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::GenAggregatedProof() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_aggregated_proof_response().id();

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenFinalProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string &inputFile, string &requestID)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;
    string inputFileContent;

    if (inputFile.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenFinalProof() found inputFile empty" << endl;
        exitProcess();
    }
    file2string(inputFile, inputFileContent);

    // Allocate the final batch request
    aggregator::v1::GenFinalProofRequest *pGenFinalProofRequest = new aggregator::v1::GenFinalProofRequest();
    zkassertpermanent(pGenFinalProofRequest != NULL );
    pGenFinalProofRequest->set_recursive_proof(inputFileContent);

    // Send the gen proof request
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregatorMessage.set_allocated_gen_final_proof_request(pGenFinalProofRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenFinalProof() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::GenFinalProof() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenFinalProofResponse)
    {
        cerr << "Error: AggregatorServiceImpl::GenFinalProof() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GEN_AGGREGATED_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::GenFinalProof() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    requestID = proverMessage.gen_final_proof_response().id();

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GetProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string &requestID, aggregator::v1::GetProofResponse_Result &result, string &proof)
{
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;

    // Send a get proof request message
    aggregatorMessage.Clear();
    messageId++;
    aggregatorMessage.set_id(to_string(messageId));
    aggregator::v1::GetProofRequest * pGetProofRequest = new aggregator::v1::GetProofRequest();
    zkassertpermanent(pGetProofRequest != NULL);
    pGetProofRequest->set_id(requestID);
    aggregatorMessage.set_allocated_get_proof_request(pGetProofRequest);
    bResult = stream->Write(aggregatorMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Write(aggregatorMessage)" << endl;
        return Status::CANCELLED;
    }

    // Receive the corresponding get proof response message
    proverMessage.Clear();
    bResult = stream->Read(&proverMessage);
    if (!bResult)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() failed calling stream->Read(proverMessage)" << endl;
        return Status::CANCELLED;
    }
    
    // Check type
    if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGetProofResponse)
    {
        cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.response_case=" << proverMessage.response_case() << " instead of GET_PROOF_RESPONSE" << endl;
        return Status::CANCELLED;
    }

    // Check id
    if (proverMessage.id() != aggregatorMessage.id())
    {
        cerr << "Error: AggregatorServiceImpl::Channel() got proverMessage.id=" << proverMessage.id() << " instead of aggregatorMessage.id=" << aggregatorMessage.id() << endl;
        return Status::CANCELLED;
    }

    // Copy get proof result
    result = proverMessage.get_proof_response().result();
    if ( proverMessage.get_proof_response().has_final_proof() )
    {
        proof = proverMessage.get_proof_response().final_proof().proof();
    }
    else
    {
        proof = proverMessage.get_proof_response().recursive_proof();
    }

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenAndGetBatchProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFile, const string &outputFile)
{
    ::grpc::Status grpcStatus;
    string requestID;
    string proof;
    uint64_t i;

    // Generate batch proof 0
    grpcStatus = GenBatchProof(context, stream, inputFile, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "AggregatorServiceImpl::GenAndGetBatchProof() called GenBatchProof() and got requestID=" << requestID << endl;

    // Get batch proof 0
    for (i=0; i<AGGREGATOR_SERVER_NUMBER_OF_GET_PROOF_RETRIES; i++)
    {
        sleep(AGGREGATOR_SERVER_RETRY_SLEEP);

        aggregator::v1::GetProofResponse_Result getProofResponseResult;
        grpcStatus = GetProof(context, stream, requestID, getProofResponseResult, proof);        
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }

        if (getProofResponseResult == aggregator::v1::GetProofResponse_Result_RESULT_COMPLETED_OK)
        {
            break;
        }        
        if (getProofResponseResult == aggregator::v1::GetProofResponse_Result_RESULT_PENDING)
        {
            continue;
        }
        cerr << "Error: AggregatorServiceImpl::GenAndGetBatchProof() got getProofResponseResult=" << getProofResponseResult << " instead of RESULT_PENDING or RESULT_COMPLETED_OK" << endl;
        return Status::CANCELLED;
    }
    if (i == AGGREGATOR_SERVER_NUMBER_OF_GET_PROOF_RETRIES)
    {
        cerr << "Error: AggregatorServiceImpl::GenAndGetBatchProof() timed out waiting for batch proof" << endl;
        return Status::CANCELLED;
    }
    if (proof.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenAndGetBatchProof() got an empty batch proof" << endl;
        return Status::CANCELLED;
    }
    string2file(proof, outputFile);

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenAndGetAggregatedProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFileA, const string & inputFileB, const string &outputFile)
{
    ::grpc::Status grpcStatus;
    string requestID;
    string proof;
    uint64_t i;

    // Generate batch proof 0
    grpcStatus = GenAggregatedProof(context, stream, inputFileA, inputFileB, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "AggregatorServiceImpl::GenAndGetAggregatedProof() called GenAggregatedProof() and got requestID=" << requestID << endl;

    // Get batch proof 0
    for (i=0; i<AGGREGATOR_SERVER_NUMBER_OF_GET_PROOF_RETRIES; i++)
    {
        sleep(AGGREGATOR_SERVER_RETRY_SLEEP);

        aggregator::v1::GetProofResponse_Result getProofResponseResult;
        grpcStatus = GetProof(context, stream, requestID, getProofResponseResult, proof);        
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }

        if (getProofResponseResult == aggregator::v1::GetProofResponse_Result_RESULT_COMPLETED_OK)
        {
            break;
        }        
        if (getProofResponseResult == aggregator::v1::GetProofResponse_Result_RESULT_PENDING)
        {
            continue;
        }
        cerr << "Error: AggregatorServiceImpl::GenAndGetAggregatedProof() got getProofResponseResult=" << getProofResponseResult << " instead of RESULT_PENDING or RESULT_COMPLETED_OK" << endl;
        return Status::CANCELLED;
    }
    if (i == AGGREGATOR_SERVER_NUMBER_OF_GET_PROOF_RETRIES)
    {
        cerr << "Error: AggregatorServiceImpl::GenAndGetAggregatedProof() timed out waiting for batch proof" << endl;
        return Status::CANCELLED;
    }
    if (proof.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenAndGetAggregatedProof() got an empty batch proof" << endl;
        return Status::CANCELLED;
    }
    string2file(proof, outputFile);

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::GenAndGetFinalProof(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::aggregator::v1::AggregatorMessage, ::aggregator::v1::ProverMessage>* stream, const string & inputFile, const string &outputFile)
{
    ::grpc::Status grpcStatus;
    string requestID;
    string proof;
    uint64_t i;

    // Generate batch proof 0
    grpcStatus = GenFinalProof(context, stream, inputFile, requestID);
    if (grpcStatus.error_code() != Status::OK.error_code())
    {
        return grpcStatus;
    }
    cout << "AggregatorServiceImpl::GenAndGetFinalProof() called GenFinalProof() and got requestID=" << requestID << endl;

    // Get batch proof 0
    for (i=0; i<AGGREGATOR_SERVER_NUMBER_OF_GET_PROOF_RETRIES; i++)
    {
        sleep(AGGREGATOR_SERVER_RETRY_SLEEP);

        aggregator::v1::GetProofResponse_Result getProofResponseResult;
        grpcStatus = GetProof(context, stream, requestID, getProofResponseResult, proof);        
        if (grpcStatus.error_code() != Status::OK.error_code())
        {
            return grpcStatus;
        }

        if (getProofResponseResult == aggregator::v1::GetProofResponse_Result_RESULT_COMPLETED_OK)
        {
            break;
        }        
        if (getProofResponseResult == aggregator::v1::GetProofResponse_Result_RESULT_PENDING)
        {
            continue;
        }
        cerr << "Error: AggregatorServiceImpl::GenAndGetFinalProof() got getProofResponseResult=" << getProofResponseResult << " instead of RESULT_PENDING or RESULT_COMPLETED_OK" << endl;
        return Status::CANCELLED;
    }
    if (i == AGGREGATOR_SERVER_NUMBER_OF_GET_PROOF_RETRIES)
    {
        cerr << "Error: AggregatorServiceImpl::GenAndGetFinalProof() timed out waiting for batch proof" << endl;
        return Status::CANCELLED;
    }
    if (proof.size() == 0)
    {
        cerr << "Error: AggregatorServiceImpl::GenAndGetFinalProof() got an empty batch proof" << endl;
        return Status::CANCELLED;
    }
    string2file(proof, outputFile);

    return Status::OK;
}

::grpc::Status AggregatorServiceImpl::ChannelOld(::grpc::ServerContext* context, 
    ::grpc::ServerReaderWriter<::aggregator::v1::AggregatorMessage, 
    ::aggregator::v1::ProverMessage>* stream)
{
#ifdef LOG_SERVICE
    cout << "AggregatorServiceImpl::Channel() stream starts" << endl;
#endif
    aggregator::v1::AggregatorMessage aggregatorMessage;
    aggregator::v1::ProverMessage proverMessage;
    bool bResult;
    string uuid;

    //while (true)
    {
        // CALL GET STATUS
        aggregatorMessage.Clear();
        auto* pGetStatusRequest = new aggregator::v1::GetStatusRequest();
        zkassertpermanent(pGetStatusRequest != NULL);
        aggregatorMessage.set_allocated_get_status_request(pGetStatusRequest);
        messageId++;
        aggregatorMessage.set_id(to_string(messageId));
        
        bResult = stream->Write(aggregatorMessage);
        if (!bResult) {
            cerr << "Error: Channel() failed calling stream->Write()" << endl;
            return Status::CANCELLED;
        }

        proverMessage.Clear();
        bResult = stream->Read(&proverMessage);
        if (!bResult) {
            cerr << "Error: Channel() failed calling stream->Read()" << endl;
            return Status::CANCELLED;
        }
        
        if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGetStatusResponse) {
            cerr << "Error: Channel() got unexpected response type" << endl;
            return Status::CANCELLED;
        }

        if (proverMessage.id() != aggregatorMessage.id()) {
            cerr << "Error: Channel() got mismatched message ID" << endl;
            return Status::CANCELLED;
        }

        sleep(1);

        // CALL CANCEL (it should return an error)
        aggregatorMessage.Clear();
        messageId++;
        aggregatorMessage.set_id(to_string(messageId));
        auto* pCancelRequest = new aggregator::v1::CancelRequest();
        zkassertpermanent(pCancelRequest != NULL);
        pCancelRequest->set_id("invalid_id");
        aggregatorMessage.set_allocated_cancel_request(pCancelRequest);
        
        bResult = stream->Write(aggregatorMessage);
        if (!bResult) {
            cerr << "Error: Channel() failed calling stream->Write()" << endl;
            return Status::CANCELLED;
        }

        proverMessage.Clear();
        bResult = stream->Read(&proverMessage);
        if (!bResult) {
            cerr << "Error: Channel() failed calling stream->Read()" << endl;
            return Status::CANCELLED;
        }
        
        if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kCancelResponse) {
            cerr << "Error: Channel() got unexpected response type" << endl;
            return Status::CANCELLED;
        }

        if (proverMessage.id() != aggregatorMessage.id()) {
            cerr << "Error: Channel() got mismatched message ID" << endl;
            return Status::CANCELLED;
        }

        if (proverMessage.cancel_response().result() != aggregator::v1::Result::RESULT_ERROR) {
            cerr << "Error: Channel() got unexpected cancel result" << endl;
            return Status::CANCELLED;
        }

        sleep(1);

        // Call GEN PROOF
        if (config.inputFile.empty()) {
            cerr << "Error: Channel() found config.inputFile empty" << endl;
            exitProcess();
        }

        // 創建模擬的 input data
        auto* pInputProver = new aggregator::v1::InputProver();
        zkassertpermanent(pInputProver != NULL);
        
        // 設置模擬的 public inputs
        auto* pPublicInputs = new aggregator::v1::PublicInputs();
        pPublicInputs->set_old_state_root("0x1234");
        pPublicInputs->set_old_acc_input_hash("0x5678");
        pPublicInputs->set_old_batch_num(1);
        pPublicInputs->set_chain_id(1);
        pPublicInputs->set_fork_id(PROVER_FORK_ID);
        pPublicInputs->set_batch_l2_data("0x9abc");
        pPublicInputs->set_l1_info_root("0xdef0");
        pPublicInputs->set_timestamp_limit(1000);
        pPublicInputs->set_forced_blockhash_l1("0x1111");
        pPublicInputs->set_sequencer_addr("0x2222");
        pPublicInputs->set_aggregator_addr("0x3333");
        pInputProver->set_allocated_public_inputs(pPublicInputs);

        // 添加模擬的 db 數據
        (*pInputProver->mutable_db())["key1"] = "value1";
        (*pInputProver->mutable_db())["key2"] = "value2";

        // 添加模擬的合約數據
        (*pInputProver->mutable_contracts_bytecode())["contract1"] = "bytecode1";
        (*pInputProver->mutable_contracts_bytecode())["contract2"] = "bytecode2";

        // Send gen batch proof request
        auto* pGenBatchProofRequest = new aggregator::v1::GenBatchProofRequest();
        zkassertpermanent(pGenBatchProofRequest != NULL);
        pGenBatchProofRequest->set_allocated_input(pInputProver);

        aggregatorMessage.Clear();
        messageId++;
        aggregatorMessage.set_id(to_string(messageId));
        aggregatorMessage.set_allocated_gen_batch_proof_request(pGenBatchProofRequest);
        
        bResult = stream->Write(aggregatorMessage);
        if (!bResult) {
            cerr << "Error: Channel() failed calling stream->Write()" << endl;
            return Status::CANCELLED;
        }

        proverMessage.Clear();
        bResult = stream->Read(&proverMessage);
        if (!bResult) {
            cerr << "Error: Channel() failed calling stream->Read()" << endl;
            return Status::CANCELLED;
        }
        
        if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGenBatchProofResponse) {
            cerr << "Error: Channel() got unexpected response type" << endl;
            return Status::CANCELLED;
        }

        if (proverMessage.id() != aggregatorMessage.id()) {
            cerr << "Error: Channel() got mismatched message ID" << endl;
            return Status::CANCELLED;
        }

        uuid = proverMessage.gen_batch_proof_response().id();

        // CALL GET PROOF AND CHECK IT IS PENDING
        for (uint64_t i = 0; i < 5; i++) {
            aggregatorMessage.Clear();
            messageId++;
            aggregatorMessage.set_id(to_string(messageId));
            auto* pGetProofRequest = new aggregator::v1::GetProofRequest();
            zkassertpermanent(pGetProofRequest != NULL);
            pGetProofRequest->set_id(uuid);
            aggregatorMessage.set_allocated_get_proof_request(pGetProofRequest);
            
            bResult = stream->Write(aggregatorMessage);
            if (!bResult) {
                cerr << "Error: Channel() failed calling stream->Write()" << endl;
                return Status::CANCELLED;
            }

            proverMessage.Clear();
            bResult = stream->Read(&proverMessage);
            if (!bResult) {
                cerr << "Error: Channel() failed calling stream->Read()" << endl;
                return Status::CANCELLED;
            }
            
            if (proverMessage.response_case() != aggregator::v1::ProverMessage::ResponseCase::kGetProofResponse) {
                cerr << "Error: Channel() got unexpected response type" << endl;
                return Status::CANCELLED;
            }

            if (proverMessage.id() != aggregatorMessage.id()) {
                cerr << "Error: Channel() got mismatched message ID" << endl;
                return Status::CANCELLED;
            }

            if (proverMessage.get_proof_response().result() != 
                aggregator::v1::GetProofResponse_Result_RESULT_PENDING) {
                cerr << "Error: Channel() got unexpected proof result" << endl;
                return Status::CANCELLED;
            }

            sleep(5);
        }
    }

#ifdef LOG_SERVICE
    cout << "Channel() stream done" << endl;
#endif

    return Status::OK;
}