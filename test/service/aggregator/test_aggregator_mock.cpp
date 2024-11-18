#include <gtest/gtest.h>
#include "aggregator_client_mock.hpp"
#include "../../src/config/config.hpp"

TEST(AggregatorClientMockTest, BasicTest) {
    Config config;
    config.proverID = "test_prover";
    config.proverName = "Test Prover";
    config.aggregatorClientHost = "localhost";
    config.aggregatorClientPort = 8125;
    config.aggregatorClientMockTimeout = 5.0;
    
    AggregatorClientMock client(config);
    
    // Test GetStatus
    aggregator::v1::GetStatusResponse statusResponse;
    EXPECT_TRUE(client.GetStatus(statusResponse));
    EXPECT_EQ(statusResponse.prover_id(), config.proverID);
    EXPECT_EQ(statusResponse.prover_name(), config.proverName);
    
    // Test GenBatchProof
    aggregator::v1::GenBatchProofRequest batchRequest;
    aggregator::v1::GenBatchProofResponse batchResponse;
    EXPECT_TRUE(client.GenBatchProof(batchRequest, batchResponse));
    EXPECT_EQ(batchResponse.result(), aggregator::v1::Result::RESULT_OK);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}