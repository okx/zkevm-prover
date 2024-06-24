#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include "utils.hpp"
#include "config.hpp"
#include "version.hpp"
#include "service/aggregator/aggregator_client_mock.hpp"
#include "timer.hpp"
#include "zklog.hpp"

using namespace std;
using json = nlohmann::json;

/*
    Prover (available via GRPC service)
    |\
    | Executor (available via GRPC service)
    | |\
    | | Main State Machine
    | | Byte4 State Machine
    | | Binary State Machine
    | | Memory State Machine
    | | Mem Align State Machine
    | | Arithmetic State Machine
    | | Storage State Machine------\
    | |                             |--> Poseidon G State Machine
    | | Padding PG State Machine---/
    | | Padding KK SM -> Padding KK Bit -> Bits 2 Field SM -> Keccak-f SM
    |  \
    |   State DB (available via GRPC service)
    |   |\
    |   | SMT
    |    \
    |     Database
    |\
    | Stark
    |\
    | Circom
*/

int main(int argc, char **argv)
{
    /* CONFIG */

    if (argc == 2)
    {
        if ((strcmp(argv[1], "-v") == 0) || (strcmp(argv[1], "--version") == 0))
        {
            // If requested to only print the version, then exit the program
            return -1;
        }
    }

    // Parse the name of the configuration file
    char *pConfigFile = (char *)"config/config.json";
    if (argc == 3)
    {
        if ((strcmp(argv[1], "-c") == 0) || (strcmp(argv[1], "--config") == 0))
        {
            pConfigFile = argv[2];
        }
    }

    // Create one instance of Config based on the contents of the file config.json
    json configJson;
    file2json(pConfigFile, configJson);
    Config config;
    config.load(configJson);
    zklog.setJsonLogs(config.jsonLogs);
    zklog.setPID(config.proverID.substr(0, 7)); // Set the logs prefix

    // Print the zkProver version
    zklog.info("Version: " + string(ZKEVM_PROVER_VERSION));

    // Test that stderr is properly logged
    cerr << "Checking error channel; ignore this trace\n";
    zklog.warning("Checking warning channel; ignore this trace");

    // Print the configuration file name
    string configFileName = pConfigFile;
    zklog.info("Config file: " + configFileName);

    config.print();

    // Create output directory, if specified; otherwise, current working directory will be used to store output files
    if (config.outputPath.size() > 0)
    {
        ensureDirectoryExists(config.outputPath);
    }

    assert(config.runAggregatorClientMock);
    // Create the aggregator client and run it, if configured
    AggregatorClientMock * pAggregatorClientMock = new AggregatorClientMock(config);
    assert(pAggregatorClientMock != NULL);
    zklog.info("Launching aggregator client mock thread...");
    pAggregatorClientMock->runThread();

    pAggregatorClientMock->waitForThread();
    sleep(1);
    zklog.info("Done");

    return 0;
}