#ifndef AGGREGATOR_SERVER_HPP
#define AGGREGATOR_SERVER_HPP

#include "config.hpp"
#include <pthread.h>

class AggregatorServer
{
    Config &config;
    pthread_t t;
public:
    explicit AggregatorServer(Config &config) : config(config) {};
    void run (void);
    void runThread (void);
    void waitForThread (void);
};

void* aggregatorServerThread(void* arg);

#endif