NVCC := /usr/local/cuda/bin/nvcc

$(shell cd src/goldilocks && ./configure.sh && cd ../..)
$(shell sleep 2)
include src/goldilocks/CudaArch.mk

TARGET_ZKP := zkProver
TARGET_ZKP_GPU := zkProver
TARGET_BCT := bctree
TARGET_MNG += mainGenerator
TARGET_PLG += polsGenerator
TARGET_TEST := zkProverTest

BUILD_DIR := ./build
BUILD_DIR_GPU := ./build-gpu
SRC_DIRS := ./src ./test ./tools

GRPCPP_FLAGS := $(shell pkg-config grpc++ --cflags)
GRPCPP_LIBS := $(shell pkg-config grpc++ --libs) -lgrpc++_reflection
ifndef GRPCPP_LIBS
$(error gRPC++ could not be found via pkg-config, you need to install them)
endif

CXX := g++
AS := nasm
CXXFLAGS := -std=c++17 -Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic -mavx2 $(GRPCPP_FLAGS) #-Wfatal-errors
LDFLAGS := -lprotobuf -lsodium -lgpr -lpthread -lpqxx -lpq -lgmp -lstdc++ -lgmpxx -lsecp256k1 -lcrypto -luuid -liomp5 $(GRPCPP_LIBS)
CFLAGS := -fopenmp
ASFLAGS := -felf64

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g -D DEBUG
else
      CXXFLAGS += -O3
endif

# Verify if AVX-512 is supported
AVX512_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx512' -m 1)

ifneq ($(AVX512_SUPPORTED),)
	CXXFLAGS += -mavx512f -D__AVX512__
endif

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

INC_DIRS := $(shell find $(SRC_DIRS) -type d) $(sort $(dir))
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

SRCS_ZKP := $(shell find $(SRC_DIRS) ! -path "./tools/starkpil/bctree/*" ! -path "./test/prover/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" ! -path "./src/goldilocks/utils/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)

SRCS_ZKP_GPU := $(shell find $(SRC_DIRS) ! -path "./tools/starkpil/bctree/*" ! -path "./test/prover/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" ! -path "./src/goldilocks/utils/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc -or -name *.cu)

OBJS_ZKP := $(SRCS_ZKP:%=$(BUILD_DIR)/%.o)
DEPS_ZKP := $(OBJS_ZKP:.o=.d)

OBJS_ZKP_GPU := $(SRCS_ZKP_GPU:%=$(BUILD_DIR_GPU)/%.o)

SRCS_BCT := $(shell find $(SRC_DIRS) ! -path "./src/main.cpp" ! -path "./test/prover/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_BCT := $(SRCS_BCT:%=$(BUILD_DIR)/%.o)
DEPS_BCT := $(OBJS_BCT:.o=.d)

SRCS_TEST := $(shell find $(SRC_DIRS) ! -path "./src/main.cpp" ! -path "./tools/starkpil/bctree/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/benchs/*" ! -path "./src/goldilocks/tests/*" ! -path "./src/main_generator/*" ! -path "./src/pols_generator/*" -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_TEST := $(SRCS_TEST:%=$(BUILD_DIR)/%.o)
DEPS_TEST := $(OBJS_TEST:.o=.d)

cpu: $(BUILD_DIR)/$(TARGET_ZKP)

gpu: $(BUILD_DIR_GPU)/$(TARGET_ZKP_GPU)

bctree: $(BUILD_DIR)/$(TARGET_BCT)

test: $(BUILD_DIR)/$(TARGET_TEST)

$(BUILD_DIR)/$(TARGET_ZKP): $(OBJS_ZKP)
	$(CXX) $(OBJS_ZKP) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR_GPU)/$(TARGET_ZKP_GPU): $(OBJS_ZKP_GPU)
	$(NVCC) $(OBJS_ZKP_GPU) -O3 -arch=$(CUDA_ARCH) -o $@ $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_BCT): $(OBJS_BCT)
	$(CXX) $(OBJS_BCT) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_TEST): $(OBJS_TEST)
	$(CXX) $(OBJS_TEST) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

# assembly
$(BUILD_DIR)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# assembly
$(BUILD_DIR_GPU)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -D__USE_CUDA__ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -Xcompiler -O3 -O3 -arch=$(CUDA_ARCH) -O3 $< -dc --output-file $@

main_generator: $(BUILD_DIR)/$(TARGET_MNG)

$(BUILD_DIR)/$(TARGET_MNG): ./src/main_generator/main_generator.cpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/main_generator/main_generator.cpp -o $@ -lgmp

pols_generator: $(BUILD_DIR)/$(TARGET_PLG)

$(BUILD_DIR)/$(TARGET_PLG): ./src/pols_generator/pols_generator.cpp
	$(MKDIR_P) $(BUILD_DIR)
	g++ -g ./src/pols_generator/pols_generator.cpp -o $@ -lgmp

.PHONY: clean

clean:
	$(RM) -rf $(BUILD_DIR)
	$(RM) -rf $(BUILD_DIR_GPU)

-include $(DEPS_ZKP)
-include $(DEPS_BCT)

MKDIR_P ?= mkdir -p
