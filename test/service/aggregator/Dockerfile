# Multi-stage build
FROM --platform=linux/arm64 ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgmp-dev \
    libpqxx-dev \
    libgrpc++-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    libgtest-dev \
    libgmock-dev \
    libprotobuf-dev \
    libgrpc-dev \
    grpc-proto \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Copy source code and CMake files
COPY . .

# Remove existing build directory and create new one
RUN rm -rf build && mkdir build

# Create symbolic links for gRPC CMake files
RUN mkdir -p /usr/lib/cmake/grpc && \
    ln -s /usr/lib/aarch64-linux-gnu/cmake/grpc/* /usr/lib/cmake/grpc/

# Build with explicit paths
RUN cd build && \
    PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig \
    cmake \
    -DCMAKE_PREFIX_PATH="/usr;/usr/local" \
    -DgRPC_DIR=/usr/lib/cmake/grpc \
    -DProtobuf_DIR=/usr/lib/aarch64-linux-gnu/cmake/protobuf \
    .. && \
    make

# Runtime stage
FROM --platform=linux/arm64 ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgmp-dev \
    libpqxx-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Copy compiled binary from builder stage
COPY --from=builder /usr/src/app/build/test_aggregator_mock .

# Expose ports
EXPOSE 50051 50061 50071

# Start command
CMD ["./test_aggregator_mock"]