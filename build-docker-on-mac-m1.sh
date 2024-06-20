base_image=$(docker images | grep zkevm-prover-base)
if [[ -z "${base_image}" ]]; then
    docker build -t zkevm-prover-base -f Dockerfile-BASE .
fi

docker build -t zkevm-prover-mock -f Dockerfile-MOCK .
