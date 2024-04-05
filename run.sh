#/!bin/bash

set +e

for size in 128 256 512 1024 2048 4096 8192 16384 32768 65536 
do
    echo "Running with size ${size}"
    python pallas/pallas_mm.py -m "${size}" -n "${size}" -k "${size}" -p fp32 -o perf.csv || true
    python jax/jax_mm.py -m "${size}" -n "${size}" -k "${size}" -p fp32 -o perf.csv || true
    ./build/cuda/cuda_gemm -m "${size}" -n "${size}" -k "${size}" -p fp32 -co perf.csv -cb perf.csv || true
    ./build/cutlass/cutlass_gemm -m "${size}" -n "${size}" -k "${size}" -p fp32 -co perf.csv || true
    ./build/thrust/thrust_gemm -m "${size}" -n "${size}" -k "${size}" -p fp32 -co perf.csv || true
    ./build/matx/matx_gemm -m "${size}" -n "${size}" -k "${size}" -p fp32 -co perf.csv || true
done





