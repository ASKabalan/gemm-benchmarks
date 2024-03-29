# gemm benchmarks

Benchmarking every GEMM algorithm provided by NVIDIA & JAX

Any GEMM can be tested  like so\


```bash
# C++ 
./build/exe -m 4096 -n 4096 -k 4096 -p fp32 -o perf.csv
# python
python python_mm.py -m 4096 -n 4096 -k 4096 -p fp32 -o perf.csv
```


Currenty I have these


```bash
python pallas/pallas_mm.py -m 4096 -n 4096 -k 4096 -p fp32 -o perf.csv
python jax/jax_mm.py -m 4096 -n 4096 -k 4096 -p fp32 -o perf.csv
./build/cuda/cuda_gemm -m 4096 -n 4096 -k 4096 -p fp32 -co perf.csv -cb perf.csv
./build/cutlass/cutlass_gemm -m 4096 -n 4096 -k 4096 -p fp32 -co perf.csv
./build/thrust/thrust_gemm -m 4096 -n 4096 -k 4096 -p fp32 -co perf.csv
./build/matx/matx_gemm -m 4096 -n 4096 -k 4096 -p fp32 -co perf.csv
```
