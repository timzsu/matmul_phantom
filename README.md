# Matmul example

- Tested on CUDA 12 and an H100 80GB GPU

## Environment Setup

### Docker

Build the docker image with the Dockerfile at the root directory. 

```bash
docker build -t matmul_example:v1.0 .
```

The code is copied to `/workspace`.

To launch the instance, use 

```bash
docker run -it --gpus all matmul_example:v1.0
cd /workspace
```

and then proceed to compilation. 

### Anaconda

- CUDA 12.1 installed in the system or by conda
- CMake == 3.31.4
- GCC == 12.4
- G++ == 12.4
- GMP == 6.2.1
- eigen == 3.4.0
- catch2 == 3.8.0
- NTL library built from [source](https://libntl.org/doc/tour-unix.html)

Here are commands to install NTL. 
```bash
wget https://libntl.org/ntl-11.5.1.tar.gz
tar xzvf ntl-11.5.1.tar.gz
cd ntl-11.5.1/src
./configure DEF_PREFIX=$CONDA_PREFIX
make -j8
make install
```

## Compiling NEXUS-CUDA

```bash
cmake -S . -B build
cmake --build build --parallel
```

This should produce an test executable `bin/matmul_test` inside the `build` directory. 

To run without benchmarks: 
```bash
./build/bin/matmul_test --skip-benchmarks
```

To run benchmarks with 5 repetition: (The default repetition is 100)
```bash
./build/bin/matmul_test --benchmark-samples 5
```

To test a certain section: 
```bash
./build/bin/matmul_test --skip-benchmarks -c $SECTION_NAME
```
where `$SECTION_NAME` is the string inside `SECTION()`. For example, to multiply a ct matrix of size 128x768 with a pt matrix of size 768x768, run 
```bash
./build/bin/matmul_test --skip-benchmarks -c "ct 128x768 pt 768x768"
```
