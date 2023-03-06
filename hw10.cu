/*
* Homework 10: GPU Programming
* 
* Kelley Kelley
* CSCI 4239/5239 S22
* 
* Run and then it will prompt you for values
* 
* I took a class where we spent like 2 months using different
* algorithms to find the highest value of a function and I thought
* it'd be funny and fun to just brute force it especially since
* a lot of them like gradient descent have very specific parameters
* to the function (convex, continuous, so on) but brute force would 
* work no matter the function I think (at least get close ish hopefully we'll see)
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
//#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

//namespace cg = cooperative_groups;

//  Return elapsed wall time since last call (seconds)
static double t0 = 0;
double Elapsed(void);
//  Print message to stderr and exit
void Fatal(const char* format, ...);
//  Initialize fastest GPU device
int InitGPU(int verbose);

// function we are evaluating
float foo(float x) {
    float y = 0.0;
    // function I built that will hopefully cause gradient descent to fail 99% of the time
    // also very easy derivative
    y += 50.0 * cos(0.004 * x);
    y += 10.0 * sin(0.3 * x);
    y -= 5.0 * pow(cos(0.04 * x), 2);
    y += 2.0 * pow(sin(5.0 * x), 3);
    return y;
}

// derivative of function
float foodx(float x) {
    float y = 0;
    y += 50.0 * 0.004 * -1.0 * sin(0.004 * x);
    y += 10.0 * 0.3 * cos(0.3 * x);
    y -= 10.0 * cos(0.04 * x) * 0.04 * -1.0 * sin(0.04 * x);
    y += 6.0 * pow(sin(5.0 * x), 2) * 5.0 * cos(5.0 * x);
    return y;
}

// gradient descent function to find highest point
float gradient_descent(float start, float range) {
    // max iterations a gradient descent does
    int max_iter = 10000;
    // max random values we start with for gradient descent
    int max_rands = 1000;
    // tolerance
    float tolerance = .001;
    // the greatest value we've found so far and it's x
    float max_val = -1000;
    float max_x;
    // for gradient descent (how much derivative effects our value)
    float learn_rate = .5;
    // variables used in for loops
    float difference;
    float x;
    // start with a random value for gradient descent max_rands times
    for (int i = 0; i < max_rands; i++) {
        x = (rand() % (2 * (int)range)) - (int)range + start;
        // gradient descent until max_iter reached or converges
        for (int j = 0; j < max_iter; j++) {
            difference = learn_rate * foodx(x);
            if (fabs(difference) < tolerance)
                break;
            x = x - difference;
        }
        // test against our max to see if this gradient descent performed better
        if (foo(x) > max_val) {
            max_val = foo(x);
            max_x = x;
        }
    }
    printf("Gradient Descent x value = %f\n", max_x);
    printf("Gradient Descent function value = %f\n", foo(max_x));
    return max_x;
}

// brute force on host
float brute_force_h(float midpoint, float range, float stepsize) {
    float max_x = 0;
    float max_val = -1000;
    for (float i = midpoint - range; i <= midpoint + range; i += stepsize) {
        //printf("%f\n", i);
        if (foo(i) > max_val) {
            max_val = foo(i);
            max_x = i;
        }
    }
    printf("Brute force host x value = %f\n", max_x);
    printf("Brute force host function value = %f\n", foo(max_x));
    return max_x;
}

__global__ void brute_force(float midpoint, float range, float stepsize, const int Bw, float* max_val, float* max_x) {
    float x = blockIdx.x + midpoint - range + threadIdx.x * stepsize;
    unsigned int tid = threadIdx.x;
    __shared__ float reduce_val[1024];
    __shared__ float reduce_x[1024];
    float y = 0.0;
    y += 50.0 * cos(0.004 * x);
    y += 10.0 * sin(0.3 * x);
    y -= 5.0 * pow(cos(0.04 * x), 2);
    y += 2.0 * pow(sin(5.0 * x), 3);
    reduce_val[tid] = y;
    reduce_x[tid] = x;
    //printf("%i, %f\n", tid, reduce_val[tid]);

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2 * s)) == 0) {
            if (reduce_val[tid + s] > reduce_val[tid]) {
                //printf("%i, %f > %i, %f\n", tid + s, reduce_val[tid + s], tid, reduce_val[tid]);
                reduce_val[tid] = reduce_val[tid + s];
                reduce_x[tid] = reduce_x[tid + s];
            }
            //else {
                //printf("%i, %f < %i, %f\n", tid + s, reduce_val[tid + s], tid, reduce_val[tid]);
            //}
        }

        __syncthreads();
    }

    if (tid == 0) {
        max_x[blockIdx.x] = reduce_x[0];
        max_val[blockIdx.x] = reduce_val[0];
        //printf("%i: %f, %f\n", blockIdx.x, max_x[blockIdx.x], max_val[blockIdx.x]);
    }

    /* Idk why but I can't get grid groups to work
    cg::grid_group grid = cg::this_grid();
    cg::sync(grid);

    printf("%i\n", blockIdx.x);

    if (grid.thread_rank() == 0) {
        for (int block = 1; block < gridDim.x; block++) {
            if (max_val[0] > max_val[block]) {
                max_val[0] = max_val[block];
                max_x[0] = max_x[block];
                //printf("%f, %f\n", max_x[0], max_val[0]);
            }
        }
    }
    */
}

__global__ void reduce_blocks(float* rvalues, float* rxs, float* values, float* xs, int Bn) {
    unsigned int position = blockIdx.x * 1024 + threadIdx.x;
    unsigned int tid = threadIdx.x;
    __shared__ float rv[1024];
    __shared__ float rx[1024];
    rv[tid] = -100;
    rx[tid] = 0;
    
    __syncthreads();

    if (position < Bn) {
        rv[tid] = values[position];
        rx[tid] = xs[position];
        //printf("%i: %f, %f\n", tid, rv[tid], rx[tid]);
    }

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2 * s)) == 0) {
            if (rv[tid + s] > rv[tid]) {
                //printf("%i: %f, %f > %i: %f, %f\n", tid + s, rx[tid + s], rv[tid + s], tid, rx[tid], rv[tid]);
                rv[tid] = rv[tid + s];
                rx[tid] = rx[tid + s];
            }
            //else {
            //    printf("%i, %f < %i, %f\n", tid + s, reduce_val[tid + s], tid, reduce_val[tid]);
            //}
        }

        __syncthreads();
    }

    if (tid == 0) {
        rxs[blockIdx.x] = rx[0];
        rvalues[blockIdx.x] = rv[0];
        //printf("%i: %f, %f\n", blockIdx.x, rxs[blockIdx.x], rvalues[blockIdx.x]);
    }

}

// brute force on device
float brute_force_d(const float mp, const float rge, const float ss, int Bw, int Bn) {
    float* max_val;
    float* max_x;
    if (cudaMalloc((void**)&max_val, sizeof(float) * Bn)) Fatal("Cannot allocate device memory max_val\n");
    if (cudaMalloc((void**)&max_x, sizeof(float) * Bn)) Fatal("Cannot allocate device memory max_x\n");

    //float* midpoint;
    //float* range;
    //float* stepsize;
    //if (cudaMalloc((void**)&midpoint, sizeof(float))) Fatal("Cannot allocate device memory midpoint\n");
    //if (cudaMalloc((void**)&range, sizeof(float))) Fatal("Cannot allocate device memory range\n");
    //if (cudaMalloc((void**)&stepsize, sizeof(float))) Fatal("Cannot allocate device memory stepsize\n");

    //if (cudaMemcpy(midpoint, (void**)&mp, sizeof(float), cudaMemcpyHostToDevice)) Fatal("Cannot copy midpoint from host to device");
    //if (cudaMemcpy(range, (void**)&rge, sizeof(float), cudaMemcpyHostToDevice)) Fatal("Cannot copy range from host to device");
    //if (cudaMemcpy(stepsize, (void**)&ss, sizeof(float), cudaMemcpyHostToDevice)) Fatal("Cannot copy stepsize from host to device");

    brute_force<<<Bn,Bw>>>(mp, rge, ss, Bw, max_val, max_x);
    if (cudaGetLastError()) Fatal("Brute Force Failed\n");

    //float x_found;
    //if (cudaMemcpy((void**)&x_found, max_x, sizeof(float), cudaMemcpyDeviceToHost)) Fatal("Cannot copy x from device to host\n");

    // I can't figure out block reduction it's just not working so second kernel will reduce
    // then CPU will do final pass of at most like 60 elements
    float big_x = 0;
    float big_val = 0;
    
    float* reduce_val;
    float* reduce_x;

    int size = sizeof(float) * ceil(Bn / 1024.0);

    float* reduce_val_h = (float*)malloc(size);
    float* reduce_x_h = (float*)malloc(size);

    if (cudaMalloc((void**)&reduce_val, size)) Fatal("Cannot allocate device memory reduce_val\n");
    if (cudaMalloc((void**)&reduce_x, size)) Fatal("Cannot allocate device memory reduce_x\n");

    int block_num = ceil(Bn / 1024.0);
    reduce_blocks<<<block_num,1024>>>(reduce_val, reduce_x, max_val, max_x, Bn);
    if (cudaGetLastError()) Fatal("Brute Force Reduction Failed\n");
    
    if (cudaMemcpy(reduce_val_h, reduce_val, size, cudaMemcpyDeviceToHost)) Fatal("Cannot copy reduce_val from device to host\n");
    if (cudaMemcpy(reduce_x_h, reduce_x, size, cudaMemcpyDeviceToHost)) Fatal("Cannot copy reduce_x from device to host\n");

    big_x = reduce_x_h[0];
    big_val = reduce_val_h[0];
    for (int i = 1; i < ceil(Bn / 1024.0); i++) {
        if (reduce_val_h[i] > big_val) {
            big_val = reduce_val_h[i];
            big_x = reduce_x_h[i];
        }
    }

    cudaFree(reduce_val);
    cudaFree(reduce_x);
    free(reduce_val_h);
    free(reduce_x_h);

    printf("Device value = %f, x = %f\n", foo(big_x), big_x);

    cudaFree(max_val);
    cudaFree(max_x);
    //cudaFree(midpoint);
    //cudaFree(range);
    //cudaFree(stepsize);

    return big_x;
}

//
//  Main program
//
int main(int argc, char* argv[])
{
    // Prompt User for values
    // midpoint: used as start of gradient descent, and center for brute force
    float midpoint;
    printf("Enter midpoint value for evaluation (float): ");
    if (scanf("%f", &midpoint) != 1)
        Fatal("Entered bad midpoint value\n");
    // range: how far from midpoint we go in negative and positive direction
    float range;
    printf("Enter range for evaluation (float): ");
    if (scanf("%f", &range) != 1 || range == 0)
        Fatal("Entered bad range value\n");
    // step: used as step size for brute force and tolerance in gradient descent
    float step;
    printf("Enter step size for brute force (float): ");
    if (scanf("%f", &step) != 1 || step == 0)
        Fatal("Entered bad step size value\n");
    // check if we can do that with floating point precision
    if (midpoint - range == midpoint - range + step || midpoint + range == midpoint + range - step)
        Fatal("difference in range and step size too small for floating point precision\n");

    // here I'm just making sure the way I did step and range won't create too big a problem
    // for how I do block and threads
    // I could do a grid of threads for below 0 that would split the grid by how much precision you wanted
    // ie each grid is a multiple of 0.001
    // same thing with blocks, grid of blocks each block is a multiple of 65000, buuutttt, the CPU dies way before that
    // so not really relevant rn, only if I wanted to actually do this correctly for like an actual project
    int Bn = ceil(2*range+1);
    if (Bn >= 65001)
        Fatal("Haven't done that big a range yet. limit 32500\n");
    int Bw = ceil(1 / step);

    // Initialize GPU
    int Mw = InitGPU(0);
    //printf("Thread Count %d\n", Mw);
    if (Mw < Bw)
        Fatal("Haven't done that small a step size yet. Thread count %d exceeds threads per block of %d\n", Bw, Mw);

    // find largest value with gradient descent
    printf("--------------------\n");
    Elapsed();
    float grad_x = gradient_descent(midpoint, range);
    double grad_time = Elapsed();
    printf("Found in %fs\n", grad_time);

    // brute force host
    printf("--------------------\n");
    Elapsed();
    float brute_h_x = brute_force_h(midpoint, range, step);
    double brute_h_time = Elapsed();
    printf("Found in %fs\n", brute_h_time);

    // brute force device
    printf("--------------------\n");
    Elapsed();
    float brute_d_x = brute_force_d(midpoint, range, step, Bw, Bn);
    double brute_d_time = Elapsed();
    printf("Found in %fs\n", brute_d_time);

    //  Done
    return 0;
}

//
//  Initialize fastest GPU device
//
int InitGPU(int verbose)
{
    //  Get number of CUDA devices
    int num;
    if (cudaGetDeviceCount(&num)) Fatal("Cannot get number of CUDA devices\n");
    if (num < 1) Fatal("No CUDA devices found\n");

    //  Get fastest device
    cudaDeviceProp prop;
    int   MaxDevice = -1;
    int   MaxGflops = -1;
    for (int dev = 0; dev < num; dev++)
    {
        if (cudaGetDeviceProperties(&prop, dev)) Fatal("Error getting device %d properties\n", dev);
        int Gflops = prop.multiProcessorCount * prop.clockRate;
        if (verbose) printf("CUDA Device %d: %s Gflops %f Processors %d Threads/Block %d\n", dev, prop.name, 1e-6 * Gflops, prop.multiProcessorCount, prop.maxThreadsPerBlock);
        if (Gflops > MaxGflops)
        {
            MaxGflops = Gflops;
            MaxDevice = dev;
        }
    }

    //  Print and set device
    if (cudaGetDeviceProperties(&prop, MaxDevice)) Fatal("Error getting device %d properties\n", MaxDevice);
    printf("Fastest CUDA Device %d: %s\n", MaxDevice, prop.name);
    cudaSetDevice(MaxDevice);

    //  Return max thread count
    return prop.maxThreadsPerBlock;
}

//
//  Return elapsed wall time since last call (seconds)
//
double Elapsed(void)
{
#ifdef _WIN32
    //  Windows version of wall time
    LARGE_INTEGER tv, freq;
    QueryPerformanceCounter((LARGE_INTEGER*)&tv);
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    double t = tv.QuadPart / (double)freq.QuadPart;
#else
    //  Unix/Linux/OSX version of wall time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double t = tv.tv_sec + 1e-6 * tv.tv_usec;
#endif
    double s = t - t0;
    t0 = t;
    return s;
}

//
//  Print message to stderr and exit
//
void Fatal(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    exit(1);
}