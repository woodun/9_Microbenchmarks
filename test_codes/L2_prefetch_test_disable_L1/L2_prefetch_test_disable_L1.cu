#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>

///////////using multiple loops to measure the latency.
///////////when L1 is enabled. every miss will cause L2 to fetch 4 cache lines * 32 bytes to fill the 1 cache line * 128 byte in L1. Is it true? Change the starting offset to see.
///////////conclusion: By comparing with L1 disabled, it's clear that one L1 miss will fetch 4 L2 cache lines. When L1 is disabled, each L2 miss will only fetch 1 L2 cache line.

void init_cpu_data(int* A, int size, int stride, int mod){
	for (int i = 0; i < size; ++i){
		A[i]=(i + stride) % mod;
   	}
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void P_chasing(int mark, int *A, int iterations, int *B, int starting_index, float clock_rate, int data_stride){
	
	int j = starting_index;/////make them in the same page, and miss near in cache lines
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
			
	for (int it = 0; it < iterations; it++){
		j = A[j];
	}
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside%d:%fms\n", mark, (total_time / (float)clock_rate) / ((float)iterations));//////clock, average latency
	
	B[0] = j;
}

__global__ void tlb_latency_test(int *A, int iterations, int *B, float clock_rate, int mod, int data_stride){	
	
	P_chasing(0, A, iterations, B, 0, clock_rate, data_stride);////////saturate the L1 not L2
	P_chasing(7, A, iterations, B, 7, clock_rate, data_stride);////////access different parts of the 128 byte on L2
	P_chasing(15, A, iterations, B, 15, clock_rate, data_stride);////////access different parts of the 128 byte on L2
	P_chasing(23, A, iterations, B, 23, clock_rate, data_stride);////////access different parts of the 128 byte on L2
	P_chasing(31, A, iterations, B, 31, clock_rate, data_stride);////////access different parts of the 128 byte on L2
	
	 __syncthreads();
}

int main(int argc, char **argv)
{
	printf("\n");
	
    // set device
    cudaDeviceProp device_prop;
    //int dev_id = findCudaDevice(argc, (const char **) argv);
	int dev_id = 0;
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));
	
	int peak_clk = 1;//kHz
	checkCudaErrors(cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, dev_id));
	float clock_rate = (float) peak_clk;
	
	//printf("clock_rate_out_kernel:%f\n", clock_rate);

    if (!device_prop.managedMemory) { 
        // This samples requires being run on a device that supports Unified Memory
        fprintf(stderr, "Unified Memory not supported on this device\n");

        exit(EXIT_WAIVED);
    }

    if (device_prop.computeMode == cudaComputeModeProhibited)
    {
        // This sample requires being run with a default or process exclusive mode
        fprintf(stderr, "This sample requires a device in either default or process exclusive mode\n");

        exit(EXIT_WAIVED);
    }
	
	///////////////////////////////////////////////////////////////////GPU data out
	int *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(int) * 1));
	
	printf("################L1 saturated############################\n");	
	for(int data_stride = 32; data_stride <= 32; data_stride = data_stride + 1){/////////stride shall be L1 cache line size.
		printf("###################data_stride%d#########################\n", data_stride);
	//for(int mod = 1024 * 256 * 2; mod > 0; mod = mod - 32 * 1024){/////kepler L2 1.5m
	for(int mod = 1024 * 256 * 1; mod >= 1024 * 256 * 1; mod = mod / 2){/////kepler L2 1.5m ////////saturate the L1 not L2
		///////////////////////////////////////////////////////////////////CPU data begin
		int data_size = 512 * 1024 * 30;/////size = iteration * stride = 30 2mb pages.		
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		int iterations = mod / data_stride;
	
		int *CPU_data_in;
		CPU_data_in = (int*)malloc(sizeof(int) * data_size);	
		init_cpu_data(CPU_data_in, data_size, data_stride, mod);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		int *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
		
		tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);//////////////////////////////////////////////kernel is here	
		cudaDeviceSynchronize();
		
		checkCudaErrors(cudaFree(GPU_data_in));
		free(CPU_data_in);
	}
		printf("############################################\n\n");
	}
	
	if(0){////////////this part is not accurate because iteration is too small.
	printf("################L1 not saturated############################\n");
	for(int data_stride = 32; data_stride <= 32; data_stride = data_stride + 1){/////////stride shall be L1 cache line size.
		printf("###################data_stride%d#########################\n", data_stride);
	//for(int mod = 1024 * 256 * 2; mod > 0; mod = mod - 32 * 1024){/////kepler L2 1.5m
	for(int mod = 1024 * 1; mod >= 1024 * 1; mod = mod / 2){/////kepler L2 1.5m ////////saturate the L1 not L2
		///////////////////////////////////////////////////////////////////CPU data begin
		int data_size = 512 * 1024 * 30;/////size = iteration * stride = 30 2mb pages.		
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		int iterations = mod / data_stride;
	
		int *CPU_data_in;
		CPU_data_in = (int*)malloc(sizeof(int) * data_size);	
		init_cpu_data(CPU_data_in, data_size, data_stride, mod);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		int *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
		
		tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);//////////////////////////////////////////////kernel is here	
		cudaDeviceSynchronize();
		
		checkCudaErrors(cudaFree(GPU_data_in));
		free(CPU_data_in);
	}
		printf("############################################\n\n");
	}
	}
				
	checkCudaErrors(cudaFree(GPU_data_out));	
	//free(CPU_data_out);
	
    exit(EXIT_SUCCESS);
}
