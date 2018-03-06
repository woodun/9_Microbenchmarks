#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>

/////////////////////////////access consecutive data. change the strides to see L1 cache line size. question: why it does not show difference when passing the L2 cache line size?


void init_cpu_data(int* A, int size, int stride){
	for (int i = 0; i < size; ++i){
		A[i]=(i + stride) % size;
   	}
}


//////////min page size 4kb = 4096b = 32 * 128.
__device__ void P_chasing(int mark, int *A, int iterations, int *B, int starting_index, float clock_rate){
	
	int j = starting_index;/////make them in the same page, and miss near in cache lines
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside%d:%fms\n", mark, (total_time / (float)clock_rate) / (float)iterations);//////clock, average latency
	
	B[0] = j;
}

__global__ void tlb_latency_test_stride(int *A, int iterations, int *B, float clock_rate, int iter, int stride){
	
	printf("stride%d:\n", stride);
			
	P_chasing(-7, A, 16, B, 7 * 524288, clock_rate);/////warmup	
	
	P_chasing(8, A, 16, B, 8 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(8, A, 16, B, 8 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(8, A, 16, B, 8 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(8, A, 16, B, 8 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	
	P_chasing(-16, A, 1, B, 16 * 524288 + 0 * stride, clock_rate);/////warmup TLB
	P_chasing(-16, A, 1, B, 16 * 524288 + 31 * stride, clock_rate);/////warmup TLB
	P_chasing(16, A, 16, B, 16 * 524288 + 1 * stride, clock_rate);/////try to generate TLB hit and cache miss ///////
	P_chasing(16, A, 16, B, 16 * 524288 + 1 * stride, clock_rate);/////try to generate TLB hit and cache hit ///////
	P_chasing(16, A, 16, B, 16 * 524288 + 0 * stride, clock_rate);/////try to generate TLB hit and cache hit ///////
	P_chasing(16, A, 16, B, 16 * 524288 + 0 * stride, clock_rate);/////try to generate TLB hit and cache hit ///////
	
	P_chasing(-1, A, 1, B, 0 * 524288 + 0 * stride, clock_rate);/////warmup TLB
	P_chasing(-1, A, 1, B, 0 * 524288 + 31 * stride, clock_rate);/////warmup TLB
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * stride, clock_rate);/////try to generate TLB hit and cache miss ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * stride, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 0 * stride, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 0 * stride, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	
	printf("\n");	
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
	
	///////////////////////////////////////////////////////////////////GPU output data
	int *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(int) * 1));
		
	int data_size = 512 * 1024 * 100;/////size = iteration * stride = 100 2mb pages.
	for(int data_stride = 64; data_stride > 0; data_stride = data_stride / 2){	
		///////////////////////////////////////////////////////////////////CPU data begin
		int iterations_stride = data_size / data_stride;		
		int *CPU_data_in;	
		CPU_data_in = (int*)malloc(sizeof(int) * data_size);	
		init_cpu_data(CPU_data_in, data_size, data_stride);
		
		///////////////////////////////////////////////////////////////////GPU input data
		int *GPU_data_in;	
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		checkCudaErrors(cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice));
		
		tlb_latency_test_stride<<<1, 1>>>(GPU_data_in, iterations_stride, GPU_data_out, clock_rate, 16, data_stride);//////////////////////////////////////////////kernel is here
		cudaDeviceSynchronize();
	
		checkCudaErrors(cudaFree(GPU_data_in));	
		free(CPU_data_in);
	}	
	
	checkCudaErrors(cudaFree(GPU_data_out));
		
    exit(EXIT_SUCCESS);
}
