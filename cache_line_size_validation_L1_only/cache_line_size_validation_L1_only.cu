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

//////////////////////////////////////////////////////4 * (8) * 32 * 32 = 128kb ///////////////////48 * 128kb = 6144kb ///////////12 * 128kb = 1536kb ////////////// 16 * 64 = 1024 = 4kb
__global__ void tlb_latency_test_stride1(int *A, int iterations, int *B, float clock_rate){	

	printf("stride1:\n");

	int index = 0;
	
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
		
	P_chasing(15, A, 16, B, 15 * 524288, clock_rate);/////warmup
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(0, A, 1, B, 0 * 524288 + 0 * 1, clock_rate);/////warmup TLB
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 1, clock_rate);/////try to generate TLB hit and cache miss ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 1, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(16, A, 16, B, 16 * 524288 + 16 * 32, clock_rate);/////try to generate TLB hit and cache miss
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n");
	
	__syncthreads();
}

__global__ void tlb_latency_test_stride2(int *A, int iterations, int *B, float clock_rate){
	
	printf("stride2:\n");

	int index = 0;
	
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
		
	P_chasing(15, A, 16, B, 15 * 524288, clock_rate);/////warmup
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(0, A, 1, B, 0 * 524288 + 0 * 2, clock_rate);/////warmup TLB
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 2, clock_rate);/////try to generate TLB hit and cache miss ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 2, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(16, A, 16, B, 16 * 524288 + 16 * 32, clock_rate);/////try to generate TLB hit and cache miss
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n");
	
	__syncthreads();
}

__global__ void tlb_latency_test_stride4(int *A, int iterations, int *B, float clock_rate){
	
	printf("stride4:\n");

	int index = 0;
	
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
		
	P_chasing(15, A, 16, B, 15 * 524288, clock_rate);/////warmup
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(0, A, 1, B, 0 * 524288 + 0 * 4, clock_rate);/////warmup TLB
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 4, clock_rate);/////try to generate TLB hit and cache miss ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 4, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(16, A, 16, B, 16 * 524288 + 16 * 32, clock_rate);/////try to generate TLB hit and cache miss
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n");
	
	__syncthreads();
}

__global__ void tlb_latency_test_stride8(int *A, int iterations, int *B, float clock_rate){
	
	printf("stride8:\n");

	int index = 0;
	
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
		
	P_chasing(15, A, 16, B, 15 * 524288, clock_rate);/////warmup
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(0, A, 1, B, 0 * 524288 + 0 * 8, clock_rate);/////warmup TLB
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 8, clock_rate);/////try to generate TLB hit and cache miss ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 8, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(16, A, 16, B, 16 * 524288 + 16 * 32, clock_rate);/////try to generate TLB hit and cache miss
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n");
	
	__syncthreads();
}

__global__ void tlb_latency_test_stride16(int *A, int iterations, int *B, float clock_rate){
	
	printf("stride16:\n");

	int index = 0;
	
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
		
	P_chasing(15, A, 16, B, 15 * 524288, clock_rate);/////warmup
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(0, A, 1, B, 0 * 524288 + 0 * 16, clock_rate);/////warmup TLB
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 16, clock_rate);/////try to generate TLB hit and cache miss ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 16, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(16, A, 16, B, 16 * 524288 + 16 * 32, clock_rate);/////try to generate TLB hit and cache miss
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n");
	
	__syncthreads();
}

__global__ void tlb_latency_test_stride32(int *A, int iterations, int *B, float clock_rate){
	
	printf("stride32:\n");

	int index = 0;
	
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
		
	P_chasing(15, A, 16, B, 15 * 524288, clock_rate);/////warmup
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(0, A, 1, B, 0 * 524288 + 0 * 32, clock_rate);/////warmup TLB
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 32, clock_rate);/////try to generate TLB hit and cache miss ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 32, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(16, A, 16, B, 16 * 524288 + 16 * 32, clock_rate);/////try to generate TLB hit and cache miss
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(16, A, 16, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n");
	
	__syncthreads();
}

__global__ void tlb_latency_test_stride64(int *A, int iterations, int *B, float clock_rate){
	
	printf("stride64:\n");

	int index = 0;
	
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
		
	P_chasing(15, A, 8, B, 15 * 524288, clock_rate);/////warmup
	P_chasing(16, A, 8, B, 16 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(0, A, 1, B, 0 * 524288 + 0 * 64, clock_rate);/////warmup TLB
	P_chasing(0, A, 8, B, 0 * 524288 + 1 * 64, clock_rate);/////try to generate TLB hit and cache miss ///////(1)
	P_chasing(0, A, 8, B, 0 * 524288 + 1 * 64, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(16, A, 8, B, 16 * 524288 + 16 * 32, clock_rate);/////try to generate TLB hit and cache miss
	P_chasing(16, A, 8, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(16, A, 8, B, 16 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n");
	
	__syncthreads();
}

int main(int argc, char **argv)
{
	printf("\n");
	
    // set device
    cudaDeviceProp device_prop;
    //int dev_id = findCudaDevice(argc, (const char **) argv);
	int dev_id = 1;
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
		
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride64
	///////////////////////////////////////////////////////////////////CPU data begin
	int iterations_stride64 = 16384 * 100 / 2;	
	int data_stride_stride64 = 64;/////256b. Pointing to the next cacheline.	
	int data_size_stride64 = iterations_stride64 * data_stride_stride64;/////size = iteration * stride = 100 2mb pages.
	
	int *CPU_data_in_stride64;	
	CPU_data_in_stride64 = (int*)malloc(sizeof(int) * data_size_stride64);	
	init_cpu_data(CPU_data_in_stride64, data_size_stride64, data_stride_stride64);
	///////////////////////////////////////////////////////////////////CPU data end
	
	///////////////////////////////////////////////////////////////////GPU input data
	int *GPU_data_in_stride64;	
	checkCudaErrors(cudaMalloc(&GPU_data_in_stride64, sizeof(int) * data_size_stride64));	
	checkCudaErrors(cudaMemcpy(GPU_data_in_stride64, CPU_data_in_stride64, sizeof(int) * data_size_stride64, cudaMemcpyHostToDevice));
		
	tlb_latency_test_stride64<<<1, 1>>>(GPU_data_in_stride64, iterations_stride64, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	
	checkCudaErrors(cudaFree(GPU_data_in_stride64));	
	free(CPU_data_in_stride64);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride64
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride32
	///////////////////////////////////////////////////////////////////CPU data begin
	int iterations_stride32 = 1 * 16384 * 100;	
	int data_stride_stride32 = 32;/////64b. Pointing to the next cacheline.	
	int data_size_stride32 = iterations_stride32 * data_stride_stride32;/////size = iteration * stride = 100 2mb pages.
	
	int *CPU_data_in_stride32;	
	CPU_data_in_stride32 = (int*)malloc(sizeof(int) * data_size_stride32);	
	init_cpu_data(CPU_data_in_stride32, data_size_stride32, data_stride_stride32);
	///////////////////////////////////////////////////////////////////CPU data end
	
	///////////////////////////////////////////////////////////////////GPU input data
	int *GPU_data_in_stride32;	
	checkCudaErrors(cudaMalloc(&GPU_data_in_stride32, sizeof(int) * data_size_stride32));	
	checkCudaErrors(cudaMemcpy(GPU_data_in_stride32, CPU_data_in_stride32, sizeof(int) * data_size_stride32, cudaMemcpyHostToDevice));
		
	tlb_latency_test_stride32<<<1, 1>>>(GPU_data_in_stride32, iterations_stride32, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	
	checkCudaErrors(cudaFree(GPU_data_in_stride32));	
	free(CPU_data_in_stride32);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride32
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride16
	///////////////////////////////////////////////////////////////////CPU data begin
	int iterations_stride16 = 2 * 16384 * 100;	
	int data_stride_stride16 = 16;/////64b. Pointing to the next cacheline.	
	int data_size_stride16 = iterations_stride16 * data_stride_stride16;/////size = iteration * stride = 100 2mb pages.
	
	int *CPU_data_in_stride16;	
	CPU_data_in_stride16 = (int*)malloc(sizeof(int) * data_size_stride16);	
	init_cpu_data(CPU_data_in_stride16, data_size_stride16, data_stride_stride16);
	///////////////////////////////////////////////////////////////////CPU data end
	
	///////////////////////////////////////////////////////////////////GPU input data
	int *GPU_data_in_stride16;	
	checkCudaErrors(cudaMalloc(&GPU_data_in_stride16, sizeof(int) * data_size_stride16));	
	checkCudaErrors(cudaMemcpy(GPU_data_in_stride16, CPU_data_in_stride16, sizeof(int) * data_size_stride16, cudaMemcpyHostToDevice));
		
	tlb_latency_test_stride16<<<1, 1>>>(GPU_data_in_stride16, iterations_stride16, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	
	checkCudaErrors(cudaFree(GPU_data_in_stride16));	
	free(CPU_data_in_stride16);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride16
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride8
	///////////////////////////////////////////////////////////////////CPU data begin
	int iterations_stride8 = 4 * 16384 * 100;	
	int data_stride_stride8 = 8;/////64b. Pointing to the next cacheline.	
	int data_size_stride8 = iterations_stride8 * data_stride_stride8;/////size = iteration * stride = 100 2mb pages.
	
	int *CPU_data_in_stride8;	
	CPU_data_in_stride8 = (int*)malloc(sizeof(int) * data_size_stride8);	
	init_cpu_data(CPU_data_in_stride8, data_size_stride8, data_stride_stride8);
	///////////////////////////////////////////////////////////////////CPU data end
	
	///////////////////////////////////////////////////////////////////GPU input data
	int *GPU_data_in_stride8;	
	checkCudaErrors(cudaMalloc(&GPU_data_in_stride8, sizeof(int) * data_size_stride8));	
	checkCudaErrors(cudaMemcpy(GPU_data_in_stride8, CPU_data_in_stride8, sizeof(int) * data_size_stride8, cudaMemcpyHostToDevice));
		
	tlb_latency_test_stride8<<<1, 1>>>(GPU_data_in_stride8, iterations_stride8, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	
	checkCudaErrors(cudaFree(GPU_data_in_stride8));	
	free(CPU_data_in_stride8);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride8
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride4
	///////////////////////////////////////////////////////////////////CPU data begin
	int iterations_stride4 = 8 * 16384 * 100;	
	int data_stride_stride4 = 4;/////64b. Pointing to the next cacheline.	
	int data_size_stride4 = iterations_stride4 * data_stride_stride4;/////size = iteration * stride = 100 2mb pages.
	
	int *CPU_data_in_stride4;	
	CPU_data_in_stride4 = (int*)malloc(sizeof(int) * data_size_stride4);	
	init_cpu_data(CPU_data_in_stride4, data_size_stride4, data_stride_stride4);
	///////////////////////////////////////////////////////////////////CPU data end
	
	///////////////////////////////////////////////////////////////////GPU input data
	int *GPU_data_in_stride4;	
	checkCudaErrors(cudaMalloc(&GPU_data_in_stride4, sizeof(int) * data_size_stride4));	
	checkCudaErrors(cudaMemcpy(GPU_data_in_stride4, CPU_data_in_stride4, sizeof(int) * data_size_stride4, cudaMemcpyHostToDevice));
		
	tlb_latency_test_stride4<<<1, 1>>>(GPU_data_in_stride4, iterations_stride4, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	
	checkCudaErrors(cudaFree(GPU_data_in_stride4));	
	free(CPU_data_in_stride4);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride4
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride2
	///////////////////////////////////////////////////////////////////CPU data begin
	int iterations_stride2 = 16 * 16384 * 100;	
	int data_stride_stride2 = 2;/////64b. Pointing to the next cacheline.	
	int data_size_stride2 = iterations_stride2 * data_stride_stride2;/////size = iteration * stride = 100 2mb pages.
	
	int *CPU_data_in_stride2;	
	CPU_data_in_stride2 = (int*)malloc(sizeof(int) * data_size_stride2);	
	init_cpu_data(CPU_data_in_stride2, data_size_stride2, data_stride_stride2);
	///////////////////////////////////////////////////////////////////CPU data end
	
	///////////////////////////////////////////////////////////////////GPU input data
	int *GPU_data_in_stride2;	
	checkCudaErrors(cudaMalloc(&GPU_data_in_stride2, sizeof(int) * data_size_stride2));	
	checkCudaErrors(cudaMemcpy(GPU_data_in_stride2, CPU_data_in_stride2, sizeof(int) * data_size_stride2, cudaMemcpyHostToDevice));
		
	tlb_latency_test_stride2<<<1, 1>>>(GPU_data_in_stride2, iterations_stride2, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	
	checkCudaErrors(cudaFree(GPU_data_in_stride2));	
	free(CPU_data_in_stride2);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride2
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride1
	///////////////////////////////////////////////////////////////////CPU data begin
	int iterations_stride1 = 32 * 16384 * 100;	
	int data_stride_stride1 = 1;/////64b. Pointing to the next cacheline.	
	int data_size_stride1 = iterations_stride1 * data_stride_stride1;/////size = iteration * stride = 100 2mb pages.
	
	int *CPU_data_in_stride1;	
	CPU_data_in_stride1 = (int*)malloc(sizeof(int) * data_size_stride1);	
	init_cpu_data(CPU_data_in_stride1, data_size_stride1, data_stride_stride1);
	///////////////////////////////////////////////////////////////////CPU data end
	
	///////////////////////////////////////////////////////////////////GPU input data
	int *GPU_data_in_stride1;	
	checkCudaErrors(cudaMalloc(&GPU_data_in_stride1, sizeof(int) * data_size_stride1));	
	checkCudaErrors(cudaMemcpy(GPU_data_in_stride1, CPU_data_in_stride1, sizeof(int) * data_size_stride1, cudaMemcpyHostToDevice));
		
	tlb_latency_test_stride1<<<1, 1>>>(GPU_data_in_stride1, iterations_stride1, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	
	checkCudaErrors(cudaFree(GPU_data_in_stride1));	
	free(CPU_data_in_stride1);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////_stride1
	
	
	checkCudaErrors(cudaFree(GPU_data_out));
		
    exit(EXIT_SUCCESS);
}
