#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>

/////////////////////////////check if L1 and L2 flush between kernels.


void init_cpu_data(int* A, int size, int stride){
	for (int i = 0; i < size; ++i){
		A[i]=(i + stride) % size;
   	}
}

/*
__device__ void cache_warmup(int *A, int iterations, int *B){
	
	int j = 0;
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	B[0] = j;
}
*/

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
__global__ void tlb_latency_test(int *A, int iterations, int *B, float clock_rate){	

	int index = 0;
	
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
		
	P_chasing(-2, A, 7, B, 2 * 524288, clock_rate);/////warmup GPU
	
	P_chasing(-1, A, 1, B, 0 * 524288 + 0 * 16, clock_rate);/////warmup TLB
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 16, clock_rate);/////try to generate TLB hit and cache miss ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 16, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 0 * 16, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	
	P_chasing(3, A, 16, B, 3 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(3, A, 16, B, 3 * 524288 + 32 * 16, clock_rate);/////try to generate TLB hit and cache miss
	P_chasing(3, A, 16, B, 3 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(3, A, 16, B, 3 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n");
}

//////////////////////////////////////////////////////4 * (8) * 32 * 32 = 128kb ///////////////////48 * 128kb = 6144kb ///////////12 * 128kb = 1536kb
__global__ void tlb_latency_test_2(int *A, int iterations, int *B, float clock_rate){	

	int index = 0;
	
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
		
	//P_chasing(0, A, 1, B, 31 * 32, clock_rate);/////warmup TLB
	P_chasing(1, A, 16, B, 1 * 524288, clock_rate);/////warmup GPU
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 16, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 1 * 16, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(0, A, 16, B, 0 * 524288 + 0 * 16, clock_rate);/////try to generate TLB hit and cache hit ///////(1)
	P_chasing(14, A, 16, B, 14 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(15, A, 16, B, 15 * 524288 + 16 * 16, clock_rate);/////try to generate TLB hit and cache miss	
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n");
}

//////////////////////////////////////////////////////4 * (8) * 32 * 32 = 128kb ///////////////////48 * 128kb = 6144kb ///////////12 * 128kb = 1536kb
__global__ void tlb_latency_test_3(int *A, int iterations, int *B, float clock_rate){	

	int index = 0;
	
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
		
	//P_chasing(0, A, 1, B, 31 * 32, clock_rate);/////warmup TLB
	P_chasing(17, A, 16, B, 17 * 524288, clock_rate);/////warmup GPU
	P_chasing(1, A, 16, B, 1 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	P_chasing(0, A, 16, B, 0 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(1, A, 16, B, 1 * 524288 + 16 * 16, clock_rate);/////try to generate TLB hit and cache miss	
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n");
}

//////////////////////////////////////////////////////4 * (8) * 32 * 32 = 128kb ///////////////////48 * 128kb = 6144kb ///////////12 * 128kb = 1536kb
__global__ void tlb_latency_test_4(int *A, int iterations, int *B, float clock_rate){	

	//P_chasing(0, A, 1, B, 31 * 32, clock_rate);/////warmup TLB
	P_chasing(1, A, 16, B, 1 * 524288, clock_rate);/////warmup
	P_chasing(0, A, 16, B, 0 * 524288, clock_rate);/////try to generate TLB hit and cache hit
	P_chasing(17, A, 16, B, 17 * 524288 + 16 * 16, clock_rate);/////try to generate TLB hit and cache miss
	P_chasing(18, A, 16, B, 18 * 524288, clock_rate);/////try to generate TLB miss and cache miss
	printf("\n");
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
		
	///////////////////////////////////////////////////////////////////CPU data begin
	int iterations = 2 * 16384 * 100;
	////////size(int) = 4, 32 = 128b, 256 = 1kb, 16 * 64 = 32 * 32 = 1024 = 4kb, 262144 = 1mb, 16384 * 32 = 512 * 1024 = 524288 = 2mb.
	int data_stride = 16;/////64b.
	//int data_size = 524288000;/////1000 * 2mb. ##### size = iteration * stride. ##### This can support 1000 iteration. The 1001st iteration starts from head again.
	int data_size = iterations * data_stride;/////size = iteration * stride = 100 2mb pages.
	
	int *CPU_data_in;	
	CPU_data_in = (int*)malloc(sizeof(int) * data_size);
	//int *CPU_data_out;
	//CPU_data_out = (int*)malloc(data_size * sizeof(int));
	
	init_cpu_data(CPU_data_in, data_size, data_stride);
	///////////////////////////////////////////////////////////////////CPU data end
	
	///////////////////////////////////////////////////////////////////GPU data begin
	int *GPU_data_in;
	//////checkCudaErrors(cudaMallocManaged(&data, sizeof(int) * data_size));
	checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
	checkCudaErrors(cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice));
	
	int *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(int) * 1));	
	///////////////////////////////////////////////////////////////////GPU data end				  
		
	tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	
	//cudaMemcpy(CPU_data_out, GPU_data_out, sizeof(int) * data_size, cudaMemcpyDeviceToHost);	
    cudaDeviceSynchronize();
	
	tlb_latency_test_2<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	cudaDeviceSynchronize();
	tlb_latency_test_2<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	cudaDeviceSynchronize();
	tlb_latency_test_2<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	cudaDeviceSynchronize();
	tlb_latency_test_3<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	cudaDeviceSynchronize();
	tlb_latency_test_3<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	cudaDeviceSynchronize();
	tlb_latency_test_3<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	cudaDeviceSynchronize();	
		
	
	tlb_latency_test_4<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	cudaDeviceSynchronize();
	/////////////check between copies
	int *GPU_data_in_2;	
	checkCudaErrors(cudaMalloc(&GPU_data_in_2, sizeof(int) * data_size));
	checkCudaErrors(cudaMemcpy(GPU_data_in_2, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice));
	
	tlb_latency_test_4<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	cudaDeviceSynchronize();
	
	tlb_latency_test_4<<<1, 1>>>(GPU_data_in_2, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	cudaDeviceSynchronize();
	
	checkCudaErrors(cudaFree(GPU_data_in_2));
	/////////////check between copies
	
	checkCudaErrors(cudaFree(GPU_data_in));
	checkCudaErrors(cudaFree(GPU_data_out));
	free(CPU_data_in);
	//free(CPU_data_out);
		
    exit(EXIT_SUCCESS);
}
