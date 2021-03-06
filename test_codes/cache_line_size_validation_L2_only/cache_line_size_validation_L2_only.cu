#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>

/////////////////access different range of data to check the cacheline size. L1 is likely saturated in this example. L1 can also be disabled to check if it is from L1 or L2.

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
__device__ void P_chasing(int mark, int *A, int iterations, int *B, int starting_index, float clock_rate){//////////////should not hit in the tlb, and should also miss in the cache, to see the time difference.
	
	int j = starting_index;/////make them in the same page, and miss near in cache lines
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside%d:%fms\n", mark, ((total_time / (float)clock_rate))/ (float) iterations);//////clock
	
	B[0] = j;
}

__global__ void tlb_latency_test(int *A, int iterations, int *B, float clock_rate, int iter){///why 128 does not saturate the cache? (volta set number might be strange. try with other stride value)
	
	printf("iter%d:\n", iter);
	
	//long long int start_time = 0;///////////clock
	//long long int end_time = 0;///////////clock	
	//start_time = clock64();///////////clock
	
	P_chasing(0, A, 7, B, 2072 * 524288, clock_rate);/////GPU warmup
	P_chasing(0, A, iter, B, 0 * 32, clock_rate);/////TLB miss and cache miss /////////(1) (TLB miss and cache miss VS. TLB hit cache miss) applied
	P_chasing(1, A, iter, B, 0 * 32 + 1, clock_rate);/////make them in the same page, and hit near in cache lines
	P_chasing(4, A, iter, B, 0 * 32 + 4, clock_rate);/////make them in the same page, and hit near in cache lines	
	P_chasing(7, A, iter, B, 0 * 32 + 7, clock_rate);/////////////make them in the same page, and hit far in cache lines
	P_chasing(8, A, iter, B, 0 * 32 + 8, clock_rate);/////////////make them in the same page, and hit far in cache lines
	P_chasing(9, A, iter, B, 0 * 32 + 9, clock_rate);/////////////make them in the same page, and hit far in cache lines
	//P_chasing(7, A, iter, B, 0 * 32 + 7, clock_rate);/////////////make them in the same page, and hit far in cache lines
	P_chasing(15, A, iter, B, 0 * 32 + 15, clock_rate);////////////make them in the same page, and hit far in cache lines
	P_chasing(16, A, iter, B, 0 * 32 + 16, clock_rate);////////////make them in the same page, and hit far in cache lines
	P_chasing(24, A, iter, B, 0 * 32 + 24, clock_rate);////////////make them in the same page, and hit far in cache lines
	P_chasing(1, A, iter, B, 1 * 32, clock_rate);/////make them in the same page, and miss near in cache lines
	P_chasing(4, A, iter, B, 4 * 32, clock_rate);/////make them in the same page, and miss near in cache lines
	P_chasing(8, A, iter, B, 8 * 32, clock_rate);//////////////make them in the same page, and miss far in cache lines
	P_chasing(16, A, iter, B, 16 * 32, clock_rate);/////////////TLB hit and cache miss /////////(3) (TLB hit and cache hit VS. TLB hit cache miss) applied
	P_chasing(16, A, iter, B, 16 * 32, clock_rate);/////////////TLB hit and cache hit /////////(3) (it is beter to apply this in consecutive pattern to use a large amount)
	P_chasing(24, A, iter, B, 24 * 32, clock_rate);/////////////TLB hit and cache miss /////////(1) (no cache prefetch? no TLB miss? cache way saturated?)
	P_chasing(24, A, iter, B, 24 * 32, clock_rate);/////////////TLB hit and cache hit /////////(2) (TLB miss and cache hit(hard to make) VS. TLB hit and cache hit) not applied
	P_chasing(7, A, iter, B, 0 * 32 + 7, clock_rate);/////////////is this still there?
		
	//end_time=clock64();///////////clock		
	//long long int total_time = end_time - start_time;///////////clock
	//printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
	printf("\n", iter);
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
		
	///////////////////////////////////////////////////////////////////CPU data begin
	////////size(int) = 4, 256 = 1kb, 262144 = 1mb, 524288 = 2mb.
	int iterations = 2100;
	int data_stride = 524288;/////2mb. Pointing to the next page.
	//int data_size = 524288000;/////1000 * 2mb. ##### size = iteration * stride. ##### This can support 1000 iteration. The 1001st iteration starts from head again.
	int data_size = iterations * data_stride;/////size = iteration * stride = 200 pages.
	
	int *CPU_data_in;	
	CPU_data_in = (int*)malloc(sizeof(int) * data_size);	
	init_cpu_data(CPU_data_in, data_size, data_stride);
	///////////////////////////////////////////////////////////////////CPU data end
	
	///////////////////////////////////////////////////////////////////GPU data out
	int *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(int) * 1));
	

	for(int iter = 2048; iter > 0; iter = iter / 2){
	///////////////////////////////////////////////////////////////////GPU data in
		int *GPU_data_in2;	
		checkCudaErrors(cudaMalloc(&GPU_data_in2, sizeof(int) * data_size));
		cudaMemcpy(GPU_data_in2, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
			
		tlb_latency_test<<<1, 1>>>(GPU_data_in2, iterations, GPU_data_out, clock_rate, iter);//////////////////////////////////////////////kernel is here			
		cudaDeviceSynchronize();		
	
		checkCudaErrors(cudaFree(GPU_data_in2));
	}
	
	
	checkCudaErrors(cudaFree(GPU_data_out));	
	free(CPU_data_in);
	
    exit(EXIT_SUCCESS);
}
