#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>

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
	printf("inside%d:%fms\n", mark, total_time / (float)clock_rate);//////clock
	
	B[0] = j;
}

__global__ void tlb_latency_test(int *A, int iterations, int *B, float clock_rate){	

	int i = 4;
	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock
	start_time = clock64();///////////clock
	
	P_chasing(0, A, i, B, 0 * 32, clock_rate);/////TLB and cache warmup
	P_chasing(1, A, i, B, 0 * 32 + 6, clock_rate);/////make them in the same page, and hit near in cache lines
	P_chasing(2, A, i, B, 0 * 32 + 7, clock_rate);/////make them in the same page, and hit near in cache lines
	P_chasing(3, A, i, B, 0 * 32 + 8, clock_rate);/////make them in the same page, and hit near in cache lines
	P_chasing(4, A, i, B, 0 * 32 + 14, clock_rate);/////////////make them in the same page, and hit far in cache lines
	P_chasing(5, A, i, B, 0 * 32 + 15, clock_rate);////////////make them in the same page, and hit far in cache lines
	P_chasing(6, A, i, B, 0 * 32 + 16, clock_rate);////////////make them in the same page, and hit far in cache lines
	P_chasing(7, A, i, B, 1 * 32, clock_rate);/////make them in the same page, and miss near in cache lines
	P_chasing(8, A, i, B, 2 * 32, clock_rate);/////make them in the same page, and miss near in cache lines
	P_chasing(9, A, i, B, 3 * 32, clock_rate);/////make them in the same page, and miss near in cache lines
	P_chasing(10, A, i, B, 8 * 32, clock_rate);//////////////make them in the same page, and miss far in cache lines
	P_chasing(11, A, i, B, 16 * 32, clock_rate);/////////////make them in the same page, and miss far in cache lines
	P_chasing(12, A, i, B, 24 * 32, clock_rate);/////////////make them in the same page, and miss far in cache lines
	P_chasing(13, A, i, B, 16 * 524288, clock_rate);//////////////TLB miss, 17th page
	P_chasing(14, A, i, B, 32 * 524288, clock_rate);/////////////TLB miss, 33rd page
	P_chasing(15, A, i, B, 48 * 524288, clock_rate);/////////////TLB miss, 49th page
	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
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
	int iterations = 100;
	////////size(int) = 4, 32 = 128b, 256 = 1kb, 32 * 32 = 1024 = 4kb, 262144 = 1mb, 16384 * 32 = 512 * 1024 = 524288 = 2mb.
	int data_stride = 524288;/////2mb. Pointing to the next page.
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
	
	int *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(int) * 1));
	
	cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
	///////////////////////////////////////////////////////////////////GPU data end				  
		
	tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	
	//cudaMemcpy(CPU_data_out, GPU_data_out, sizeof(int) * data_size, cudaMemcpyDeviceToHost);
	
    cudaDeviceSynchronize();	
	
	checkCudaErrors(cudaFree(GPU_data_in));
	checkCudaErrors(cudaFree(GPU_data_out));
	
	free(CPU_data_in);
	//free(CPU_data_out);
	
    exit(EXIT_SUCCESS);
}
