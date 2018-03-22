#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>

///////////per request timing.

//typedef unsigned char byte;

void init_cpu_data(int* A, int size, int stride, int mod){
	for (int i = 0; i < size; ++i){
		A[i]=(i + stride) % mod;
   	}
}

__device__ void P_chasing0(int mark, int *A, int iterations, int *B, int *C, long long int *D, int starting_index, float clock_rate, int data_stride){	
	
	int j = starting_index;/////make them in the same page, and miss near in cache lines
			
	for (int it = 0; it < iterations; it++){	
		j = A[j];		
	}	
		
	B[0] = j;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void P_chasing1(int mark, int *A, int iterations, int *B, int *C, long long int *D, int starting_index, float clock_rate, int data_stride){	
	
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

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void P_chasing2(int mark, int *A, int iterations, int *B, int *C, long long int *D, int starting_index, float clock_rate, int data_stride){	
	
	__shared__ long long int s_tvalue[1024 * 2];
	__shared__ int s_index[1024 * 2];
	
	int j = starting_index;/////make them in the same page, and miss near in cache lines
	//int j = B[0];
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	long long int time_interval = 0;//////clock
	//long long int total_time = end_time - start_time;//////clock
	
	/*		
	for (int it = 0; it < iterations; it++){
		
		start_time = clock64();//////clock		
		j = A[j];
		//s_index[it] = j;
		end_time=clock64();//////clock		
		s_tvalue[it] = end_time - start_time;
	}
	*/
	
		asm(".reg .u64 t1;\n\t"
		".reg .u64 t2;\n\t");
	
	for (int it = 0; it < iterations; it++){
		
		/*
		asm("mul.wide.u32 	t1, %3, %5;\n\t"	
		"add.u64 	t2, t1, %4;\n\t"		
		"mov.u64 	%0, %clock64;\n\t"		
		"ld.global.u32 	%2, [t2];\n\t"
		"mov.u64 	%1, %clock64;"
		: "=l"(start_time), "=l"(end_time), "=r"(j) : "r"(j), "l"(A), "r"(4));
		*/

		asm("mul.wide.u32 	t1, %2, %4;\n\t"	
		"add.u64 	t2, t1, %3;\n\t"		
		"mov.u64 	%0, %clock64;\n\t"		
		"ld.global.u32 	%1, [t2];\n\t"		
		: "=l"(start_time), "=r"(j) : "r"(j), "l"(A), "r"(4));
		
		s_index[it] = j;		
		asm volatile ("mov.u64 %0, %clock64;": "=l"(end_time));
		
		time_interval = end_time - start_time;
		//if(it >= 4 * 1024){
		s_tvalue[it] = time_interval;
		//}
	}
	
	//printf("inside%d:%fms\n", mark, (total_time / (float)clock_rate) / ((float)iterations));//////clock, average latency
	
	B[0] = j;
	
	for (int it = 0; it < iterations; it++){		
		C[it] = s_index[it];
		D[it] = s_tvalue[it];
	}
}

__global__ void tlb_latency_test(int *A, int iterations, int *B, int *C, long long int *D, float clock_rate, int mod, int data_stride){
	
	P_chasing0(0, A, iterations, B, C, D, 0, clock_rate, data_stride);
	//P_chasing1(0, A, iterations, B, C, D, 0, clock_rate, data_stride);
	//P_chasing1(0, A, iterations, B, C, D, 0, clock_rate, data_stride);////////saturate the L1 not L2
	//P_chasing1(0, A, iterations, B, C, D, 0, clock_rate, data_stride);////////saturate the L1 not L2
	P_chasing2(0, A, iterations, B, C, D, 0, clock_rate, data_stride);////////saturate the L1 not L2
	
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
	
	FILE * pFile;
    pFile = fopen ("output.txt","w");		
	
	for(int data_stride = 32; data_stride <= 32; data_stride = data_stride + 1){/////////stride shall be L1 cache line size.
		printf("###################data_stride%d#########################\n", data_stride);
	//for(int mod = 1024 * 256 * 2; mod > 0; mod = mod - 32 * 1024){/////kepler L2 1.5m
	for(int mod = 1024 ; mod >= 1024 ; mod = mod / 2){/////kepler L2 1.5m ////////saturate the L1 not L2
		///////////////////////////////////////////////////////////////////CPU data begin
		int data_size = 512 * 1024 * 30;/////size = iteration * stride = 30 2mb pages.		
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		int iterations = mod / data_stride * 2;
	
		int *CPU_data_in;
		CPU_data_in = (int*)malloc(sizeof(int) * data_size);	
		init_cpu_data(CPU_data_in, data_size, data_stride, mod);
		
		int *CPU_data_out_index;
		CPU_data_out_index = (int*)malloc(sizeof(int) * iterations);
		long long int *CPU_data_out_time;
		CPU_data_out_time = (long long int*)malloc(sizeof(long long int) * iterations);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		int *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		int *GPU_data_out_index;
		checkCudaErrors(cudaMalloc(&GPU_data_out_index, sizeof(int) * iterations));
		long long int *GPU_data_out_time;
		checkCudaErrors(cudaMalloc(&GPU_data_out_time, sizeof(long long int) * iterations));
		
		tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, GPU_data_out_index, GPU_data_out_time, clock_rate, mod, data_stride);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		cudaMemcpy(CPU_data_out_index, GPU_data_out_index, sizeof(int) * iterations, cudaMemcpyDeviceToHost);
		cudaMemcpy(CPU_data_out_time, GPU_data_out_time, sizeof(long long int) * iterations, cudaMemcpyDeviceToHost);
				
		for (int it = 0; it < iterations; it++){
			fprintf (pFile, "%d %fms\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate);
			//fprintf (pFile, "%d %fms\n", it, CPU_data_out_time[it] / (float)clock_rate);
		}
		
		checkCudaErrors(cudaFree(GPU_data_out_index));
		checkCudaErrors(cudaFree(GPU_data_out_time));
		checkCudaErrors(cudaFree(GPU_data_in));
		free(CPU_data_in);
		free(CPU_data_out_index);
		free(CPU_data_out_time);
	}
		printf("############################################\n\n");
	}
			
	checkCudaErrors(cudaFree(GPU_data_out));	
	//free(CPU_data_out);
	fclose (pFile);
	
    exit(EXIT_SUCCESS);
}
