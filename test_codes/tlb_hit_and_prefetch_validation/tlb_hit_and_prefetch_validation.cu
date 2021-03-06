#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>

//////////cache flush test: can I test multiple kernels in the same run? will they cause cache hits? then I can launch different strides to figure out if the tlb and cache miss or not.

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

//////////min page size 4kb = 4096b = 32 * 128 = 32 * 4 * 32.
__device__ void tlb_warmup(int *A, int iterations, int *B, float clock_rate){
		
	//iterations = 8;///////should not saturate the tlb
	
	int j = 31 * 32;/////make them in the same page, but far in cache lines
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}	
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside warmup:%fms\n", total_time / (float)clock_rate);//////clock
	
	B[0] = j;	
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_1(int *A, int iterations, int *B, float clock_rate){//////////////should hit in the tlb, but miss in the cache, to prove tlb hit exists.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 0;/////make them in the same page, but far in cache lines
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside1:%fms\n", total_time / (float)clock_rate);//////clock
	
	B[0] = j;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_2(int *A, int iterations, int *B, float clock_rate){//////////////should hit in the tlb, but miss in the cache, to prove tlb hit exists.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 8 * 32;/////make them in the same page, but far in cache lines
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside2:%fms\n", total_time / (float)clock_rate);//////clock
	
	B[0] = j;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_3(int *A, int iterations, int *B, float clock_rate){//////////////should hit in the tlb, but miss in the cache, to prove tlb hit exists.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 16 * 32;/////make them in the same page, but far in cache lines
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside3:%fms\n", total_time / (float)clock_rate);//////clock
	
	B[0] = j;
}


//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_4(int *A, int iterations, int *B, float clock_rate){//////////////should not hit in the tlb, and should also miss in the cache, to see the time difference.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 524288 * 32;/////make them in the different page, 524288 = 2mb. The 33th page.
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside4:%fms\n", total_time / (float)clock_rate);//////clock
	
	B[0] = j;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_5(int *A, int iterations, int *B, float clock_rate){//////////////should not hit in the tlb, and should also miss in the cache, to see the time difference.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 524288 * 64;/////make them in the different page, 524288 = 2mb. The 65th page.
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside5:%fms\n", total_time / (float)clock_rate);//////clock
	
	B[0] = j;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_6(int *A, int iterations, int *B, float clock_rate){//////////////should not hit in the tlb, and should also miss in the cache, to see the time difference.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 524288 * 96;/////make them in the different page, 524288 = 2mb. The 97th page.
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside6:%fms\n", total_time / (float)clock_rate);//////clock
	
	B[0] = j;
}


__global__ void tlb_latency_test(int *A, int iterations, int *B, float clock_rate){
	
	//int j = 0;	
	//for (int it =0; it < iterations; it ++){
	//	j = A[j];
	//}	
	//B[0] = j;

	long long int start_time = 0;///////////clock
	long long int end_time = 0;///////////clock	
	start_time = clock64();///////////clock
	tlb_warmup(A, 16, B, clock_rate);	
	cache_miss_1(A, 16, B, clock_rate);
	cache_miss_2(A, 16, B, clock_rate);
	cache_miss_3(A, 16, B, clock_rate);
	cache_miss_4(A, 16, B, clock_rate);
	cache_miss_5(A, 16, B, clock_rate);
	cache_miss_6(A, 16, B, clock_rate);	
	end_time=clock64();///////////clock
		
	long long int total_time = end_time - start_time;///////////clock
	printf("outside1:%fms\n", total_time / (float)clock_rate);///////////clock
}

int main(int argc, char **argv)
{
	printf("\n");//////clock
	
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
	
	//printf("%d\n", sizeof(int));//////////size of int is 4.
	//exit(0);
	
	//////////////CPU data begin
	////////size(int) = 4, 256 = 1kb, 262144 = 1mb, 524288 = 2mb.
	int iterations = 1000;
	int data_stride = 524288;/////2mb. Pointing to the next page.
	//int data_size = 524288000;/////1000 * 2mb. ##### size = iteration * stride. ##### This can support 1000 iteration. The 1001st iteration starts from head again.
	int data_size = iterations * data_stride;/////size = iteration * stride = 1000 pages.
	
	int *CPU_data_in;	
	CPU_data_in = (int*)malloc(sizeof(int) * data_size);
	//int *CPU_data_out;
	//CPU_data_out = (int*)malloc(data_size * sizeof(int));
	
	init_cpu_data(CPU_data_in, data_size, data_stride);
	//////////////CPU data end
	
	//////////////GPU data begin
	int *GPU_data_in;
	//////checkCudaErrors(cudaMallocManaged(&data, sizeof(int) * data_size));
	checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));
	
	int *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(int) * 1));
	
	cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
	//////////////GPU data end
				
    //cudaEvent_t start, stop;////////events timer is not accurate.
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);	

	//cudaEventRecord(start);////////events timer
	//cudaEventSynchronize(start);
	
	///////////CPU timer also becomes inaccurate when events timer is not used.
	//struct timespec ts_start, ts_end;
	//clock_gettime(CLOCK_REALTIME, &ts_start);///////////CPU timer
	
	tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);//////////////////////////////////////////////kernel is here
	
	///////////CPU timer
	//clock_gettime(CLOCK_REALTIME, &ts_end);///////////CPU timer
	
	//cudaEventRecord(stop);////////events timer
	//cudaEventSynchronize(stop);
	
	//cudaMemcpy(CPU_data_out, GPU_data_out, sizeof(int) * data_size, cudaMemcpyDeviceToHost);
	
    cudaDeviceSynchronize();
	
	///////////CPU timer
	//printf("CPU clock: %fms\n", (double)(ts_end.tv_nsec - ts_start.tv_nsec) / 1000000);///////////CPU timer
	
	//float milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("out kernel:%fms %fms\n", milliseconds, milliseconds / iterations);
	
	checkCudaErrors(cudaFree(GPU_data_in));
	checkCudaErrors(cudaFree(GPU_data_out));
	
	free(CPU_data_in);
	//free(CPU_data_out);
		
    exit(EXIT_SUCCESS);
}
