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

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void tlb_warmup(int *A, int iterations, int *B, float clock_rate){
	
	long long int start_time = 0;
	long long int end_time = 0;
	
	start_time = clock64();
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 31;/////make them in the same page, but far in cache lines
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}	
	
	B[0] = j;
	
	end_time=clock64();
	long long int total_time = end_time - start_time;
	printf("inside:%lld\n", total_time);
	printf("inside:%fms\n", total_time / (float)clock_rate);

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
__device__ void cache_miss_1(int *A, int iterations, int *B){//////////////should hit in the tlb, but miss in the cache, to prove tlb hit exists.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 0;/////make them in the same page, but far in cache lines
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	B[0] = j;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_2(int *A, int iterations, int *B){//////////////should hit in the tlb, but miss in the cache, to prove tlb hit exists.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 8;/////make them in the same page, but far in cache lines
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	B[0] = j;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_3(int *A, int iterations, int *B){//////////////should hit in the tlb, but miss in the cache, to prove tlb hit exists.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 16;/////make them in the same page, but far in cache lines
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	B[0] = j;
}


//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_4(int *A, int iterations, int *B){//////////////should not hit in the tlb, and should also miss in the cache, to see the time difference.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 8388608;/////make them in the different page, 524288 * 16 = 8388608. 2m * 16. The 17th page.
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	B[0] = j;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_5(int *A, int iterations, int *B){//////////////should not hit in the tlb, and should also miss in the cache, to see the time difference.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 16777216;/////make them in the different page, 524288 * 32 = 8388608. 2m * 32. The 33rd page.
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	B[0] = j;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void cache_miss_6(int *A, int iterations, int *B){//////////////should not hit in the tlb, and should also miss in the cache, to see the time difference.
	
	//iterations = 8;///////should not saturate the tlb
	
	int j = 16777216;/////make them in the different page, 524288 * 48 = 25165824. 2m * 48. The 49th page.
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}
	
	B[0] = j;
}


__global__ void tlb_latency_test(int *A, int iterations, int *B, float clock_rate){
	
	//int j = 0;	
	//for (int it =0; it < iterations; it ++){
	//	j = A[j];
	//}	
	//B[0] = j;

	long long int start_time = 0;
	long long int end_time = 0;
	
	long long int end_time2 = 0;
	
	start_time = clock64();
		
	tlb_warmup(A, 8, B, clock_rate);	
	
	end_time2=clock64();
		
	cache_miss_1(A, 8, B);
	cache_miss_2(A, 8, B);
	cache_miss_3(A, 8, B);
	cache_miss_4(A, 8, B);
	cache_miss_5(A, 8, B);
	cache_miss_6(A, 8, B);
	
	end_time=clock64();	
	
	long long int total_time2 = end_time2 - start_time;	
	printf("outside2:%fms\n", total_time2 / (float)clock_rate);
	
	long long int total_time = end_time - start_time;
	printf("outside1:%fms\n", total_time / (float)clock_rate);
}

int main(int argc, char **argv)
{
	
    // set device
    cudaDeviceProp device_prop;
    //int dev_id = findCudaDevice(argc, (const char **) argv);
	int dev_id = 0;
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));
	
	int peak_clk = 1;//kHz
	checkCudaErrors(cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, dev_id));
	float clock_rate = (float) peak_clk;
	
	printf("clock_rate_out_kernel:%f\n", clock_rate);

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
	int data_stride = 524288;/////2mb.
	//int data_size = 524288000;/////1000 * 2mb. ##### size = iteration * stride. ##### This can support 1000 iteration. The 1001st iteration starts from head again.
	int data_size = iterations * data_stride;/////size = iteration * stride.
	
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
				
    cudaEvent_t start, stop;////////events timer is not accurate.
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	

	//////////////kernel begin
	cudaEventRecord(start);	
	cudaEventSynchronize(start);
	
	///////////CPU timer
	struct timespec ts_start, ts_end;
	clock_gettime(CLOCK_REALTIME, ts_start);///////////CPU timer
	
	tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate);
	
	///////////CPU timer
	clock_gettime(CLOCK_REALTIME, ts_end);///////////CPU timer
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	//////////////kernel end
	
	//cudaMemcpy(CPU_data_out, GPU_data_out, sizeof(int) * data_size, cudaMemcpyDeviceToHost);
	
    cudaDeviceSynchronize();
	
	///////////CPU timer
	printf("CPU clock: %lu\n", ts_end.tv_nsec - ts_start.tv_nsec);///////////CPU timer
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
    printf("out kernel:%f %f\n", milliseconds, milliseconds / iterations);
	
    exit(EXIT_SUCCESS);
}
