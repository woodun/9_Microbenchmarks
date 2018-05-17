#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>

/////////////////////////////change the data size to larger than 16 gb to test for different memories. L1 is enabled. "ALL_CCFLAGS += -Xptxas -dlcm=ca"

void init_cpu_data(long long int* A, long long int size, long long int stride, long long int mod){
	if(1){////////////normal
		for (long long int i = 0; i < size - stride; i = i + stride){
			A[i]=(i + stride);
		}
		
		//for (long long int i = 3; i < size - stride; i = i + stride){
		//	A[i]=(i + stride);
		//}
				
		A[size - stride]=0;
		//A[size - stride + 3]=0;
	}
	
	if(0){////////////reversed
		//for (long long int i = 0; i <= size - stride; i = i + stride){
		//	A[i]=(i - stride);
		//}
		
		for (long long int i = 3; i <= size - stride + 3; i = i + stride){
			A[i]=(i - stride);
		}
		
		//A[0]=size - stride;
		A[3]=size - stride + 3;
	}
	
	/////54521859 returned page fault starting point for 2147483648.
	///////////////////2147483648 - 54521859 = 2092961789.
	///////////////////2092961789 -4096 + 3 = 1996 * 1M = 15968 MB (out of 16280 MB out of 16384 MB)
	/////2202267651 returned page fault starting point for 4294967296
	///////////////////4294967296 - 2202267651 = 2092699645.
	///////////////////2092699645 -4096 + 3 = 1995.75 * 1M = 15966 MB (out of 16280 MB out of 16384 MB)
}

timespec time_diff(timespec start, timespec end){
	timespec temp;
	if ((end.tv_nsec - start.tv_nsec) < 0){
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	} 
	else{
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	}
	return temp;
}

__device__ void P_chasing2(int mark, long long int *A, long long int iterations, long long int *B, long long int starting_index, float clock_rate, long long int data_stride){		
	
	long long int j = starting_index;
	
	//long long int start_time = 0;//////clock
	//long long int end_time = 0;//////clock
	//start_time = clock64();//////clock
			
	for (long long int it = 0; it < iterations; it++){
		j = A[j];
	}
	
	//end_time = clock64();//////clock
	//long long int total_time = end_time - start_time;//////clock
	//printf("*\n*\n*\nruntime%d: %f\n", mark, total_time / ((double)clock_rate / 1000000));//////clock, average latency //////////the print will flush the L1?!
	
	B[mark] = j;
}

__global__ void tlb_latency_test(long long int *A, long long int iterations, long long int *B, float clock_rate, long long int mod, long long int data_stride){
			
	P_chasing2(1, A, iterations/8, B, 0, clock_rate, data_stride);
	P_chasing2(2, A, iterations/8, B, 536870912, clock_rate, data_stride);
	P_chasing2(3, A, iterations/8, B, 1073741824, clock_rate, data_stride);
	P_chasing2(4, A, iterations/8, B, 1610612736, clock_rate, data_stride);
	P_chasing2(5, A, iterations/8, B, 2147483648, clock_rate, data_stride);
	P_chasing2(6, A, iterations/8, B, 2684354560, clock_rate, data_stride);
	P_chasing2(7, A, iterations/8, B, 3221225472, clock_rate, data_stride);
	P_chasing2(8, A, iterations/8, B, 3758096384, clock_rate, data_stride);
	P_chasing2(1, A, iterations/8, B, 0, clock_rate, data_stride);
	P_chasing2(2, A, iterations/8, B, 536870912, clock_rate, data_stride);
	P_chasing2(3, A, iterations/8, B, 1073741824, clock_rate, data_stride);
	P_chasing2(4, A, iterations/8, B, 1610612736, clock_rate, data_stride);
	P_chasing2(5, A, iterations/8, B, 2147483648, clock_rate, data_stride);
	P_chasing2(6, A, iterations/8, B, 2684354560, clock_rate, data_stride);
	P_chasing2(7, A, iterations/8, B, 3221225472, clock_rate, data_stride);
	P_chasing2(8, A, iterations/8, B, 3758096384, clock_rate, data_stride);
	//P_chasing2(1, A, iterations, B, 0, clock_rate, data_stride);
	//P_chasing2(0, A, iterations, B, mod - data_stride + 3, clock_rate, data_stride);
	
	__syncthreads();
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
	
	if (device_prop.concurrentManagedAccess == 1){
		printf("This device supports concurrent Managed Access.\n");
    }else{
		printf("This device does not support concurrent Managed Access.\n");
	}
	
	int value1 = 1;
	checkCudaErrors(cudaDeviceGetAttribute(&value1, cudaDevAttrConcurrentManagedAccess, dev_id));
	printf("cudaDevAttrConcurrentManagedAccess = %d\n", value1);	
	
	///////////////////////////////////////////////////////////////////GPU data out
	long long int *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(long long int) * 256));			
	
	int counter = 0;	
	for(long long int data_stride = 1 * 256 * 1024; data_stride <= 1 * 256 * 1024; data_stride = data_stride * 2){/////////32mb stride

	//plain managed
	printf("*\n*\n*\n plain managed\n");
	for(long long int mod = 4294967296; mod <= 4294967296; mod = mod * 2){////134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin
		long long int data_size = mod;
		long long int iterations = mod / data_stride;
			
		long long int *CPU_data_in;
		//CPU_data_in = (int*)malloc(sizeof(int) * data_size);
		checkCudaErrors(cudaMallocManaged(&CPU_data_in, sizeof(long long int) * data_size));/////////////using unified memory		
		init_cpu_data(CPU_data_in, data_size, data_stride, mod);		
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		//int *GPU_data_in;
		//checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		//cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);		
		
		printf("###################data_stride%lld#########################\n", data_stride);
		printf("###############Mod%lld##############%lld\n", mod, iterations);		
						
		tlb_latency_test<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);///kernel is here	
		cudaDeviceSynchronize();
		
		init_cpu_data(CPU_data_in, data_size, data_stride, mod);
		
		//checkCudaErrors(cudaFree(GPU_data_in));
		checkCudaErrors(cudaFree(CPU_data_in));
		//free(CPU_data_in);		
	}
	}
			
	checkCudaErrors(cudaFree(GPU_data_out));	
	
    exit(EXIT_SUCCESS);
}