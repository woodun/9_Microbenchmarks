#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>
#include <sys/time.h>

/////////////////////////////L1 is enabled. "ALL_CCFLAGS += -Xptxas -dlcm=ca"

void init_cpu_data(long long int* A, long long int size, long long int stride){
	
	for (long long int i = 0; i < size; i++){
		A[i]=1;
	}
	
	/*
	for (long long int i = 0; i < size - stride; i++){
		A[i]=(i + stride);
	}
			
	for (long long int i = size - stride; i < size; i++){
		A[i]=0;
	}
	*/
}

long long unsigned time_diff(timespec start, timespec end){
	struct timespec temp;
	if ((end.tv_nsec - start.tv_nsec) < 0){
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	} 
	else{
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	}
	
	long long unsigned time_long long interval_ns = temp.tv_nsec;
	long long unsigned time_long long interval_s = temp.tv_sec;
	time_long long interval_s = time_long long interval_s * 1000000000;
	
	return time_long long interval_s + time_long long interval_ns;
}

__global__ void Page_visitor(long long int *A, long long int *B, long long int data_stride, long long int clock_count){
		
	/*
	long long int index = threadIdx.x;
	
	/////////////////////////////////time
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	long long int time_long long interval = 0;//////clock
	
	if(index = 0){
		start_time= clock64();
	}
	__syncthreads();
	*/
	
	long long int index = threadIdx.x * data_stride;
	
	long long int value = A[index];
	
	//////////////////////////////////////////////sleep
	long long int start_clock = clock64();
    long long int clock_offset = 0;
    while (clock_offset < clock_count)
    {
        clock_offset = clock64() - start_clock;
    }
    //d_o[0] = clock_offset;
	//////////////////////////////////////////////sleep
	
	B[index] = value;
	
	/*
	__syncthreads();
	/////////////////////////////////time
	if(index = 0){
		start_time= clock64();
		time_long long interval = end_time - start_time;//////clock
	}	
	//B[0] = time_long long interval;
	*/
}

long long int main(long long int argc, char **argv)
{
	prlong long intf("\n");
	
    // set device
    cudaDeviceProp device_prop;
    //long long int dev_id = findCudaDevice(argc, (const char **) argv);
	long long int dev_id = 0;
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));
	
	long long int peak_clk = 1;//kHz
	checkCudaErrors(cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, dev_id));
	float clock_rate = (float) peak_clk;
	
	prlong long intf("clock_rate:%f\n", clock_rate);

    if (!device_prop.managedMemory) { 
        // This samples requires being run on a device that supports Unified Memory
        fprlong long intf(stderr, "Unified Memory not supported on this device\n");

        exit(EXIT_WAIVED);
    }

    if (device_prop.computeMode == cudaComputeModeProhibited)
    {
        // This sample requires being run with a default or process exclusive mode
        fprlong long intf(stderr, "This sample requires a device in either default or process exclusive mode\n");

        exit(EXIT_WAIVED);
    }
	
	if (device_prop.concurrentManagedAccess == 1){
		prlong long intf("This device supports concurrent Managed Access.\n");
    }else{
		prlong long intf("This device does not support concurrent Managed Access.\n");
	}
	
	long long int value1 = 1;
	checkCudaErrors(cudaDeviceGetAttribute(&value1, cudaDevAttrConcurrentManagedAccess, dev_id));
	prlong long intf("cudaDevAttrConcurrentManagedAccess = %d\n", value1);	
	
	//plain managed
	prlong long intf("*\n*\n*\n plain managed\n");
	for(long long int data_stride = 1 * 256 * 1024; data_stride <= 1 * 256 * 1024; data_stride = data_stride * 2){
	for(long long int mod = 4294967296; mod <= 4294967296; mod = mod * 2){////134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
	for(long long int clock_count = 1000; clock_count <= 1000; clock_count = clock_count * 2){
		///////////////////////////////////////////////////////////////////CPU data begin		
		long long int data_size = mod;
		//long long int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		long long int *CPU_data_in;
		//CPU_data_in = (long long int*)malloc(sizeof(long long int) * data_size);
		checkCudaErrors(cudaMallocManaged(&CPU_data_in, sizeof(long long int) * data_size));/////////////using unified memory		
		init_cpu_data(CPU_data_in, data_size, data_stride);				
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		//long long int *GPU_data_in;
		//checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(long long int) * data_size));	
		//cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(long long int) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		long long int *GPU_data_out;
		//checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(long long int) * 2));
		checkCudaErrors(cudaMallocManaged(&GPU_data_out, sizeof(long long int) * data_size));/////////////using unified memory		
				
		/////////////////////////////////time
		struct timespec ts1;
		clock_gettime(CLOCK_REALTIME, &ts1);
  
		Page_visitor<<<8, 2048>>>(CPU_data_in, GPU_data_out, data_stride, clock_count);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		/////////////////////////////////time
		struct timespec ts2;
		clock_gettime(CLOCK_REALTIME, &ts2);
		
		prlong long intf("###################data_stride%d#########################clock_count:%lld\n", data_stride, clock_count);
		prlong long intf("*\n*\n*\nruntime:  %lluns\n", time_diff(ts1, ts2));	
		
		//checkCudaErrors(cudaFree(GPU_data_in));
		checkCudaErrors(cudaFree(CPU_data_in));
		//free(CPU_data_in);
		checkCudaErrors(cudaFree(GPU_data_out));
	}
	}
	}
	
	/*
	//memcopy
	prlong long intf("*\n*\n*\n memcopy\n");
	for(long long int data_stride = 1 * 128 * 1024; data_stride <= 2 * 256 * 1024; data_stride = data_stride * 2){
	for(long long int mod = 536870912; mod <= 536870912; mod = mod * 2){////268435456 = 1gb, 536870912 = 2gb, 1073741824 = 4gb, 2147483648 = 8gb, 4294967296 = 16gb, 8589934592 = 32gb.	
	for(long long int clock_count = 1000; clock_count <= 1000; clock_count = clock_count * 2){

		///////////////////////////////////////////////////////////////////CPU data begin		
		long long int data_size = mod;
		//long long int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		long long int *CPU_data_in;
		CPU_data_in = (long long int*)malloc(sizeof(long long int) * data_size);		
		init_cpu_data(CPU_data_in, data_size, data_stride);				
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		long long int *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(long long int) * data_size));	
		
		///////////////////////////////////////////////////////////////////GPU data out
		long long int *GPU_data_out;
		checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(long long int) * data_size));
				
		/////////////////////////////////time
		struct timespec ts1;
		clock_gettime(CLOCK_REALTIME, &ts1);

		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(long long int) * data_size, cudaMemcpyHostToDevice);
		Page_visitor<<<1, 512>>>(GPU_data_in, GPU_data_out, data_stride, clock_count);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		/////////////////////////////////time
		struct timespec ts2;
		clock_gettime(CLOCK_REALTIME, &ts2);
		
		prlong long intf("###################data_stride%d#########################clock_count:%lld\n", data_stride, clock_count);
		prlong long intf("*\n*\n*\nruntime:  %lluns\n", time_diff(ts1, ts2));
		
		checkCudaErrors(cudaFree(GPU_data_in));
		//checkCudaErrors(cudaFree(CPU_data_in));
		free(CPU_data_in);
		checkCudaErrors(cudaFree(GPU_data_out));
	}
	}
	}
	*/
	
    exit(EXIT_SUCCESS);
}