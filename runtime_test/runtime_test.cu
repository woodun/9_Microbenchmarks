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

//typedef unsigned char byte;

void init_cpu_data(int* A, int size, int stride){
	
	for (int i = 0; i < size; i++){
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

__global__ void Page_visitor(int *A, int *B, int data_stride, long long int clock_count){
		
	/*
	int index = threadIdx.x;
	
	/////////////////////////////////time
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	long long int time_interval = 0;//////clock
	
	if(index = 0){
		start_time= clock64();
	}
	__syncthreads();
	*/
	
	int index = threadIdx.x * data_stride;
	
	int value = A[index];
	
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
		time_interval = end_time - start_time;//////clock
	}	
	//B[0] = time_interval;
	*/
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
		
	FILE * pFile;
    pFile = fopen ("output.txt","w");		
	
	int counter = 0;
	
	//plain managed
	printf("*\n*\n*\n plain managed\n");
	for(int data_stride = 1 * 128 * 1024; data_stride <= 2 * 256 * 1024; data_stride = data_stride * 2){

	for(int mod = 268435456; mod <= 268435456; mod = mod * 2){////268435456 = 1gb, 536870912 = 2gb, 1073741824 = 4gb, 2147483648 = 8gb, 4294967296 = 16gb, 8589934592 = 32gb.
	
	for(long long int clock_count = 1000; clock_count <= 1000; clock_count = clock_count * 2){
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin		
		int data_size = mod;
		if(data_size < 4194304){//////////data size at least 16mb to prevent L2 prefetch
			data_size = 4194304;
		}		
		//int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		int *CPU_data_in;
		//CPU_data_in = (int*)malloc(sizeof(int) * data_size);
		checkCudaErrors(cudaMallocManaged(&CPU_data_in, sizeof(int) * data_size));/////////////using unified memory		
		init_cpu_data(CPU_data_in, data_size, data_stride);				
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		//int *GPU_data_in;
		//checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		//cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		int *GPU_data_out;
		//checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(int) * 2));
		checkCudaErrors(cudaMallocManaged(&GPU_data_out, sizeof(int) * data_size));/////////////using unified memory		
				
		/////////////////////////////////time
		struct timespec ts1;
		clock_gettime(CLOCK_REALTIME, &ts1);

		//printf("s:  %lu\n", ts1.tv_sec);
		//printf("ns: %lu\n", ts1.tv_nsec);
  
		Page_visitor<<<1, 512>>>(CPU_data_in, GPU_data_out, data_stride, clock_count);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		/////////////////////////////////time
		struct timespec ts2;
		clock_gettime(CLOCK_REALTIME, &ts2);

		//printf("s:  %lu\n", ts2.tv_sec);
		//printf("ns: %lu\n", ts2.tv_nsec);
		//printf("s:  %lu\n", ts2.tv_sec - ts1.tv_sec);
		
		printf("###################data_stride%d#########################clock_count:%lld\n", data_stride, clock_count);
		printf("runtime:  %luns\n", ts2.tv_nsec - ts1.tv_nsec);		
		
		//checkCudaErrors(cudaFree(GPU_data_in));
		checkCudaErrors(cudaFree(CPU_data_in));
		//free(CPU_data_in);
		checkCudaErrors(cudaFree(GPU_data_out));
	}
	}
	}
	
	/*
	//preferredlocation
	fprintf(pFile,"*\n*\n*\n preferredlocation\n");
	fflush(pFile);
	for(long long int mod2 = 1073741824; mod2 <= 4294967296; mod2 = mod2 * 2){////268435456 = 1gb, 536870912 = 2gb, 1073741824 = 4gb, 2147483648 = 8gb, 4294967296 = 16gb, 8589934592 = 32gb.
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin
		//int data_size = 2 * 256 * 1024 * 32;/////size = iteration * stride = 32 2mb pages.
		long long int mod = mod2;

		long long int data_size = mod;
		if(data_size < 4194304){//////////data size at least 16mb to prevent L2 prefetch
			data_size = 4194304;
		}	
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		long long int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		long long int *CPU_data_in;
		//CPU_data_in = (int*)malloc(sizeof(int) * data_size);
		checkCudaErrors(cudaMallocManaged(&CPU_data_in, sizeof(long long int) * data_size));/////////////using unified memory
		checkCudaErrors(cudaMemAdvise(CPU_data_in, sizeof(long long int) * data_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));////////using hint		
		init_cpu_data(CPU_data_in, data_size, data_stride, mod);
		
		long long int reduced_iter = iterations;
		if(reduced_iter > 2048){
			reduced_iter = 2048;
		}else if(reduced_iter < 16){
			reduced_iter = 16;
		}
		
		long long int *CPU_data_out_index;
		CPU_data_out_index = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		long long int *CPU_data_out_time;
		CPU_data_out_time = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		//int *GPU_data_in;
		//checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		//cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		long long int *GPU_data_out_index;
		checkCudaErrors(cudaMalloc(&GPU_data_out_index, sizeof(long long int) * reduced_iter));
		long long int *GPU_data_out_time;
		checkCudaErrors(cudaMalloc(&GPU_data_out_time, sizeof(long long int) * reduced_iter));
		
		tlb_latency_test<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, GPU_data_out_index, GPU_data_out_time, clock_rate, mod, data_stride);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		cudaMemcpy(CPU_data_out_index, GPU_data_out_index, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
		cudaMemcpy(CPU_data_out_time, GPU_data_out_time, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
				

		fprintf(pFile, "###################data_stride%d#########################\n", data_stride);
		fprintf (pFile, "###############Mod%lld##############%lld\n", mod, iterations);
		for (long long int it = 0; it < reduced_iter; it++){		
			fprintf (pFile, "%lld %fms %lldcycles\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate, CPU_data_out_time[it]);
			//fprintf (pFile, "%d %fms\n", it, CPU_data_out_time[it] / (float)clock_rate);
			//printf ("%d %fms\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate);
		}
		fflush(pFile);
		
		checkCudaErrors(cudaFree(GPU_data_out_index));
		checkCudaErrors(cudaFree(GPU_data_out_time));
		//checkCudaErrors(cudaFree(GPU_data_in));
		checkCudaErrors(cudaFree(CPU_data_in));
		//free(CPU_data_in);
		free(CPU_data_out_index);
		free(CPU_data_out_time);
	}
	
	//accessedby
	fprintf(pFile,"*\n*\n*\n accessedby\n");
	fflush(pFile);
	for(long long int mod2 = 1073741824; mod2 <= 4294967296; mod2 = mod2 * 2){////268435456 = 1gb, 536870912 = 2gb, 1073741824 = 4gb, 2147483648 = 8gb, 4294967296 = 16gb, 8589934592 = 32gb.
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin
		//int data_size = 2 * 256 * 1024 * 32;/////size = iteration * stride = 32 2mb pages.
		long long int mod = mod2;

		long long int data_size = mod;
		if(data_size < 4194304){//////////data size at least 16mb to prevent L2 prefetch
			data_size = 4194304;
		}	
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		long long int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		long long int *CPU_data_in;
		//CPU_data_in = (int*)malloc(sizeof(int) * data_size);
		checkCudaErrors(cudaMallocManaged(&CPU_data_in, sizeof(long long int) * data_size));/////////////using unified memory
		checkCudaErrors(cudaMemAdvise(CPU_data_in, sizeof(long long int) * data_size, cudaMemAdviseSetAccessedBy, dev_id));//////////using hint	
		init_cpu_data(CPU_data_in, data_size, data_stride, mod);
		
		long long int reduced_iter = iterations;
		if(reduced_iter > 2048){
			reduced_iter = 2048;
		}else if(reduced_iter < 16){
			reduced_iter = 16;
		}
		
		long long int *CPU_data_out_index;
		CPU_data_out_index = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		long long int *CPU_data_out_time;
		CPU_data_out_time = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		//int *GPU_data_in;
		//checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));
		//cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		long long int *GPU_data_out_index;
		checkCudaErrors(cudaMalloc(&GPU_data_out_index, sizeof(long long int) * reduced_iter));
		long long int *GPU_data_out_time;
		checkCudaErrors(cudaMalloc(&GPU_data_out_time, sizeof(long long int) * reduced_iter));
		
		tlb_latency_test<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, GPU_data_out_index, GPU_data_out_time, clock_rate, mod, data_stride);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		cudaMemcpy(CPU_data_out_index, GPU_data_out_index, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
		cudaMemcpy(CPU_data_out_time, GPU_data_out_time, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
				

		fprintf(pFile, "###################data_stride%d#########################\n", data_stride);
		fprintf (pFile, "###############Mod%lld##############%lld\n", mod, iterations);
		for (long long int it = 0; it < reduced_iter; it++){
			fprintf (pFile, "%lld %fms %lldcycles\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate, CPU_data_out_time[it]);
			//fprintf (pFile, "%d %fms\n", it, CPU_data_out_time[it] / (float)clock_rate);
			//printf ("%d %fms\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate);
		}
		fflush(pFile);
		
		checkCudaErrors(cudaFree(GPU_data_out_index));
		checkCudaErrors(cudaFree(GPU_data_out_time));
		//checkCudaErrors(cudaFree(GPU_data_in));
		checkCudaErrors(cudaFree(CPU_data_in));
		//free(CPU_data_in);
		free(CPU_data_out_index);
		free(CPU_data_out_time);
	}
	
	//pinned
	fprintf(pFile,"*\n*\n*\n pinned\n");
	fflush(pFile);
	for(long long int mod2 = 1073741824; mod2 <= 4294967296; mod2 = mod2 * 2){////268435456 = 1gb, 536870912 = 2gb, 1073741824 = 4gb, 2147483648 = 8gb, 4294967296 = 16gb, 8589934592 = 32gb.
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin
		//int data_size = 2 * 256 * 1024 * 32;/////size = iteration * stride = 32 2mb pages.
		long long int mod = mod2;

		long long int data_size = mod;
		if(data_size < 4194304){//////////data size at least 16mb to prevent L2 prefetch
			data_size = 4194304;
		}	
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		long long int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		long long int *CPU_data_in;
		//CPU_data_in = (int*)malloc(sizeof(int) * data_size);
		checkCudaErrors(cudaHostAlloc((void**)&CPU_data_in, sizeof(long long int) * data_size, cudaHostAllocDefault));//////////using pinned memory	
		init_cpu_data(CPU_data_in, data_size, data_stride, mod);
		
		long long int reduced_iter = iterations;
		if(reduced_iter > 2048){
			reduced_iter = 2048;
		}else if(reduced_iter < 16){
			reduced_iter = 16;
		}
		
		long long int *CPU_data_out_index;
		CPU_data_out_index = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		long long int *CPU_data_out_time;
		CPU_data_out_time = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		//int *GPU_data_in;
		//checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		//cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		long long int *GPU_data_out_index;
		checkCudaErrors(cudaMalloc(&GPU_data_out_index, sizeof(long long int) * reduced_iter));
		long long int *GPU_data_out_time;
		checkCudaErrors(cudaMalloc(&GPU_data_out_time, sizeof(long long int) * reduced_iter));
		
		tlb_latency_test<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, GPU_data_out_index, GPU_data_out_time, clock_rate, mod, data_stride);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		cudaMemcpy(CPU_data_out_index, GPU_data_out_index, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
		cudaMemcpy(CPU_data_out_time, GPU_data_out_time, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
				

		fprintf(pFile, "###################data_stride%d#########################\n", data_stride);
		fprintf (pFile, "###############Mod%lld##############%lld\n", mod, iterations);
		for (long long int it = 0; it < reduced_iter; it++){		
			fprintf (pFile, "%lld %fms %lldcycles\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate, CPU_data_out_time[it]);
			//fprintf (pFile, "%d %fms\n", it, CPU_data_out_time[it] / (float)clock_rate);
			//printf ("%d %fms\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate);
		}
		fflush(pFile);
		
		checkCudaErrors(cudaFree(GPU_data_out_index));
		checkCudaErrors(cudaFree(GPU_data_out_time));
		//checkCudaErrors(cudaFree(GPU_data_in));
		//checkCudaErrors(cudaFree(CPU_data_in));
		checkCudaErrors(cudaFreeHost(CPU_data_in));//////using pinned memory
		//free(CPU_data_in);
		free(CPU_data_out_index);
		free(CPU_data_out_time);
	}
	
	//memcopy
	fprintf(pFile,"*\n*\n*\n memcopy\n");
	fflush(pFile);
	for(long long int mod2 = 1073741824; mod2 <= 4294967296; mod2 = mod2 * 2){////268435456 = 1gb, 536870912 = 2gb, 1073741824 = 4gb, 2147483648 = 8gb, 4294967296 = 16gb, 8589934592 = 32gb.
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin
		//int data_size = 2 * 256 * 1024 * 32;/////size = iteration * stride = 32 2mb pages.
		long long int mod = mod2;

		long long int data_size = mod;
		if(data_size < 4194304){//////////data size at least 16mb to prevent L2 prefetch
			data_size = 4194304;
		}	
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		long long int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		long long int *CPU_data_in;
		CPU_data_in = (long long int*)malloc(sizeof(long long int) * data_size);
		init_cpu_data(CPU_data_in, data_size, data_stride, mod);
		
		long long int reduced_iter = iterations;
		if(reduced_iter > 2048){
			reduced_iter = 2048;
		}else if(reduced_iter < 16){
			reduced_iter = 16;
		}
		
		long long int *CPU_data_out_index;
		CPU_data_out_index = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		long long int *CPU_data_out_time;
		CPU_data_out_time = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		long long int *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(long long int) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(long long int) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		long long int *GPU_data_out_index;
		checkCudaErrors(cudaMalloc(&GPU_data_out_index, sizeof(long long int) * reduced_iter));
		long long int *GPU_data_out_time;
		checkCudaErrors(cudaMalloc(&GPU_data_out_time, sizeof(long long int) * reduced_iter));
		
		tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, GPU_data_out_index, GPU_data_out_time, clock_rate, mod, data_stride);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		cudaMemcpy(CPU_data_out_index, GPU_data_out_index, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
		cudaMemcpy(CPU_data_out_time, GPU_data_out_time, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
				

		fprintf(pFile, "###################data_stride%d#########################\n", data_stride);
		fprintf (pFile, "###############Mod%lld##############%lld\n", mod, iterations);
		for (long long int it = 0; it < reduced_iter; it++){		
			fprintf (pFile, "%lld %fms %lldcycles\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate, CPU_data_out_time[it]);
			//fprintf (pFile, "%d %fms\n", it, CPU_data_out_time[it] / (float)clock_rate);
			//printf ("%d %fms\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate);
		}
		fflush(pFile);
		
		checkCudaErrors(cudaFree(GPU_data_out_index));
		checkCudaErrors(cudaFree(GPU_data_out_time));
		checkCudaErrors(cudaFree(GPU_data_in));	
		free(CPU_data_in);
		free(CPU_data_out_index);
		free(CPU_data_out_time);
	}
	*/

	fclose (pFile);
	
    exit(EXIT_SUCCESS);
}