#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>

/////////////saturate L2 with long consecutive data. this one use the method in the paper which initialize the data multiple times. L1 is disabled with "ALL_CCFLAGS += -Xptxas -dlcm=cg"
/////////////see what happens with byte data in the ptx. can we eliminate the multiplication? is there change in latency?

void init_cpu_data(unsigned char* A, unsigned char size, unsigned char stride, int mod){
	for (unsigned char i = 0; i <= size; ++i){
		A[i] = (unsigned char)((i + stride) % mod);
   	}
}

void init_cpu_data2(int* A, int size, int stride, int mod){
	for (int i = 0; i < size; ++i){
		A[i]=(i + stride) % mod;
   	}
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void P_chasing(int mark, unsigned char *A, int iterations, unsigned char *B, unsigned char starting_index, float clock_rate, unsigned char data_stride){
	
	unsigned char k = starting_index;/////make them in the same page, and miss near in cache lines
	for (int it = 0; it < iterations; it++){/////////////warmup
		k = A[k];
	}
	B[0] = k;///////////////it will disappear without this line.
	
	unsigned char j = starting_index;/////make them in the same page, and miss near in cache lines
	//int stride = 0;///////////
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	start_time = clock64();//////clock
			
	for (int it = 0; it < iterations; it++){/////////////warmup
		j = A[j];
	}
	
	end_time=clock64();//////clock
	long long int total_time = end_time - start_time;//////clock
	printf("inside%d:%fms\n", mark, (total_time / (float)clock_rate) / ((float)iterations));//////clock, average latency
	
	B[0] = j;
}

__global__ void tlb_latency_test(unsigned char *A, int iterations, unsigned char *B, float clock_rate, int mod, unsigned char data_stride){	
	
	P_chasing(mod, A, iterations, B, 0, clock_rate, data_stride);
	
	 __syncthreads();
}




//////////min page size 4kb = 4096b = 32 * 128.
__device__ void P_chasing2(int mark, int *A, int iterations, int *B, int starting_index, float clock_rate, int data_stride){
	
	int k = starting_index;/////make them in the same page, and miss near in cache lines
	for (int it = 0; it < mark; it++){/////////////warmup
		k = A[k];
	}
	B[0] = k;///////////////it will disappear without this line.
	
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

__global__ void tlb_latency_test2(int *A, int iterations, int *B, float clock_rate, int mod, int data_stride){	
	
	P_chasing2(mod, A, iterations, B, 0, clock_rate, data_stride);
	
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

	if(1){
	printf("################byte data############################\n");
	
	///////////////////////////////////////////////////////////////////GPU data out
	unsigned char *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(unsigned char) * 1));	
	
	printf("################fixing data range, changing stride############################\n");
	//for(int mod = 1024 * 256 * 8; mod > 0; mod = mod / 2){/////volta L2 6m
	//for(int mod = 1024 * 256 * 7 ; mod >= 1024 * 256 * 6; mod = mod - 256 * 128){/////volta L2 6m
	for(unsigned char data_stride = 4; data_stride <= 128; data_stride = data_stride * 2){
		printf("###################data_stride%d#########################\n", data_stride / 4);
	//for(int mod = 1024 * 256 * 2; mod > 0; mod = mod - 32 * 1024){/////kepler L2 1.5m
	for(int mod = 256; mod >= 256; mod = mod / 2){/////kepler L2 1.5m //////////////1024 * 4 * 3 /////////8 /////////// 1024 * 256 * 1.5 / 1024 * 4 * 3 / 8 = 4 sets? 
		///////////////////////////////////////////////////////////////////CPU data begin
		int data_size = 512 * 1024 * 30 * 4;/////size = iteration * stride = 30 2mb pages.		
		//int iterations = data_size / data_stride;
		int iterations = mod * 1024;
	
		unsigned char *CPU_data_in = (unsigned char*) malloc(sizeof(unsigned char) * data_size);	
		init_cpu_data(CPU_data_in, 255, data_stride, mod);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		unsigned char *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(unsigned char) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(unsigned char) * data_size, cudaMemcpyHostToDevice);
		
		tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate, mod / 4, data_stride);//////////////////////////////////////////////kernel is here	
		cudaDeviceSynchronize();
		
		checkCudaErrors(cudaFree(GPU_data_in));
		free(CPU_data_in);
	}
		printf("############################################\n\n");
	}
	
	printf("\n\n################fixing stride, changing data range############################\n\n");
	//for(int mod = 1024 * 256 * 8; mod > 0; mod = mod / 2){/////volta L2 6m
	//for(int mod = 1024 * 256 * 7 ; mod >= 1024 * 256 * 6; mod = mod - 256 * 128){/////volta L2 6m
	for(unsigned char data_stride = 4; data_stride <= 4; data_stride = data_stride * 2){
		printf("###################data_stride%d#########################\n", data_stride / 4);
	for(int mod = 256; mod >= 4; mod = mod / 2){/////kepler L2 1.5m
	//for(int mod = 1024 * 256 * 6; mod > 0; mod = mod / 2){/////kepler L2 1.5m //////////////1024 * 256 * 6 / 128 = 1024 * 2 * 6 ///////8 /////// 1024 * 256 * 1.5 / 1024 * 2 * 6 / 8 = 4 sets? 
		///////////////////////////////////////////////////////////////////CPU data begin
		int data_size = 1024 * 512 * 30 * 4;/////size = iteration * stride = 30 2mb pages.		
		//int iterations = data_size / data_stride;
		//int iterations = data_size;
		int iterations = mod * 1024;
	
		unsigned char *CPU_data_in = (unsigned char*) malloc(sizeof(unsigned char) * data_size);	
		init_cpu_data(CPU_data_in, 255, data_stride, mod);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		unsigned char *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(unsigned char) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(unsigned char) * data_size, cudaMemcpyHostToDevice);
		
		tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate, mod / 4, data_stride);//////////////////////////////////////////////kernel is here	
		cudaDeviceSynchronize();
		
		checkCudaErrors(cudaFree(GPU_data_in));
		free(CPU_data_in);
	}
		printf("############################################\n\n");
	}
			
	checkCudaErrors(cudaFree(GPU_data_out));	
	//free(CPU_data_out);
	}
	
	if(1){
	printf("################float data############################\n");
	
	int *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(int) * 1));	
	
	printf("################fixing data range, changing stride############################\n");
	//for(int mod = 1024 * 256 * 8; mod > 0; mod = mod / 2){/////volta L2 6m
	//for(int mod = 1024 * 256 * 7 ; mod >= 1024 * 256 * 6; mod = mod - 256 * 128){/////volta L2 6m
	for(int data_stride = 1; data_stride <= 1024; data_stride = data_stride * 2){
		printf("###################data_stride%d#########################\n", data_stride);
	//for(int mod = 1024 * 256 * 2; mod > 0; mod = mod - 32 * 1024){/////kepler L2 1.5m
	for(int mod = 1024 * 256 * 6; mod >= 1024 * 256 * 6; mod = mod / 2){/////kepler L2 1.5m //////////////1024 * 4 * 3 /////////8 /////////// 1024 * 256 * 1.5 / 1024 * 4 * 3 / 8 = 4 sets? 
		///////////////////////////////////////////////////////////////////CPU data begin
		int data_size = 512 * 1024 * 30;/////size = iteration * stride = 30 2mb pages.		
		//int iterations = data_size / data_stride;
		int iterations = mod * 2;
	
		int *CPU_data_in;
		CPU_data_in = (int*)malloc(sizeof(int) * data_size);	
		init_cpu_data2(CPU_data_in, data_size, data_stride, mod);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		int *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
		
		tlb_latency_test2<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);//////////////////////////////////////////////kernel is here	
		cudaDeviceSynchronize();
		
		checkCudaErrors(cudaFree(GPU_data_in));
		free(CPU_data_in);
	}
		printf("############################################\n\n");
	}	
	
	printf("\n\n################fixing stride, changing data range############################\n\n");
	//for(int mod = 1024 * 256 * 8; mod > 0; mod = mod / 2){/////volta L2 6m
	//for(int mod = 1024 * 256 * 7 ; mod >= 1024 * 256 * 6; mod = mod - 256 * 128){/////volta L2 6m
	for(int data_stride = 4; data_stride <= 4; data_stride = data_stride * 2){
		printf("###################data_stride%d#########################\n", data_stride);
	for(int mod = 1024 * 256 * 1.5 + 32 * 1024; mod > 1024 * 256 * 1.5 - 16 * 1024; mod = mod - 1024){/////kepler L2 1.5m
	//for(int mod = 1024 * 256 * 6; mod > 0; mod = mod / 2){/////kepler L2 1.5m //////////////1024 * 256 * 6 / 128 = 1024 * 2 * 6 ///////8 /////// 1024 * 256 * 1.5 / 1024 * 2 * 6 / 8 = 4 sets? 
		///////////////////////////////////////////////////////////////////CPU data begin
		int data_size = 1024 * 512 * 30;/////size = iteration * stride = 30 2mb pages.		
		//int iterations = data_size / data_stride;
		//int iterations = data_size;
		int iterations = mod * 2;
	
		int *CPU_data_in;
		CPU_data_in = (int*)malloc(sizeof(int) * data_size);	
		init_cpu_data2(CPU_data_in, data_size, data_stride, mod);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		int *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
		
		tlb_latency_test2<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);//////////////////////////////////////////////kernel is here	
		cudaDeviceSynchronize();
		
		checkCudaErrors(cudaFree(GPU_data_in));
		free(CPU_data_in);
	}
		printf("############################################\n\n");
	}
			
	checkCudaErrors(cudaFree(GPU_data_out));
	}
	
    exit(EXIT_SUCCESS);
}
