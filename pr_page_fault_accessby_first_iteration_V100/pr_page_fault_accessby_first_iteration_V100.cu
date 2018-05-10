#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>

///////////per request timing. L1 enabled. 
///////////cudaMemAdviseSetAccessedBy is not supported for K40 using dev_id (because cudaDevAttrConcurrentManagedAccess = 0).
///////////When using cudaCpuDeviceId, both first and second iteration are the same as the cudaMemAdviseSetPreferredLocation case.
///////////cudaMemAdviseSetPreferredLocation doesn't seem to have noticeable effect on K40.
///////////P.S. The 800s actually happens randomly. Thus it is not another condition.


//typedef unsigned char byte;

void init_cpu_data(int* A, long long int size, int stride, long long int mod){
	for (long long int i = 0; i < size; i = i + stride){
		A[i]=(i + stride) % mod;
   	}
	
	for (long long int i = 32; i < size; i = i + stride){
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
__device__ void P_chasing1(int mark, int *A, long long int iterations, int *B, int *C, long long int *D, int starting_index, float clock_rate, int data_stride){	
	
	int j = starting_index;/////make them in the same page, and miss near in cache lines
	
	//long long int start_time = 0;//////clock
	//long long int end_time = 0;//////clock
	//start_time = clock64();//////clock
			
	for (int it = 0; it < iterations; it++){
		j = A[j];
	}
	
	//end_time=clock64();//////clock
	//long long int total_time = end_time - start_time;//////clock
	//printf("inside%d:%fms\n", mark, (total_time / (float)clock_rate) / ((float)iterations));//////clock, average latency //////////the print will flush the L1?! (
	
	B[0] = j;
	//B[1] = (int) total_time;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void P_chasing2(int mark, int *A, long long int iterations, int *B, int *C, long long int *D, int starting_index, float clock_rate, int data_stride){//////what is the effect of warmup outside vs inside?
	
	//////shared memory: 0xc000 max (49152 Bytes = 48KB)
	__shared__ long long int s_tvalue[1024 * 4];/////must be enough to contain the number of iterations.
	__shared__ int s_index[1024 * 4];
	//__shared__ int s_index[1];
	
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
	
	asm(".reg .u32 t1;\n\t"
	".reg .u64 t2;\n\t"
	".reg .u32 t3;\n\t"
	".reg .u32 t4;\n\t"
	".reg .u64 t5;\n\t"
	".reg .u32 t6;\n\t"
	".reg .u64 t7;\n\t"
	"cvta.to.shared.u64 	t5, %0;\n\t"
	"cvt.u32.u64 	t6, t5;\n\t"
	:: "l"(s_index));////////////////////////////////////cvta.to.global.u64 	%rd4, %rd25; needed??
	
	for (int it = 0; it < iterations; it++){//////////it here is limited by the size of the shared memory
		
		asm("shl.b32 	t1, %3, 2;\n\t"
		"cvt.u64.u32 	t7, t1;\n\t"
		"add.s64 	t2, t7, %4;\n\t"
		"shl.b32 	t3, %6, 2;\n\t"
		"add.s32 	t4, t3, t6;\n\t"		
		"mov.u64 	%0, %clock64;\n\t"
		"ld.global.u32 	%2, [t2];\n\t"
		"st.shared.u32 	[t4], %2;\n\t"
		"mov.u64	%1, %clock64;"
		: "=l"(start_time), "=l"(end_time), "=r"(j) : "r"(j), "l"(A), "l"(s_index), "r"(it));		
				
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

__global__ void tlb_latency_test(int *A, long long int iterations, int *B, int *C, long long int *D, float clock_rate, long long int mod, int data_stride){
	
	long long int reduced_iter = iterations;
	if(reduced_iter > 4096){
		reduced_iter = 4096;
	}else if(reduced_iter < 16){
		reduced_iter = 16;
	}
	
	///////////kepler L2 has 48 * 1024 = 49152 cache lines. But we only have 1024 * 4 slots in shared memory.
	//P_chasing1(0, A, iterations + 0, B, C, D, 0, clock_rate, data_stride);////////saturate the L2
	P_chasing2(0, A, reduced_iter, B, C, D, 0, clock_rate, data_stride);////////partially print the data
	
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
	
	if (device_prop.concurrentManagedAccess == 1){
		printf("This device supports concurrent Managed Access.\n");
    }else{
		printf("This device does not support concurrent Managed Access.\n");
	}
	
	int value1 = 1;
	checkCudaErrors(cudaDeviceGetAttribute(&value1, cudaDevAttrConcurrentManagedAccess, dev_id));
	printf("cudaDevAttrConcurrentManagedAccess = %d\n", value1);	
	
	///////////////////////////////////////////////////////////////////GPU data out
	int *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(int) * 2));			
	
	FILE * pFile;
    pFile = fopen ("output.txt","w");		
	
	int counter = 0;
	/////////change the data stride as to observe if the latency increase is caused by iteration(cache) or stride(tlb)
	for(int data_stride = 1 * 1 * 1024; data_stride <= 2 * 256 * 1024; data_stride = data_stride * 2){/////////32mb stride
		//data_stride = data_stride + 32;///offset a cache line, trying to cause L2 miss but tlb hit.
		//printf("###################data_stride%d#########################\n", data_stride);
	//for(int mod = 1024 * 256 * 2; mod > 0; mod = mod - 32 * 1024){/////kepler L2 1.5m = 12288 cache lines, L1 16k = 128 cache lines.
	for(long long int mod2 = 1 * 16 * 1024; mod2 <= 2147483648; mod2 = mod2 * 2){////268435456 = 1gb, 536870912 = 2gb, 1073741824 = 4gb, 2147483648 = 8gb, 4294967296 = 16gb.
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin
		//int data_size = 2 * 256 * 1024 * 32;/////size = iteration * stride = 32 2mb pages.
		long long int mod = mod2;
		if(mod > 2684354560){
			mod = 2684354560;
		}
		long long int data_size = mod;
		if(data_size < 4194304){//////////data size at least 16mb to prevent L2 prefetch
			data_size = 4194304;
		}	
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		long long int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		int *CPU_data_in;
		//CPU_data_in = (int*)malloc(sizeof(int) * data_size);
		checkCudaErrors(cudaMallocManaged(&CPU_data_in, sizeof(int) * data_size));/////////////using unified memory
		checkCudaErrors(cudaMemAdvise(CPU_data_in, sizeof(int) * data_size, cudaMemAdviseSetAccessedBy, dev_id));//////////using hint
		init_cpu_data(CPU_data_in, data_size, data_stride, mod);
		
		long long int reduced_iter = iterations;
		if(reduced_iter > 4096){
			reduced_iter = 4096;
		}else if(reduced_iter < 16){
			reduced_iter = 16;
		}
		
		int *CPU_data_out_index;
		CPU_data_out_index = (int*)malloc(sizeof(int) * reduced_iter);
		long long int *CPU_data_out_time;
		CPU_data_out_time = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		//int *GPU_data_in;
		//checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));	
		//cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		int *GPU_data_out_index;
		checkCudaErrors(cudaMalloc(&GPU_data_out_index, sizeof(int) * reduced_iter));
		long long int *GPU_data_out_time;
		checkCudaErrors(cudaMalloc(&GPU_data_out_time, sizeof(long long int) * reduced_iter));
		
		tlb_latency_test<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, GPU_data_out_index, GPU_data_out_time, clock_rate, mod, data_stride);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		cudaMemcpy(CPU_data_out_index, GPU_data_out_index, sizeof(int) * reduced_iter, cudaMemcpyDeviceToHost);
		cudaMemcpy(CPU_data_out_time, GPU_data_out_time, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
				

		fprintf(pFile, "###################data_stride%d#########################\n", data_stride);
		fprintf (pFile, "###############Mod%lld##############%lld\n", mod, iterations);
		for (long long int it = 0; it < reduced_iter; it++){		
			fprintf (pFile, "%d %fms %lldcycles\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate, CPU_data_out_time[it]);
			//fprintf (pFile, "%d %fms\n", it, CPU_data_out_time[it] / (float)clock_rate);
			//printf ("%d %fms\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate);
		}
		
		checkCudaErrors(cudaFree(GPU_data_out_index));
		checkCudaErrors(cudaFree(GPU_data_out_time));
		//checkCudaErrors(cudaFree(GPU_data_in));
		checkCudaErrors(cudaFree(CPU_data_in));
		//free(CPU_data_in);
		free(CPU_data_out_index);
		free(CPU_data_out_time);
	}
		//printf("############################################\n\n");
	}
			
	checkCudaErrors(cudaFree(GPU_data_out));	
	//free(CPU_data_out);
	fclose (pFile);
	
    exit(EXIT_SUCCESS);
}