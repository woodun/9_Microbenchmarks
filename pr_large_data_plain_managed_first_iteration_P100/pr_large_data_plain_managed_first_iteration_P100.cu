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
		A[size - stride]=0;
		
		long long int stride2 = 1 * 256 * 1024;////////2m
		for (long long int i = 8; i < size - stride2; i = i + stride2){
			A[i]=(i + stride2);
		}		
		A[size - stride2 + 8]=0;
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

long long int traverse_cpu_data(long long int *A, long long int iterations, long long int starting_index, long long int data_stride){	
	
	long long int j = starting_index;
			
	for (long long int it = 0; it < iterations; it++){
		j = A[j];
	}
	
	return j;
}

__device__ void P_chasing2(int mark, long long int *A, long long int iterations, long long int *B, long long int starting_index, float clock_rate, long long int data_stride){	
	
	__shared__ long long int s_index[1];
	
	long long int j = starting_index;/////make them in the same page, and miss near in cache lines
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	long long int time_interval = 0;//////clock	
		
	if(true){
		if(mark){
		asm(".reg .u64 t1;\n\t"
		".reg .u64 t2;\n\t"
		".reg .u32 t3;\n\t"
		".reg .u32 t4;\n\t"
		".reg .u64 t5;\n\t"
		".reg .u32 t6;\n\t");
		}
		
		asm("cvta.to.shared.u64 	t5, %0;\n\t"
		"cvt.u32.u64 	t6, t5;\n\t"
		:: "l"(s_index));////////////////////////////////////cvta.to.global.u64 	%rd4, %rd25; needed??
		
		for (long long int it = 0; it < iterations; it++){//////////it here is limited by the size of the shared memory
			
			asm("shl.b64 	t1, %3, 3;\n\t"	
			"add.s64 	t2, t1, %4;\n\t"
			"shl.b32 	t3, %6, 3;\n\t"
			"add.s32 	t4, t3, t6;\n\t"		
			"mov.u64 	%0, %clock64;\n\t"
			"ld.global.u64 	%2, [t2];\n\t"
			"st.shared.u64 	[t4], %2;\n\t"
			"mov.u64	%1, %clock64;"
			: "=l"(start_time), "=l"(end_time), "=l"(j) : "l"(j), "l"(A), "l"(s_index), "r"(0));		
					
			time_interval = end_time - start_time;
			printf("%lld %lld\n", j, time_interval);/////printf will affect L1 cache. Also, unknown effect to TLBs because it adds latency to L2 TLB misses.
			//////////////////////////////////////We are not using it for measurement. However, it can be used to recognize different conditions.
		}
	}

	B[0] = j;
}

__global__ void tlb_latency_test(long long int *A, long long int iterations, long long int *B, float clock_rate, long long int mod, long long int data_stride){
			
	/*
	/////////////using 32gb's iteration, address stride 1 * 128 * 1024, long long int data type
	P_chasing2(1, A, iterations/4, B, 2147483648, clock_rate, data_stride);//////////////migrate the first 8gb, starting 16 gb however.
	P_chasing2(0, A, iterations/8, B, 2147483648, clock_rate, data_stride);//////////////access the first 4gb again, starting 16 gb however.
	///////////before the next migration, take a rest, and see if the additional overhead decreases.
	long long int start_clock = clock64();
    long long int clock_offset = 0;
    while (clock_offset < 1000000000)
    {
        clock_offset = clock64() - start_clock;
    }
	P_chasing2(0, A, 3 * iterations/8, B, 0, clock_rate, data_stride);///////////migrate another 12gb, however starting at 0.
	P_chasing2(0, A, iterations/4, B, 2147483648, clock_rate, data_stride);//////////////which 4gb of the first 8gb is left?
	///////////conclusion: Still this does not change the pattern, the previous conclusion holds.
	*/
	/*
	/////////////using 32gb's iteration, address stride 1 * 128 * 1024, long long int data type
	P_chasing2(1, A, iterations/4, B, 0, clock_rate, data_stride);//////////////migrate the first 8gb
	P_chasing2(0, A, iterations/8, B, 0, clock_rate, data_stride);//////////////access the first 4gb again
	P_chasing2(0, A, 3 * iterations/8, B, 2147483648, clock_rate, data_stride);///////////migrate another 12gb, however starting at 16gb.
	P_chasing2(0, A, iterations/4, B, 0, clock_rate, data_stride);//////////////which 4gb of the first 8gb is left? what's the migration latency again?
	///////////conclusion: last access of first 8gb has low latency, 
	///////////and starting at 16gb (not continue at 8gb) the latency of first 16gb migration is still increasing as the same (with similar values).
	///////////It means that there is an additional warm up latency for the 2M group initialization.
	///////////And it is relating to the memory's physical locations itself, not relating to the address of the data.
	*/
	/*
	/////////////using 32gb's iteration, address stride 1 * 128 * 1024, long long int data type
	P_chasing2(1, A, iterations/4, B, 0, clock_rate, data_stride);//////////////migrate the first 8gb
	P_chasing2(0, A, iterations/8, B, 0, clock_rate, data_stride);//////////////access the first 4gb again
	P_chasing2(0, A, 3 * iterations/8, B, 1073741824, clock_rate, data_stride);///////////migrate another 12gb
	P_chasing2(0, A, iterations/4, B, 671088640, clock_rate, data_stride);//////////////which 4gb of the first 8gb is left? starting at 5gb.
	////////////////conclusion: the latter 4gb was left, even though the first 4gb is last accessed. The LRU is for migration not for access.
	*/
	//P_chasing2(1, A, iterations, B, 0, clock_rate, data_stride);
	//P_chasing2(0, A, iterations, B, mod - data_stride + 3, clock_rate, data_stride);
	
	__syncthreads();
}

__global__ void tlb_latency_test2(long long int *A, long long int iterations, long long int *B, float clock_rate, long long int mod, long long int data_stride){
			
	P_chasing2(1, A, iterations, B, 0, clock_rate, data_stride);//////////////migrate the first 8gb	
	//P_chasing2(1, A, iterations, B, 0, clock_rate, data_stride);
	//P_chasing2(0, A, iterations, B, mod - data_stride + 3, clock_rate, data_stride);
	
	__syncthreads();
}

__global__ void tlb_latency_test3(long long int *A, long long int iterations, long long int *B, float clock_rate, long long int mod, long long int data_stride){
			
	P_chasing2(1, A, iterations, B, 2281701376, clock_rate, data_stride);//////////////starting 17gb	
	//P_chasing2(1, A, iterations, B, 0, clock_rate, data_stride);
	//P_chasing2(0, A, iterations, B, mod - data_stride + 3, clock_rate, data_stride);
	
	__syncthreads();
}

__global__ void tlb_latency_test4(long long int *A, long long int iterations, long long int *B, float clock_rate, long long int mod, long long int data_stride){
			
	P_chasing2(1, A, iterations, B, 8, clock_rate, data_stride);//////////////starting 8, with a different stride.
	//P_chasing2(1, A, iterations, B, 0, clock_rate, data_stride);
	//P_chasing2(0, A, iterations, B, mod - data_stride + 3, clock_rate, data_stride);
	
	__syncthreads();
}

__global__ void tlb_latency_test5(long long int *A, long long int iterations, long long int *B, float clock_rate, long long int mod, long long int data_stride){
			
	P_chasing2(1, A, iterations, B, 2147483648, clock_rate, data_stride);//////////////	
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
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(long long int) * 2));			
	
	int counter = 0;	
	//for(long long int data_stride = 1 * 4 * 1024; data_stride <= 1 * 64 * 1024; data_stride = data_stride * 2){
	for(long long int data_stride = 1 * 128 * 1024; data_stride <= 1 * 128 * 1024; data_stride = data_stride * 2){

	//plain managed
	printf("*\n*\n*\n plain managed\n");	
	for(long long int mod = 4294967296; mod <= 4294967296; mod = mod * 2){////134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb.
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin
		long long int data_size = mod;
		long long int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
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

		//tlb_latency_test<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);///kernel is here	
		//cudaDeviceSynchronize();
		
		/*
		tlb_latency_test2<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);///migrate 32gb to gpu	(with warmup & no eviction & no trail) and (no warmup & with eviction & no trail)
		cudaDeviceSynchronize();
		
		traverse_cpu_data(CPU_data_in, iterations/2, 2147483648, data_stride);///////migrate last 16 gb to cpu, gpu is clear
		
		printf("location1:\n");
		
		tlb_latency_test2<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);///migrate 32gb to gpu again (no warmup & no eviction & no trail) and (no warmup & with eviction & no trail)
		cudaDeviceSynchronize();
		
		traverse_cpu_data(CPU_data_in, iterations/2, 2147483648, data_stride);///////migrate last 16 gb to cpu, gpu is clear
		
		printf("location2:\n");
		
		tlb_latency_test3<<<1, 1>>>(CPU_data_in, iterations/2, GPU_data_out, clock_rate, mod, data_stride);///migrate last 16gb (starting 17gb) to gpu again (no warmup & no eviction & with trail)
		cudaDeviceSynchronize();		
		///////////conclusion: eviction overhead exists, but page migration does not evict the page group setup (trail does exist, leave a trail when page size not dynamic).
		*/
		
		//page eviction evict the whole 2M group? 1m vs 2m strides.
		tlb_latency_test5<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);///migrate the last 16gb
		cudaDeviceSynchronize();
		
		tlb_latency_test4<<<1, 1>>>(CPU_data_in, iterations/2, GPU_data_out, clock_rate, mod, data_stride);///migrate first 16gb to gpu, offset and without covering all the previous last 16gb steps however.
		cudaDeviceSynchronize();
		
		tlb_latency_test3<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);///migrate the last 16gb again (starting 17gb), any page hit?
		cudaDeviceSynchronize();
		///////////////////conclusion: page eviction evict the whole 2M group.
		
		/*
		///////////is it migrating 64k always when not dynamic? use different stride to find out. 64 vs 128?
		tlb_latency_test5<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);///migrate the last 16gb
		cudaDeviceSynchronize();
		
		tlb_latency_test4<<<1, 1>>>(CPU_data_in, iterations/2, GPU_data_out, clock_rate, mod, data_stride);///migrate first 16gb to gpu, offset and without covering all the previous last 16gb steps however.
		cudaDeviceSynchronize();
		
		tlb_latency_test3<<<1, 1>>>(CPU_data_in, iterations, GPU_data_out, clock_rate, mod, data_stride);///migrate the last 16gb again (starting 17gb), any page hit?
		cudaDeviceSynchronize();
		///////////////////conclusion: page eviction evict the whole 2M group.
		*/
				
		//checkCudaErrors(cudaFree(GPU_data_in));
		checkCudaErrors(cudaFree(CPU_data_in));
		//free(CPU_data_in);		
	}
	}
			
	checkCudaErrors(cudaFree(GPU_data_out));	
	
    exit(EXIT_SUCCESS);
}