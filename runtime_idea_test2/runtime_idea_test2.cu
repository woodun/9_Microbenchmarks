#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>
#include <sys/time.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

/////////////////////////////L1 is enabled. "ALL_CCFLAGS += -Xptxas -dlcm=ca"
//////////////large vs small data.

void init_cpu_data(long long int* A, long long int size, double stride){
	
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

__global__ void gpu_initialization(long long int *A, double data_stride, long long int data_size){			

	long long int index = (blockIdx.x * blockDim.x + threadIdx.x);
	long long int thread_num =  gridDim.x * blockDim.x;
	
	for(long long int it = 0; it < data_size; it = it + thread_num){
		A[index + it]=23;
	}
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
	
	long long unsigned time_interval_ns = temp.tv_nsec;
	long long unsigned time_interval_s = temp.tv_sec;
	time_interval_s = time_interval_s * 1000000000;
	
	return time_interval_s + time_interval_ns;
}

//__global__ void Page_visitor(long long int *A, long long int *B, long long int data_stride, long long int clock_count){
__global__ void Page_visitor(long long int *A1, long long int *A2, long long int *B, double data_stride, long long int clock_count){////load-compute-store
			
	thread_block block = this_thread_block();
	
	long long int index = (blockIdx.x * 512 + (threadIdx.x - 32) ) * data_stride;
	long long int value1;
	long long int prefetch_index = (blockIdx.x * 512 + threadIdx.x * 16) * data_stride;
	long long int value2;
	
	if(threadIdx.x > 31){
		value1 = A1[index];
	}else{
		
		/*
		asm volatile(".reg.u64  t1;\n\t"
		".reg.u64  t2;\n\t"
		".reg.u64  t3;\n\t"
		"shl.b64  t2, %1, 3;\n\t"
		"cvta.to.global.u64  t1, %2;\n\t"
		"add.s64  t3, t2, t1;\n\t"		
		"ld.global.u64 	%0, [t3];"
		: "=l"(value2) : "l"(prefetch_index), "l"(A2));		
		*/
		
		value2 = A2[prefetch_index];
	}
	
	//block.sync();
	
	if(threadIdx.x > 31){
		//////////////////////////////////////////////loop
		long long int clock_offset = 0;
		while (clock_offset < clock_count){/////////////////what's the time overhead for addition and multiplication?
			clock_offset++;
			value1 = value1 + threadIdx.x;
		}
		
		value2 = A2[index];
	}else{
		
		/*
		asm volatile("cvta.to.global.u64  t1, %1;\n\t"
		"add.s64  t3, t2, t1;\n\t"		
		"st.global.u64 	[t3], %0;"
		: "=l"(value2) : "l"(B));
		*/
		
		B[prefetch_index] = value2;
	}	
	
	//block.sync();
	
	if(threadIdx.x > 31){
		//////////////////////////////////////////////loop
		long long int clock_offset2 = 0;
		while (clock_offset2 < clock_count){/////////////////what's the time overhead for addition and multiplication?
			clock_offset2++;
			value2 = value2 + threadIdx.x;
		}
	
		B[index] = value1 + value2;
	}
	
	/*
	if(threadIdx.x == 0){/////%tid %ntid %laneid %warpid %nwarpid %ctaid %nctaid %smid %nsmid %gridid
		int smid = 1;
		asm("mov.u32 %0, %smid;" : "=r"(smid) );
		printf("blockIdx.x: %d, smid: %d\n", blockIdx.x, smid);
		if(blockIdx.x == 55){
			int nsmid = 1;
			asm("mov.u32 %0, %smid;" : "=r"(nsmid) );
			printf("nsmid: %d\n", nsmid);
		}
	}
	*/
}

int main(int argc, char **argv)
{
	printf("\n");
	
    // set device
    cudaDeviceProp device_prop;
    //long long int dev_id = findCudaDevice(argc, (const char **) argv);
	long long int dev_id = 0;
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));
	
	int peak_clk = 1;//kHz
	checkCudaErrors(cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, dev_id));
	float clock_rate = (float) peak_clk;
	
	printf("clock_rate:%f\n", clock_rate);

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
	
	//plain managed
	//when was 64k and 4k pages used?
	//how to decrease the overhead of sync?
	//printf("###################\n#########################managed\n");
	/*
	for(long long int factor = 1; factor <= 128; factor = factor * 2){
	for(double data_stride = 1 * 1 * 0.25 * factor; data_stride <= 1 * 1 * 4 * factor; data_stride = data_stride * 2){///134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
	for(long long int clock_count = 512; clock_count <= 16384; clock_count = clock_count * 2){
	*/
	for(long long int factor = 1; factor <= 1; factor = factor * 2){
	for(double data_stride = 1 * 1 * 1 * factor; data_stride <= 1 * 1 * 1 * factor; data_stride = data_stride * 2){///134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
	for(long long int clock_count = 4096; clock_count <= 4096; clock_count = clock_count * 2){
		///////////////////////////////////////////////////////////////////CPU data begin		
		long long int data_size = data_stride * 512;
		data_size = data_size * 8192 * 512 / factor;
		
		long long int *CPU_data_in1;
		checkCudaErrors(cudaMallocManaged(&CPU_data_in1, sizeof(long long int) * data_size));/////////////using unified memory
		long long int *CPU_data_in2;
		checkCudaErrors(cudaMallocManaged(&CPU_data_in2, sizeof(long long int) * data_size));/////////////using unified memory		
		///////////////////////////////////////////////////////////////////CPU data end
		
		long long int *GPU_data_out;
		checkCudaErrors(cudaMallocManaged(&GPU_data_out, sizeof(long long int) * data_size));/////////////using unified memory
		///////////////////////////////////////////////////////////////////GPU data out	end
		
		if(1){
		gpu_initialization<<<8192 * 512 / factor, 512>>>(GPU_data_out, data_stride, data_size);///////////////1024 per block max
		cudaDeviceSynchronize();
		gpu_initialization<<<8192 * 512 / factor, 512>>>(CPU_data_in2, data_stride, data_size);///////////////1024 per block max
		cudaDeviceSynchronize();
		gpu_initialization<<<8192 * 512 / factor, 512>>>(CPU_data_in1, data_stride, data_size);///////////////1024 per block max
		cudaDeviceSynchronize();
		}else{
		init_cpu_data(GPU_data_out, data_size, data_stride);
		init_cpu_data(CPU_data_in2, data_size, data_stride);
		init_cpu_data(CPU_data_in1, data_size, data_stride);
		}
		
		/////////////////////////////////time
		struct timespec ts1;
		clock_gettime(CLOCK_REALTIME, &ts1);

		////may want to use more thread to see clock_count effect
		Page_visitor<<<8192 * 512 / factor, 512>>>(CPU_data_in1, CPU_data_in2, GPU_data_out, data_stride, clock_count);///1024 per block max
		///////////////////////////////////////////////////32 * 64 * 1 * 512 * 1024 = 8gb.
		cudaDeviceSynchronize();
				
		/////////////////////////////////time
		struct timespec ts2;
		clock_gettime(CLOCK_REALTIME, &ts2);
		
		//printf("###################data_stride%lld#########################clock_count:%lld\n", data_stride, clock_count);
		//printf("*\n*\n*\nruntime:  %lluns\n", time_diff(ts1, ts2));
		printf("%llu ", time_diff(ts1, ts2));
		
		checkCudaErrors(cudaFree(CPU_data_in1));
		checkCudaErrors(cudaFree(CPU_data_in2));
		checkCudaErrors(cudaFree(GPU_data_out));
	}
	printf("\n");
	}
	printf("####################%llu\n", factor);
	}

	exit(EXIT_SUCCESS);
}