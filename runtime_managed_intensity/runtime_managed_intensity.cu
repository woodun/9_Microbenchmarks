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
//////////////large vs small data.

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

__global__ void gpu_initialization(long long int *A, long long int data_stride, long long int data_size){			

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
__global__ void Page_visitor(long long int *A, long long int data_stride, long long int clock_count){////load-compute -store
		
	/*
	long long int index = threadIdx.x;
	
	/////////////////////////////////time
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	long long int time_interval = 0;//////clock
	
	if(index = 0){
		start_time= clock64();
	}
	__syncthreads();
	*/
	
	int smid = 1;
	asm("mov.u32 %0, %smid;" : "=r"(smid) );
	int warpid = 1;
	asm("mov.u32 %0, %warpid;" : "=r"(warpid) );
	int ctaid = 1;
	asm("mov.u32 %0, %ctaid.x;" : "=r"(ctaid) );
	int nctaid = 1;
	asm("mov.u32 %0, %nctaid.x;" : "=r"(nctaid) );
	int ntid = 1;
	asm("mov.u32 %0, %ntid.x;" : "=r"(ntid) );
	int nsmid = 1;
	asm("mov.u32 %0, %nsmid;" : "=r"(nsmid) );
	int nwarpid = 1;
	asm("mov.u32 %0, %nwarpid;" : "=r"(nwarpid) );
	
	//printf("###1###warpid: %d, ctaid: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, nctaid.x: %d, ntid: %d, nsmid: %d, nwarpid: %d \n", warpid, ctaid, blockIdx.x, blockIdx.y, blockIdx.z, nctaid, ntid, nsmid, nwarpid);
	
	if(smid == 0){
		//if(blockIdx.x == 0){
			if(threadIdx.x % 32 == 0){/////%tid %ntid %laneid %warpid %nwarpid %ctaid %nctaid %smid %nsmid %gridid
				printf("###1###warpid: %d, blockIdx.x: %d\n", warpid, blockIdx.x );///printf does not reliably print everything
			}
		//}		
	}
	
	if(threadIdx.x % 32 == 0){
		printf("warpid: %d, blockIdx.x: %d, smid: %d, threadIdx.x: %d\n",  warpid, blockIdx.x, smid, threadIdx.x);
	}
	
	long long int index = (blockIdx.x * blockDim.x + threadIdx.x) * data_stride;
	
	long long int value = A[index];
	
	if(smid == 0){
		//if(blockIdx.x == 0){
			if(threadIdx.x % 32 == 0){/////%tid %ntid %laneid %warpid %nwarpid %ctaid %nctaid %smid %nsmid %gridid
				printf("###2###warpid: %d, blockIdx.x: %d\n", warpid, blockIdx.x );
			}
		//}		
	}
	
	/*
	//////////////////////////////////////////////sleep
	long long int start_clock = clock64();
    long long int clock_offset = 0;
    while (clock_offset < clock_count)
    {
        clock_offset = clock64() - start_clock;
    }
	*/
	
	//////////////////////////////////////////////loop
	long long int clock_offset = 0;
    while (clock_offset < clock_count){/////////////////what's the time overhead for addition and multiplication?
        clock_offset++;
		value = value + threadIdx.x;
    }
	
	if(smid == 0){
		//if(blockIdx.x == 0){
			if(threadIdx.x % 32 == 0){/////%tid %ntid %laneid %warpid %nwarpid %ctaid %nctaid %smid %nsmid %gridid
				printf("###3###warpid: %d, blockIdx.x: %d\n", warpid, blockIdx.x );
			}
		//}		
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
	
    //d_o[0] = clock_offset;
	//////////////////////////////////////////////sleep
	
	A[index] = value;
	
	if(smid == 0){
		//if(blockIdx.x == 0){
			if(threadIdx.x % 32 == 0){/////%tid %ntid %laneid %warpid %nwarpid %ctaid %nctaid %smid %nsmid %gridid
				printf("###4###warpid: %d, blockIdx.x: %d\n", warpid, blockIdx.x );
			}
		//}		
	}
	
	/*
	__syncthreads();
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
    //long long int dev_id = findCudaDevice(argc, (const char **) argv);
	long long int dev_id = 7;
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));
	checkCudaErrors(cudaSetDevice(dev_id));
	
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
	//does not cause eviction
	printf("###################\n#########################managed\n");
	///32 * 64 <==> 1 * 512 * 1024 (8gb), 32 * 512 <==> 1 * 64 * 1024 (8gb), 
	///is it still true that in multi threads the dynamic page threshold is still 64k? no, it seems to be 2k.
	//for(long long int data_stride = 1 * 1 * 1; data_stride <= 1 * 512 * 1024; data_stride = data_stride * 2){////not necessarily migrating whole 2m page. (not fair comparison but interesting to look at) 512 * 1024 is 4m, see what happens after 2m.
	for(long long int data_stride = 1 * 1 * 1024; data_stride <= 1 * 1 * 1024; data_stride = data_stride * 2){///test
	for(long long int mod = 536870912; mod <= 536870912; mod = mod * 2){////134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
	//for(long long int clock_count = 128; clock_count <= 8192; clock_count = clock_count * 2){/////11 positions.
	for(long long int clock_count = 1; clock_count <= 1; clock_count = clock_count * 2){/////test
		///////////////////////////////////////////////////////////////////CPU data begin		
		//long long int data_size = mod;
		long long int data_size = data_stride;
		data_size = data_size * 2048;
		data_size = data_size * 512;
		//long long int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		long long int *CPU_data_in;
		//CPU_data_in = (long long int*)malloc(sizeof(long long int) * data_size);
		checkCudaErrors(cudaMallocManaged(&CPU_data_in, sizeof(long long int) * data_size));/////////////using unified memory
		//init_cpu_data(CPU_data_in, data_size, data_stride);				
		///////////////////////////////////////////////////////////////////CPU data end	
			
		gpu_initialization<<<2048, 512>>>(CPU_data_in, data_stride, data_size);///////////////1024 per block max
		cudaDeviceSynchronize();
		
		/////////////////////////////////time
		struct timespec ts1;
		clock_gettime(CLOCK_REALTIME, &ts1);

		////may want to use more thread to see clock_count effect
		Page_visitor<<<2048, 512>>>(CPU_data_in, data_stride, clock_count);///////////////1024 per block max
		///////////////////////////////////////////////////32 * 64 * 1 * 512 * 1024 = 8gb.
		cudaDeviceSynchronize();
				
		/////////////////////////////////time
		struct timespec ts2;
		clock_gettime(CLOCK_REALTIME, &ts2);
		
		//printf("###################data_stride%lld#########################clock_count:%lld\n", data_stride, clock_count);
		//printf("*\n*\n*\nruntime:  %lluns\n", time_diff(ts1, ts2));
		printf("%llu ", time_diff(ts1, ts2));
		
		//checkCudaErrors(cudaFree(GPU_data_in));
		checkCudaErrors(cudaFree(CPU_data_in));
		//free(CPU_data_in);
		//checkCudaErrors(cudaFree(GPU_data_out));
	}
	printf("\n");
	}
	}
	
	/*
	//plain managed
	//causing eviction
	printf("###################\n#########################managed\n");
	///32 * 64 <==> 1 * 512 * 1024 (8gb), 32 * 512 <==> 1 * 64 * 1024 (8gb), 
	///is it still true that in multi threads the dynamic page threshold is still 64k? no, it seems to be 2k.
	for(long long int data_stride = 1 * 1 * 1; data_stride <= 1 * 512 * 1024; data_stride = data_stride * 2){////not necessarily migrating whole 2m. 512 * 1024 is 4m, see what happens after 2m.
	for(long long int mod = 536870912; mod <= 536870912; mod = mod * 2){////134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
	for(long long int clock_count = 128; clock_count <= 8192; clock_count = clock_count * 2){/////11 positions.
	//for(long long int clock_count = 1; clock_count <= 1; clock_count = clock_count * 2){/////11 positions.
		///////////////////////////////////////////////////////////////////CPU data begin		
		//long long int data_size = mod;
		long long int data_size = data_stride;
		data_size = data_size * 8192;
		data_size = data_size * 512;
		//long long int iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		long long int *CPU_data_in;
		//CPU_data_in = (long long int*)malloc(sizeof(long long int) * data_size);
		checkCudaErrors(cudaMallocManaged(&CPU_data_in, sizeof(long long int) * data_size));/////////////using unified memory
		init_cpu_data(CPU_data_in, data_size, data_stride);				
		///////////////////////////////////////////////////////////////////CPU data end	
			
		/////////////////////////////////time
		struct timespec ts1;
		clock_gettime(CLOCK_REALTIME, &ts1);

		////may want to use more thread to see clock_count effect
		Page_visitor<<<8192, 512>>>(CPU_data_in, data_stride, clock_count);///////////////1024 per block max
		///////////////////////////////////////////////////32 * 64 * 1 * 512 * 1024 = 8gb.
		cudaDeviceSynchronize();
				
		/////////////////////////////////time
		struct timespec ts2;
		clock_gettime(CLOCK_REALTIME, &ts2);
		
		//printf("###################data_stride%lld#########################clock_count:%lld\n", data_stride, clock_count);
		//printf("*\n*\n*\nruntime:  %lluns\n", time_diff(ts1, ts2));
		printf("%llu ", time_diff(ts1, ts2));
		
		//checkCudaErrors(cudaFree(GPU_data_in));
		checkCudaErrors(cudaFree(CPU_data_in));
		//free(CPU_data_in);
		//checkCudaErrors(cudaFree(GPU_data_out));
	}
	printf("\n");
	}
	}
	*/
	
    exit(EXIT_SUCCESS);
}