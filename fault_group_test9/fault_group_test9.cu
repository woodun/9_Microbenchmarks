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
//////test the intra-warp coalescing. also create larger intervals between block, E.g., sm0 and sm 32? & remote address

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

#define stride 16

///////////////262144 (2m), 4194304 (32m), 8388608 (64m), 
__global__ void page_visitor(long long int *A1, long long int *B1, double data_stride, long long int clock_count){////long
			
	long long int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
	double temp = (warp_id * 32 + (threadIdx.x % 32) ) * stride;
	if(warp_id == 27){
		temp = (512 * 32 + (threadIdx.x % 32) ) * stride;
	}
	
	//double temp = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
	//double temp = ((blockIdx.x * blockDim.x + threadIdx.x) % 32) * 2 + blockIdx.x * 1;
	long long int index = __double2ll_rd(temp);
	long long int value1;

	if(warp_id == 0 || warp_id == 27){
		if(threadIdx.x % 32 <= clock_count){
			value1 = A1[index];
		
			B1[index] = value1;	
		}
	}
}


///////////long 0 - 31 same core
///////////long 0 - 64 same core
///////////long 0 - 64 different core
///////////mixed 0 - 64 same core
///////////mixed 0 - 64 different core

int main(int argc, char **argv)
{
	printf("\n");
	
    // set device
    cudaDeviceProp device_prop;
    //long long int dev_id = findCudaDevice(argc, (const char **) argv);
	long long int dev_id = 0;
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));
	
	//int peak_clk = 1;//kHz
	//checkCudaErrors(cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, dev_id));
	//float clock_rate = (float) peak_clk;
	//printf("clock_rate:%f\n", clock_rate);

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
	
	/*
	if (device_prop.concurrentManagedAccess == 1){
		printf("This device supports concurrent Managed Access.\n");
    }else{
		printf("This device does not support concurrent Managed Access.\n");
	}
	*/
	
	int value1 = 1;
	checkCudaErrors(cudaDeviceGetAttribute(&value1, cudaDevAttrConcurrentManagedAccess, dev_id));
	//printf("cudaDevAttrConcurrentManagedAccess = %d\n", value1);	
	
	
	///*
	//printf("############approach\n");
	for(long long int time = 0; time <= 0; time = time + 1){
	//printf("\n####################time: %llu\n", time);
	
	//long long int coverage2 = 0;
	for(long long int coverage = 1; coverage <= 1; coverage = coverage * 2){///////////////8192 is 2m.
		//coverage2++;
		//if(coverage2 == 2){
		//	coverage = 1;
		//}
		//printf("############coverage: %llu\n", coverage);
		
	for(long long int rate = 1; rate <= 1; rate = rate * 2){
		//printf("############rate: %llu\n", rate);
		
	//long long int offset2 = 0;
	//for(long long int offset = 0; offset <= 0; offset = offset * 2){///////8
	for(long long int offset = 0; offset <= 0; offset = offset + 8){
		//offset2++;
		//if(offset2 == 2){
		//	offset = 1;
		//}
	//printf("############offset: %llu\n", offset);
	
	for(long long int factor = 1; factor <= 1; factor = factor * 2){/////////////16384 (128k) max
	//printf("####################factor: %llu\n", factor);
	
	for(double data_stride = 1 * 1 * 1 * factor; data_stride <= 1 * 1 * 1 * factor; data_stride = data_stride * 2){///134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
	//printf("\n");

	for(long long int clock_count = 0; clock_count <= 31; clock_count = clock_count + 1){
		
	///long long int time2 = time;
	//if(time2 > clock_count){
	//	time2 = clock_count;
	//}

		///////////////////////////////////////////////////////////////////CPU data begin
		double temp = data_stride * 512;
		long long int data_size = (long long int) temp;
		//data_size = data_size * 8192 * 512 / factor;
		data_size = data_size * 8192 * 128 / factor;
		
		long long int *CPU_data_in1;
		checkCudaErrors(cudaMallocManaged(&CPU_data_in1, sizeof(long long int) * data_size));/////////////using unified memory
		///////////////////////////////////////////////////////////////////CPU data end
		
		long long int *GPU_data_out1;
		checkCudaErrors(cudaMallocManaged(&GPU_data_out1, sizeof(long long int) * data_size));/////////////using unified memory
		///////////////////////////////////////////////////////////////////GPU data out	end
		
		if(1){
			double scale = 1;
			if(data_stride < 1){
				scale = data_stride;/////////make sure threadIdx is smaller than data_size in the initialization
			}
			
			gpu_initialization<<<8192 * 128 * scale / factor, 512>>>(GPU_data_out1, data_stride, data_size);///1024 per block max
			cudaDeviceSynchronize();
			if(0){
			gpu_initialization<<<8192 * 128 * scale / factor, 512>>>(CPU_data_in1, data_stride, data_size);///1024 per block max
			cudaDeviceSynchronize();
			}else{
			init_cpu_data(CPU_data_in1, data_size, data_stride);
			}
		}else{
			init_cpu_data(GPU_data_out1, data_size, data_stride);
			init_cpu_data(CPU_data_in1, data_size, data_stride);		
		}
		
		/////////////////////////////////time
		struct timespec ts1;
		clock_gettime(CLOCK_REALTIME, &ts1);

		int block_num = 1;

		page_visitor<<<block_num, 32>>>(CPU_data_in1, GPU_data_out1, data_stride, clock_count);/////long 
	
		cudaDeviceSynchronize();
				
		/////////////////////////////////time
		struct timespec ts2;
		clock_gettime(CLOCK_REALTIME, &ts2);
		
		//printf("###################data_stride%lld#########################clock_count:%lld\n", data_stride, clock_count);
		//printf("*\n*\n*\nruntime:  %lluns\n", time_diff(ts1, ts2));
		printf("%llu ", time_diff(ts1, ts2));
		fflush(stdout);
		
		checkCudaErrors(cudaFree(CPU_data_in1));		
		checkCudaErrors(cudaFree(GPU_data_out1));
	}
	}
	}
	}
	}
	}
	}
	printf("\n");	
	
	
	for(long long int time = 0; time <= 0; time = time + 1){
	//printf("\n####################time: %llu\n", time);
	
	//long long int coverage2 = 0;
	for(long long int coverage = 1; coverage <= 1; coverage = coverage * 2){///////////////8192 is 2m.
		//coverage2++;
		//if(coverage2 == 2){
		//	coverage = 1;
		//}
		//printf("############coverage: %llu\n", coverage);
		
	for(long long int rate = 1; rate <= 1; rate = rate * 2){
		//printf("############rate: %llu\n", rate);
		
	//long long int offset2 = 0;
	//for(long long int offset = 0; offset <= 0; offset = offset * 2){///////8
	for(long long int offset = 0; offset <= 0; offset = offset + 8){
		//offset2++;
		//if(offset2 == 2){
		//	offset = 1;
		//}
	//printf("############offset: %llu\n", offset);
	
	for(long long int factor = 1; factor <= 1; factor = factor * 2){/////////////16384 (128k) max
	//printf("####################factor: %llu\n", factor);
	
	for(double data_stride = 1 * 1 * 1 * factor; data_stride <= 1 * 1 * 1 * factor; data_stride = data_stride * 2){///134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
	//printf("\n");

	for(long long int clock_count = 0; clock_count <= 31; clock_count = clock_count + 1){
		
	///long long int time2 = time;
	//if(time2 > clock_count){
	//	time2 = clock_count;
	//}

		///////////////////////////////////////////////////////////////////CPU data begin
		double temp = data_stride * 512;
		long long int data_size = (long long int) temp;
		//data_size = data_size * 8192 * 512 / factor;
		data_size = data_size * 8192 * 128 / factor;
		
		long long int *CPU_data_in1;
		checkCudaErrors(cudaMallocManaged(&CPU_data_in1, sizeof(long long int) * data_size));/////////////using unified memory
		///////////////////////////////////////////////////////////////////CPU data end
		
		long long int *GPU_data_out1;
		checkCudaErrors(cudaMallocManaged(&GPU_data_out1, sizeof(long long int) * data_size));/////////////using unified memory
		///////////////////////////////////////////////////////////////////GPU data out	end
		
		if(1){
			double scale = 1;
			if(data_stride < 1){
				scale = data_stride;/////////make sure threadIdx is smaller than data_size in the initialization
			}
			
			gpu_initialization<<<8192 * 128 * scale / factor, 512>>>(GPU_data_out1, data_stride, data_size);///1024 per block max
			cudaDeviceSynchronize();
			if(0){
			gpu_initialization<<<8192 * 128 * scale / factor, 512>>>(CPU_data_in1, data_stride, data_size);///1024 per block max
			cudaDeviceSynchronize();
			}else{
			init_cpu_data(CPU_data_in1, data_size, data_stride);
			}
		}else{
			init_cpu_data(GPU_data_out1, data_size, data_stride);
			init_cpu_data(CPU_data_in1, data_size, data_stride);		
		}
		
		/////////////////////////////////time
		struct timespec ts1;
		clock_gettime(CLOCK_REALTIME, &ts1);

		int block_num = 1;

		page_visitor<<<block_num, 1024>>>(CPU_data_in1, GPU_data_out1, data_stride, clock_count);/////long 
	
		cudaDeviceSynchronize();
				
		/////////////////////////////////time
		struct timespec ts2;
		clock_gettime(CLOCK_REALTIME, &ts2);
		
		//printf("###################data_stride%lld#########################clock_count:%lld\n", data_stride, clock_count);
		//printf("*\n*\n*\nruntime:  %lluns\n", time_diff(ts1, ts2));
		printf("%llu ", time_diff(ts1, ts2));
		fflush(stdout);
		
		checkCudaErrors(cudaFree(CPU_data_in1));		
		checkCudaErrors(cudaFree(GPU_data_out1));
	}
	}
	}
	}
	}
	}
	}
	printf("\n");
	
	
	for(long long int time = 0; time <= 0; time = time + 1){
	//printf("\n####################time: %llu\n", time);
	
	//long long int coverage2 = 0;
	for(long long int coverage = 1; coverage <= 1; coverage = coverage * 2){///////////////8192 is 2m.
		//coverage2++;
		//if(coverage2 == 2){
		//	coverage = 1;
		//}
		//printf("############coverage: %llu\n", coverage);
		
	for(long long int rate = 1; rate <= 1; rate = rate * 2){
		//printf("############rate: %llu\n", rate);
		
	//long long int offset2 = 0;
	//for(long long int offset = 0; offset <= 0; offset = offset * 2){///////8
	for(long long int offset = 0; offset <= 0; offset = offset + 8){
		//offset2++;
		//if(offset2 == 2){
		//	offset = 1;
		//}
	//printf("############offset: %llu\n", offset);
	
	for(long long int factor = 1; factor <= 1; factor = factor * 2){/////////////16384 (128k) max
	//printf("####################factor: %llu\n", factor);
	
	for(double data_stride = 1 * 1 * 1 * factor; data_stride <= 1 * 1 * 1 * factor; data_stride = data_stride * 2){///134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
	//printf("\n");

	for(long long int clock_count = 0; clock_count <= 31; clock_count = clock_count + 1){
		
	///long long int time2 = time;
	//if(time2 > clock_count){
	//	time2 = clock_count;
	//}

		///////////////////////////////////////////////////////////////////CPU data begin
		double temp = data_stride * 512;
		long long int data_size = (long long int) temp;
		//data_size = data_size * 8192 * 512 / factor;
		data_size = data_size * 8192 * 128 / factor;
		
		long long int *CPU_data_in1;
		checkCudaErrors(cudaMallocManaged(&CPU_data_in1, sizeof(long long int) * data_size));/////////////using unified memory
		///////////////////////////////////////////////////////////////////CPU data end
		
		long long int *GPU_data_out1;
		checkCudaErrors(cudaMallocManaged(&GPU_data_out1, sizeof(long long int) * data_size));/////////////using unified memory
		///////////////////////////////////////////////////////////////////GPU data out	end
		
		if(1){
			double scale = 1;
			if(data_stride < 1){
				scale = data_stride;/////////make sure threadIdx is smaller than data_size in the initialization
			}
			
			gpu_initialization<<<8192 * 128 * scale / factor, 512>>>(GPU_data_out1, data_stride, data_size);///1024 per block max
			cudaDeviceSynchronize();
			if(0){
			gpu_initialization<<<8192 * 128 * scale / factor, 512>>>(CPU_data_in1, data_stride, data_size);///1024 per block max
			cudaDeviceSynchronize();
			}else{
			init_cpu_data(CPU_data_in1, data_size, data_stride);
			}
		}else{
			init_cpu_data(GPU_data_out1, data_size, data_stride);
			init_cpu_data(CPU_data_in1, data_size, data_stride);		
		}
		
		/////////////////////////////////time
		struct timespec ts1;
		clock_gettime(CLOCK_REALTIME, &ts1);

		int block_num = 32;

		page_visitor<<<block_num, 32>>>(CPU_data_in1, GPU_data_out1, data_stride, clock_count);/////long 
	
		cudaDeviceSynchronize();
				
		/////////////////////////////////time
		struct timespec ts2;
		clock_gettime(CLOCK_REALTIME, &ts2);
		
		//printf("###################data_stride%lld#########################clock_count:%lld\n", data_stride, clock_count);
		//printf("*\n*\n*\nruntime:  %lluns\n", time_diff(ts1, ts2));
		printf("%llu ", time_diff(ts1, ts2));
		fflush(stdout);
		
		checkCudaErrors(cudaFree(CPU_data_in1));		
		checkCudaErrors(cudaFree(GPU_data_out1));
	}
	}
	}
	}
	}
	}
	}
	printf("\n");
	
	
	exit(EXIT_SUCCESS);
}