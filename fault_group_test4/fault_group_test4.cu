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
#include <cuda_profiler_api.h>

using namespace cooperative_groups;

/////////////////////////////L1 is enabled. "ALL_CCFLAGS += -Xptxas -dlcm=ca"
//////////////large vs small data.
//////creating 2 blocks doing exactly the same thing? method in the blog.
//////nvprof --profile-from-start off --print-gpu-trace --log-file 4warpsall.txt --csv ./fault_group_test15

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

//#define stride 512

__global__ void stream_thread(long long int *ptr, const long long int size, 
                              long long int *output, const long long int val) 
{ 
  long long int tid = threadIdx.x + blockIdx.x * blockDim.x; 
  long long int n = size / sizeof(long long int); 
  long long int accum = 0; 

  for(; tid < n; tid += blockDim.x * gridDim.x) 
    if (1) accum += ptr[tid]; 
      else ptr[tid] = val;  

  if (1) 
    output[threadIdx.x + blockIdx.x * blockDim.x] = accum; 
}


//#define STRIDE_64K 65536

__global__ void stream_warp(long long int *ptr, const long long int size, long long int *output, const long long int val, long long int STRIDE_64K) 
{ 
  int lane_id = threadIdx.x & 31; 
  long long int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5; 
  int warps_per_grid = (blockDim.x * gridDim.x) >> 5; 
  long long int warp_total = (size + STRIDE_64K-1) / STRIDE_64K; 

  long long int n = size / sizeof(long long int); 
  long long int accum = 0; 

  for(; warp_id < warp_total; warp_id += warps_per_grid) { 
    #pragma unroll
    for(int rep = 0; rep < STRIDE_64K/sizeof(long long int)/32; rep++) {
      long long int ind = warp_id * STRIDE_64K/sizeof(long long int) + rep * 32 + lane_id;
      if (ind < n) { 
        if (1) accum += ptr[ind]; 
        else ptr[ind] = val; 
      }
    } 
  } 

  if (1) 
    output[threadIdx.x + blockIdx.x * blockDim.x] = accum; 
}


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
		
	

	/*
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
	
	//for(double data_stride = 2684354560 * factor; data_stride <= 4294967296 * factor; data_stride = data_stride + 536870912){
	for(double data_stride = 536870912 * factor; data_stride <= 2147483648 * factor; data_stride = data_stride + 536870912){///134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
	//printf("\n");

	for(long long int clock_count = 32; clock_count <= 32; clock_count = clock_count * 2){
		
	///long long int time2 = time;
	//if(time2 > clock_count){
	//	time2 = clock_count;
	//}

		///////////////////////////////////////////////////////////////////CPU data begin
		//double temp = data_stride * 512;
		double temp = data_stride;
		long long int data_size = (long long int) temp;		
		//data_size = data_size * 8192 * 128 / factor;
		data_size = data_size / factor;
		
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
			
			gpu_initialization<<<8192 * scale / factor, 512>>>(GPU_data_out1, data_stride, data_size);///1024 per block max
			cudaDeviceSynchronize();
			if(0){
			gpu_initialization<<<8192 * scale / factor, 512>>>(CPU_data_in1, data_stride, data_size);///1024 per block max
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
		
		stream_thread<<<512, 512>>>(CPU_data_in1, 8 * data_size, GPU_data_out1, 7);

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
	*/
	
	//printf("%d\n",atoll(argv[1]));
	///*
	//////nvprof --profile-from-start off --print-gpu-trace --log-file prof512512size8gpage256.txt --csv ./fault_group_test4
	///also do for less than 16 warps same/diff cores
	////256	512	1024	2048	4096(5)	8192	16384	32768	65536(9)	131072	262144	524288	1048576	2097152(not index but real size)
	////1694768538 1363231797 1252858568 1227351872 1240169346 1272947382 1233840312 1203557761 1113896111 1339415345 1362196167 1236045906 (512 512)
	////3358277712 3273436872 2704454491 2736675685 2653977370 2627856647 3346580241 2936864316 3259807074 3837481549 3687183218 3129213694 (1 512)
	////2864158121 2808228031 2391946181 2611531792 2498506627 4187674088 4210352111 3878525896 3340698560 3407195743 2510971485 2192341097 (16 32)
	////4104063378 4184730646 3189396480 3005161192 2883182901 2699462209 2709909614 3610128080 3875286731 3876505935 3562113597 3241507431 (1 256)
	////
	//for(long long int STRIDE_64K = 256; STRIDE_64K <= 524288; STRIDE_64K = STRIDE_64K * 2){
	for(long long int STRIDE_64K = atoll(argv[1]); STRIDE_64K <= atoll(argv[1]); STRIDE_64K = STRIDE_64K * 2){
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
	
	//for(double data_stride = 2684354560 * factor; data_stride <= 4294967296 * factor; data_stride = data_stride + 536870912){
	for(double data_stride = 1073741824 * factor; data_stride <= 1073741824 * factor; data_stride = data_stride + 536870912){///134217728 = 1gb, 268435456 = 2gb, 536870912 = 4gb, 1073741824 = 8gb, 2147483648 = 16gb, 4294967296 = 32gb, 8589934592 = 64gb. (index)
	//printf("\n");

	for(long long int clock_count = 32; clock_count <= 32; clock_count = clock_count * 2){
		
	///long long int time2 = time;
	//if(time2 > clock_count){
	//	time2 = clock_count;
	//}

		///////////////////////////////////////////////////////////////////CPU data begin
		//double temp = data_stride * 512;
		double temp = data_stride;
		long long int data_size = (long long int) temp;		
		//data_size = data_size * 8192 * 128 / factor;
		data_size = data_size / factor;
		long long int data_size2 = 512 * 8192 ;	
		
		long long int *CPU_data_in1;
		checkCudaErrors(cudaMallocManaged(&CPU_data_in1, sizeof(long long int) * data_size));/////////////using unified memory
		///////////////////////////////////////////////////////////////////CPU data end
		
		long long int *GPU_data_out1;
		if(0){
			checkCudaErrors(cudaMallocManaged(&GPU_data_out1, sizeof(long long int) * data_size2));/////////////using unified memory
		}else{
			checkCudaErrors(cudaMalloc(&GPU_data_out1, sizeof(long long int) * data_size2));/////////////not using unified memory
		}
		///////////////////////////////////////////////////////////////////GPU data out	end
		
		if(1){
			double scale = 1;
			if(data_stride < 1){
				scale = data_stride;/////////make sure threadIdx is smaller than data_size in the initialization
			}
			
			//gpu_initialization<<<8192 * scale / factor, 512>>>(GPU_data_out1, data_stride, data_size);///1024 per block max
			//cudaDeviceSynchronize();
			if(0){
			gpu_initialization<<<8192 * scale / factor, 512>>>(CPU_data_in1, data_stride, data_size);///1024 per block max
			cudaDeviceSynchronize();
			}else{
			init_cpu_data(CPU_data_in1, data_size, data_stride);
			}
		}else{
			init_cpu_data(GPU_data_out1, data_size2, data_stride);
			init_cpu_data(CPU_data_in1, data_size, data_stride);		
		}
		
		cudaProfilerStart();////////////////////////////////start
		/////////////////////////////////time
		struct timespec ts1;
		clock_gettime(CLOCK_REALTIME, &ts1);
		
		stream_warp<<<16, 32>>>(CPU_data_in1, 8 * data_size, GPU_data_out1, 7, STRIDE_64K);

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
		cudaProfilerStop();/////////////////////////////////stop
	}
	}
	}
	}
	}
	}
	}
	}
	printf("\n");
	//*/
	
	exit(EXIT_SUCCESS);
}