#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>

void init_cpu_data(int* A, int size, int stride){
	for (int i = 0; i < size; ++i){
		A[i]=(i + stride) % size;
   	}
}

__global__ void tlb_latency_test(int *A, int iterations, int *B){
	
	int j = 0;
	
	for (int it =0; it < iterations; it ++){
		j = A[j];
	}	
	
	B[0] = j;
}

int main(int argc, char **argv)
{
    // set device
    cudaDeviceProp device_prop;
    int dev_id = findCudaDevice(argc, (const char **) argv);
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

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
	
	//printf("%d\n", sizeof(int));//////////size of int is 4.
	//exit(0);
	
	//////////////CPU data begin
	////////256 = 1kb, 262144 = 1mb, 524288 = 2mb.
	int iterations = 1000;
	int data_stride = 524288;/////2mb.
	//int data_size = 524288000;/////1000 * 2mb. ##### size = iteration * stride. ##### This can support 1000 iteration. The 1001st iteration starts from head again.
	int data_size = iterations * data_stride;/////size = iteration * stride.
	
	int *CPU_data_in;	
	CPU_data_in = (int*)malloc(sizeof(int) * data_size);
	//int *CPU_data_out;
	//CPU_data_out = (int*)malloc(data_size * sizeof(int));
	
	init_cpu_data(CPU_data_in, data_size, data_stride);
	//////////////CPU data end
	
	//////////////GPU data begin
	int *GPU_data_in;
	//////checkCudaErrors(cudaMallocManaged(&data, sizeof(int) * data_size));
	checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(int) * data_size));
	
	int *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(int) * 1));
	
	cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(int) * data_size, cudaMemcpyHostToDevice);
	//////////////GPU data end
		
		
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	

	//////////////kernel begin
	cudaEventRecord(start);	
	tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	//////////////kernel end
	
	
	//cudaMemcpy(CPU_data_out, GPU_data_out, sizeof(int) * data_size, cudaMemcpyDeviceToHost);
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

    cudaDeviceSynchronize();
	
	checkCudaErrors(cudaFree(GPU_data_in));
	checkCudaErrors(cudaFree(GPU_data_out));
	
	free(CPU_data_in);
	//free(CPU_data_out);

    printf("%f %f\n", milliseconds, milliseconds / 1000);
    exit(EXIT_SUCCESS);
}
