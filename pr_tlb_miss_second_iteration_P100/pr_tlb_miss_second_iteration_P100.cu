#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>
// utilities
#include <helper_cuda.h>
#include <time.h>


///////////per request timing. L1 enabled. P100.
///////////using more than 8gb.


//typedef unsigned char byte;

void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

void init_cpu_data(unsigned *A, unsigned size, unsigned stride, unsigned mod, unsigned iterations){
	for (unsigned i = 0; i < size - stride; i = i + stride){
		A[i]=(i + stride);
   	}
	
	for (unsigned i = 7; i < size - stride; i = i + stride){
		A[i]=(i + stride);
   	}
	
	int rand_sequence[iterations];
	
	//////random sequence offset 0
	for(int i = 0; i < iterations; i++){
		rand_sequence[i] = i;
	}
	//srand (time(NULL));
	srand (0);
	shuffle(rand_sequence, iterations);
	
	unsigned previous_rand_num;
	unsigned rand_num = rand_sequence[0] * stride;	
	for(unsigned i = 1; i < iterations; i++){		
		previous_rand_num = rand_num;		
		rand_num = rand_sequence[i] * stride;		
		A[previous_rand_num]=rand_num;
	}
	
	//////random sequence offset 7	
	for(int i = 0; i < iterations; i++){
		rand_sequence[i] = i;
	}
	//srand (time(NULL));
	//shuffle(rand_sequence, iterations);
	
	rand_num = rand_sequence[0] * stride + 7;	
	for(unsigned i = 1; i < iterations; i++){		
		previous_rand_num = rand_num;		
		rand_num = rand_sequence[i] * stride + 7;		
		A[previous_rand_num]=rand_num;
	}
  
	/*
	///////manually set the nodes
	A[32]=104333344;
	A[104333344]=200802336;
	A[200802336]=353370144;
	A[353370144]=372244512;
	A[372244512]=110100512;
	A[110100512]=182452256;
	A[182452256]=333971488;
	A[333971488]=225443872;
	A[225443872]=155189280;
	A[155189280]=104333344;
	*/
	
	for (unsigned i = size - stride; i < size; i++){
		A[i]=0;
   	}
}

__device__ void P_chasing0(int mark, unsigned *A, int iterations, int *B, int *C, unsigned *D, int starting_index, float clock_rate, int data_stride){	
	
	int j = starting_index;/////make them in the same page, and miss near in cache lines
			
	for (int it = 0; it < iterations; it++){	
		j = A[j];		
	}	
		
	B[0] = j;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void P_chasing1(int mark, unsigned *A, unsigned iterations, unsigned *B, unsigned *C, long long int *D, unsigned starting_index, float clock_rate, unsigned data_stride){
	
	unsigned j = starting_index;/////make them in the same page, and miss near in cache lines
	
	//unsigned start_time = 0;//////clock
	//unsigned end_time = 0;//////clock
	//start_time = clock64();//////clock
			
	for (unsigned it = 0; it < iterations; it++){
		j = A[j];
	}
	
	//end_time=clock64();//////clock
	//unsigned total_time = end_time - start_time;//////clock
	//printf("inside%d:%fms\n", mark, (total_time / (float)clock_rate) / ((float)iterations));//////clock, average latency //////////the print will flush the L1?! (
	
	B[0] = j;
	//B[1] = (int) total_time;
}

//////////min page size 4kb = 4096b = 32 * 128.
__device__ void P_chasing2(int mark, unsigned *A, unsigned iterations, unsigned *B, unsigned *C, long long int *D, unsigned starting_index, float clock_rate, unsigned data_stride){//////what is the effect of warmup outside vs inside?
	
	//////shared memory: 0xc000 max (49152 Bytes = 48KB)
	__shared__ long long int s_tvalue[1024 * 4];/////must be enough to contain the number of iterations.
	__shared__ unsigned s_index[1024 * 4];
	//__shared__ unsigned s_index[1];
	
	unsigned j = starting_index;/////make them in the same page, and miss near in cache lines
	//int j = B[0];
	
	long long int start_time = 0;//////clock
	long long int end_time = 0;//////clock
	long long int time_interval = 0;//////clock
	//unsigned total_time = end_time - start_time;//////clock
	
	/*		
	for (int it = 0; it < iterations; it++){
		
		start_time = clock64();//////clock		
		j = A[j];
		//s_index[it] = j;
		end_time=clock64();//////clock		
		s_tvalue[it] = end_time - start_time;
	}
	*/
	
		asm(".reg .u64 t1;\n\t"
		".reg .u64 t2;\n\t");
	
	for (unsigned it = 0; it < iterations; it++){
		
		/*
		asm("mul.wide.u32 	t1, %3, %5;\n\t"	
		"add.u64 	t2, t1, %4;\n\t"		
		"mov.u64 	%0, %clock64;\n\t"		
		"ld.global.u32 	%2, [t2];\n\t"
		"mov.u64 	%1, %clock64;"
		: "=l"(start_time), "=l"(end_time), "=r"(j) : "r"(j), "l"(A), "r"(4));
		*/

		asm("mul.wide.u32 	t1, %2, %4;\n\t"	
		"add.u64 	t2, t1, %3;\n\t"		
		"mov.u64 	%0, %clock64;\n\t"		
		"ld.global.u32 	%1, [t2];\n\t"		
		: "=l"(start_time), "=r"(j) : "r"(j), "l"(A), "r"(4));
		
		s_index[it] = j;////what if without this? ///Then it is not accurate and cannot get the access time at all, due to the ILP. (another way is to use average time, but inevitably containing other instructions:setp, add).
		
		asm volatile ("mov.u64 %0, %clock64;": "=l"(end_time));
		
		time_interval = end_time - start_time;
		//if(it >= 4 * 1024){
		s_tvalue[it] = time_interval;
		//}
	}
	
	//printf("inside%d:%fms\n", mark, (total_time / (float)clock_rate) / ((float)iterations));//////clock, average latency
	
	B[0] = j;
	
	for (unsigned it = 0; it < iterations; it++){		
		C[it] = s_index[it];
		D[it] = s_tvalue[it];
	}
}

__global__ void tlb_latency_test(unsigned *A, unsigned iterations, unsigned *B, unsigned *C, long long int *D, float clock_rate, unsigned mod, int data_stride){
	
	unsigned reduced_iter = iterations;
	if(reduced_iter > 512){
		reduced_iter = 512;
	}else if(reduced_iter < 16){
		reduced_iter = 16;
	}
	
	///////////kepler L2 has 48 * 1024 = 49152 cache lines. But we only have 1024 * 4 slots in shared memory.
	P_chasing1(0, A, iterations + 0, B, C, D, 0, clock_rate, data_stride);////////saturate the L2
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
	
	///////////////////////////////////////////////////////////////////GPU data out
	unsigned *GPU_data_out;
	checkCudaErrors(cudaMalloc(&GPU_data_out, sizeof(unsigned) * 2));			
	
	FILE * pFile;
    pFile = fopen ("output.txt","w");		
	
	unsigned counter = 0;
	for(unsigned data_stride = 2 * 256 * 1024; data_stride <= 2 * 256 * 1024; data_stride = data_stride * 2){/////////32mb stride
		//data_stride = data_stride + 32;///offset a cache line, trying to cause L2 miss but tlb hit.
		//printf("###################data_stride%d#########################\n", data_stride);
	//for(int mod = 1024 * 256 * 2; mod > 0; mod = mod - 32 * 1024){/////kepler L2 1.5m = 12288 cache lines, L1 16k = 128 cache lines.
	for(unsigned mod2 = 2 * 256 * 1024; mod2 <= 1073741824; mod2 = mod2 * 2){////268435456 = 1gb, 536870912 = 2gb, 1073741824 = 4gb, 2147483648 = 8gb, 4294967296 = 16gb.
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin
		//int data_size = 2 * 256 * 1024 * 32;/////size = iteration * stride = 32 2mb pages.
		unsigned mod = mod2;
		if(mod > 3221225472){
			mod = 3221225472;
		}
		unsigned data_size = mod;
		if(data_size < 4194304){//////////data size at least 16mb to prevent L2 prefetch
			data_size = 4194304;
		}
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		unsigned iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		unsigned *CPU_data_in;
		CPU_data_in = (unsigned*)malloc(sizeof(unsigned) * data_size);
		init_cpu_data(CPU_data_in, data_size, data_stride, mod, iterations);
		
		
		unsigned reduced_iter = iterations;
		if(reduced_iter > 512){
			reduced_iter = 512;
		}else if(reduced_iter < 16){
			reduced_iter = 16;
		}
		
		unsigned *CPU_data_out_index;
		CPU_data_out_index = (unsigned*)malloc(sizeof(unsigned) * reduced_iter);
		long long int *CPU_data_out_time;
		CPU_data_out_time = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		unsigned *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(unsigned) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(unsigned) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		unsigned *GPU_data_out_index;
		checkCudaErrors(cudaMalloc(&GPU_data_out_index, sizeof(unsigned) * reduced_iter));
		long long int *GPU_data_out_time;
		checkCudaErrors(cudaMalloc(&GPU_data_out_time, sizeof(long long int) * reduced_iter));
		
		tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, GPU_data_out_index, GPU_data_out_time, clock_rate, mod, data_stride);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		cudaMemcpy(CPU_data_out_index, GPU_data_out_index, sizeof(unsigned) * reduced_iter, cudaMemcpyDeviceToHost);
		cudaMemcpy(CPU_data_out_time, GPU_data_out_time, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
				

		fprintf(pFile, "###################data_stride%u#########################\n", data_stride);
		fprintf (pFile, "###############Mod%u##############%u\n", mod, iterations);
		for (unsigned it = 0; it < reduced_iter; it++){			
			fprintf (pFile, "%u %fms %lldcycles\n", CPU_data_out_index[it], (double)CPU_data_out_time[it] / (float)clock_rate, CPU_data_out_time[it]);
			//fprintf (pFile, "%d %fms\n", it, CPU_data_out_time[it] / (float)clock_rate);
			//printf ("%d %fms\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate);
		}
		
		checkCudaErrors(cudaFree(GPU_data_out_index));
		checkCudaErrors(cudaFree(GPU_data_out_time));
		checkCudaErrors(cudaFree(GPU_data_in));
		free(CPU_data_in);
		free(CPU_data_out_index);
		free(CPU_data_out_time);
	}
	
	for(unsigned mod2 = 1; mod2 <= 1; mod2 = mod2 * 2){////268435456 = 1gb, 536870912 = 2gb, 1073741824 = 4gb, 2147483648 = 8gb, 4294967296 = 16gb.
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin
		//int data_size = 2 * 256 * 1024 * 32;/////size = iteration * stride = 32 2mb pages.
		unsigned mod = 2147483648;
		if(mod > 3221225472){
			mod = 3221225472;
		}
		unsigned data_size = mod;
		if(data_size < 4194304){//////////data size at least 16mb to prevent L2 prefetch
			data_size = 4194304;
		}
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		unsigned iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		unsigned *CPU_data_in;
		CPU_data_in = (unsigned*)malloc(sizeof(unsigned) * data_size);
		init_cpu_data(CPU_data_in, data_size, data_stride, mod, iterations);
		
		
		unsigned reduced_iter = iterations;
		if(reduced_iter > 512){
			reduced_iter = 512;
		}else if(reduced_iter < 16){
			reduced_iter = 16;
		}
		
		unsigned *CPU_data_out_index;
		CPU_data_out_index = (unsigned*)malloc(sizeof(unsigned) * reduced_iter);
		long long int *CPU_data_out_time;
		CPU_data_out_time = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		unsigned *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(unsigned) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(unsigned) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		unsigned *GPU_data_out_index;
		checkCudaErrors(cudaMalloc(&GPU_data_out_index, sizeof(unsigned) * reduced_iter));
		long long int *GPU_data_out_time;
		checkCudaErrors(cudaMalloc(&GPU_data_out_time, sizeof(long long int) * reduced_iter));
		
		tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, GPU_data_out_index, GPU_data_out_time, clock_rate, mod, data_stride);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		cudaMemcpy(CPU_data_out_index, GPU_data_out_index, sizeof(unsigned) * reduced_iter, cudaMemcpyDeviceToHost);
		cudaMemcpy(CPU_data_out_time, GPU_data_out_time, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
				

		fprintf(pFile, "###################data_stride%u#########################\n", data_stride);
		fprintf (pFile, "###############Mod%u##############%u\n", mod, iterations);
		for (unsigned it = 0; it < reduced_iter; it++){			
			fprintf (pFile, "%u %fms %lldcycles\n", CPU_data_out_index[it], (double)CPU_data_out_time[it] / (float)clock_rate, CPU_data_out_time[it]);
			//fprintf (pFile, "%d %fms\n", it, CPU_data_out_time[it] / (float)clock_rate);
			//printf ("%d %fms\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate);
		}
		
		checkCudaErrors(cudaFree(GPU_data_out_index));
		checkCudaErrors(cudaFree(GPU_data_out_time));
		checkCudaErrors(cudaFree(GPU_data_in));
		free(CPU_data_in);
		free(CPU_data_out_index);
		free(CPU_data_out_time);
	}
	
	for(unsigned mod2 = 1; mod2 <= 1; mod2 = mod2 * 2){////268435456 = 1gb, 536870912 = 2gb, 1073741824 = 4gb, 2147483648 = 8gb, 4294967296 = 16gb.
		counter++;
		///////////////////////////////////////////////////////////////////CPU data begin
		//int data_size = 2 * 256 * 1024 * 32;/////size = iteration * stride = 32 2mb pages.
		unsigned mod = 3221225472;
		if(mod > 3221225472){
			mod = 3221225472;
		}
		unsigned data_size = mod;
		if(data_size < 4194304){//////////data size at least 16mb to prevent L2 prefetch
			data_size = 4194304;
		}
		//int iterations = data_size / data_stride;
		//int iterations = 1024 * 256 * 8;
		unsigned iterations = mod / data_stride;////32 * 32 * 4 / 32 * 2 = 256
	
		unsigned *CPU_data_in;
		CPU_data_in = (unsigned*)malloc(sizeof(unsigned) * data_size);
		init_cpu_data(CPU_data_in, data_size, data_stride, mod, iterations);
		
		
		unsigned reduced_iter = iterations;
		if(reduced_iter > 512){
			reduced_iter = 512;
		}else if(reduced_iter < 16){
			reduced_iter = 16;
		}
		
		unsigned *CPU_data_out_index;
		CPU_data_out_index = (unsigned*)malloc(sizeof(unsigned) * reduced_iter);
		long long int *CPU_data_out_time;
		CPU_data_out_time = (long long int*)malloc(sizeof(long long int) * reduced_iter);
		///////////////////////////////////////////////////////////////////CPU data end	
	
		///////////////////////////////////////////////////////////////////GPU data in	
		unsigned *GPU_data_in;
		checkCudaErrors(cudaMalloc(&GPU_data_in, sizeof(unsigned) * data_size));	
		cudaMemcpy(GPU_data_in, CPU_data_in, sizeof(unsigned) * data_size, cudaMemcpyHostToDevice);
		
		///////////////////////////////////////////////////////////////////GPU data out
		unsigned *GPU_data_out_index;
		checkCudaErrors(cudaMalloc(&GPU_data_out_index, sizeof(unsigned) * reduced_iter));
		long long int *GPU_data_out_time;
		checkCudaErrors(cudaMalloc(&GPU_data_out_time, sizeof(long long int) * reduced_iter));
		
		tlb_latency_test<<<1, 1>>>(GPU_data_in, iterations, GPU_data_out, GPU_data_out_index, GPU_data_out_time, clock_rate, mod, data_stride);///////////////kernel is here	
		cudaDeviceSynchronize();
				
		cudaMemcpy(CPU_data_out_index, GPU_data_out_index, sizeof(unsigned) * reduced_iter, cudaMemcpyDeviceToHost);
		cudaMemcpy(CPU_data_out_time, GPU_data_out_time, sizeof(long long int) * reduced_iter, cudaMemcpyDeviceToHost);
				

		fprintf(pFile, "###################data_stride%u#########################\n", data_stride);
		fprintf (pFile, "###############Mod%u##############%u\n", mod, iterations);
		for (unsigned it = 0; it < reduced_iter; it++){			
			fprintf (pFile, "%u %fms %lldcycles\n", CPU_data_out_index[it], (double)CPU_data_out_time[it] / (float)clock_rate, CPU_data_out_time[it]);
			//fprintf (pFile, "%d %fms\n", it, CPU_data_out_time[it] / (float)clock_rate);
			//printf ("%d %fms\n", CPU_data_out_index[it], CPU_data_out_time[it] / (float)clock_rate);
		}
		
		checkCudaErrors(cudaFree(GPU_data_out_index));
		checkCudaErrors(cudaFree(GPU_data_out_time));
		checkCudaErrors(cudaFree(GPU_data_in));
		free(CPU_data_in);
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
