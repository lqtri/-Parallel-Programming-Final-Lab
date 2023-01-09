#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, 
		int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}
void writePnmEnergy(int * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

__global__ void convertRgb2GrayKernel(uint8_t *inPixels, int width, int height,
                                      uint8_t *outPixels) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < height && c < width) {
    int i = r * width + c;
    outPixels[i] = 0.299 * inPixels[3 * i] + 0.587 * inPixels[3 * i + 1] +
                   0.114 * inPixels[3 * i + 2];
  }
}

__global__ void energyCalculatorKernel(uint8_t *inPixels, int width, int height,
                               	float *xSobel, float* ySobel,
                               	int* energy) {

  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < height && c < width) {
    int i = r * width + c;
	float xvalue = 0, yvalue = 0;
    for (int r_sobel = 0; r_sobel < 3; r_sobel++) {
      for (int c_sobel = 0; c_sobel < 3; c_sobel++) {

        int i_sobel = r_sobel * 3 + c_sobel;
		int r_img = r - 1 + r_sobel;
		int c_img = c - 1 + c_sobel;
		r_img = min(max(0,r_img), height-1);
		c_img = min(max(0,c_img), width-1);

        int i_img = r_img * width + c_img;
        xvalue += xSobel[i_sobel] * inPixels[i_img];
		yvalue += ySobel[i_sobel] * inPixels[i_img];
      } 
    }
	energy[i] = abs(xvalue) + abs(yvalue);
  }
}

__global__ void seamImportanceCalculator (int* map, int8_t* backtrack, int width, int height) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t<width)
		backtrack[t] = 0;
	__syncthreads();

	int c = blockIdx.x * blockDim.x + threadIdx.x;
	for (int r = 1; r < height; r++) {
		if (c < width) {
			int idx = r*width+c;
			int idx_cmp = idx-width;
			int d=0, min = map[idx_cmp];

			if (c==0){
				if (map[idx_cmp+1]<min){
					min = map[idx_cmp+1];
					d = 1;
				}
			}
			else if(c==width-1){
				if (map[idx_cmp-1]<min){
					min = map[idx_cmp-1];
					d = -1;
				}		
			}
			else{
				if (map[idx_cmp-1]<min){
					min = map[idx_cmp-1];
					d = -1;
				}
				if (map[idx_cmp+1]<min){
					min = map[idx_cmp+1];
					d = 1;
				}
			}
			map[idx] += min;
			backtrack[idx] = d;
			__syncthreads();
		}
	}
}

__global__ void transferDataKernel (int *dst, int *src, int width, int height){
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	if (r<height && c<width) {	
		int i = r*width+c;
		dst[i] = src[i];
	}
}

__global__ void copyARowKernel (int *dst, int *src, int head, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		dst[i] = src[i+head];
}

__global__ void removeInEnergy (int * energy, int start, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i + start + 1 < n){
		int value = energy[i + start + 1];
		__syncthreads();
		energy[i+start] = value;
	}
}
__global__ void removeInPixels (uint8_t * inPixels, int start, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i + start + 3 < n){
		uint8_t value = inPixels[i + start + 3];
		__syncthreads();
		inPixels[i+start] = value;
	}
}	

__global__ void createIndices (int* out_indices, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<n){
		out_indices[i] = i;
	}
}

__global__ void findMinsKernel(int * in_mins, int* in_indices, int n, int* mins, int* min_indices)
{
    int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
    int i = numElemsBeforeBlk + threadIdx.x;
    for (int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride){
			if (in_mins[i] > in_mins[i+stride]){
				in_mins[i] = in_mins[i+stride];
				in_indices[i] = in_indices[i+stride];
			}
		}
        __syncthreads(); 
    }

    if (threadIdx.x == 0){
        mins[blockIdx.x] = in_mins[numElemsBeforeBlk];
		min_indices[blockIdx.x] = in_indices[numElemsBeforeBlk];
	}
}

int findMinIndexFromHost (int * mins, int * indices, int n){
	int min = mins[0];
	int min_index = indices[0];

	for (int i=1; i<n; i++){
		if (mins[i] < min){
			min = mins[i];
			min_index = indices[i];
		}
	}
	return min_index;
}
int findMinIndexFromHost1 (int * mins, int n){
	int min = mins[0];
	int min_index = 0;

	for (int i=1; i<n; i++){
		if (mins[i] < min){
			min = mins[i];
			min_index = i;
		}
	}
	return min_index;
}

void printSeam(int8_t* backtrack, int width, int index){
	while (index >= 0){
		printf("%d ", index);
		index = index - width + backtrack[index];
	}
}

void removeSeamFromDevice (uint8_t* d_inPixels, int * d_energy, int width, int height, int min_index, int8_t * backtrack, dim3 blockSize) {
	int n = width*height;
	min_index += (height-1)*width;
	for (int h = 0; h < height; h++) {
		dim3 gridSize1((n-min_index-1)/blockSize.x+1,1);
		dim3 gridSize2((n-min_index-1)*3/blockSize.x+1,1);
		removeInEnergy<<<gridSize1, blockSize>>>(d_energy, min_index, n-h);	
		removeInPixels<<<gridSize2, blockSize>>>(d_inPixels, 3*min_index, 3*(n-h));
		cudaDeviceSynchronize();
        CHECK(cudaGetLastError());	
		min_index = min_index - width + backtrack[min_index];
	}
}

void seamCarving(uint8_t * inPixels, int width, int height, float * xSobel, float * ySobel,
				uint8_t * grayPixels, int * energy, dim3 blockSize=dim3(1), float scalingPercentage=0.8)
{
	GpuTimer timer;
	timer.Start();
	int new_width = width*scalingPercentage;
	size_t nBytes = width * height * sizeof(uint8_t);

	// Host allocates memories on device
	uint8_t *d_inPixels, *d_grayPixels;
	int *d_energy;
	float *d_xSobel, *d_ySobel;
	CHECK(cudaMalloc(&d_inPixels, nBytes*3));
	CHECK(cudaMalloc(&d_grayPixels, nBytes));
	CHECK(cudaMalloc(&d_energy, width * height * sizeof(int)));
	CHECK(cudaMalloc(&d_xSobel, 9*sizeof(float)));
	CHECK(cudaMalloc(&d_ySobel, 9*sizeof(float)));

	// Host copies data to device memories
	CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes*3, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_xSobel, xSobel, 9*sizeof(float), cudaMemcpyHostToDevice))
	CHECK(cudaMemcpy(d_ySobel, ySobel, 9*sizeof(float), cudaMemcpyHostToDevice))
	
	// Host invokes kernel function to add vectors on device
	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
	convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
	cudaDeviceSynchronize();
	CHECK(cudaGetLastError());

	// Host copies result from device memory
	CHECK(cudaMemcpy(grayPixels, d_grayPixels, nBytes, cudaMemcpyDeviceToHost));
	
	energyCalculatorKernel<<<gridSize, blockSize>>>(d_grayPixels, width, height, d_xSobel, d_ySobel, d_energy);
	cudaDeviceSynchronize();
	CHECK(cudaGetLastError());

	// Host copies result from device memory
	CHECK(cudaMemcpy(energy, d_energy, width*height*sizeof(int), cudaMemcpyDeviceToHost));

	for (int w = width; w > new_width; w--){
		int *d_map;
		int8_t *d_backtrack;
		CHECK(cudaMalloc(&d_map, w*height*sizeof(int)));
		CHECK(cudaMalloc(&d_backtrack, w*height*sizeof(int8_t)));
		int8_t* backtrack = (int8_t*)malloc(w*height*sizeof(int8_t));
		CHECK(cudaMemcpy(d_map, d_energy, w*height*sizeof(int), cudaMemcpyDeviceToHost));

		dim3 gridSize1((w-1)/blockSize.x+1, (height-1)/blockSize.y+1);
		dim3 gridSize2((w-1)/blockSize.x+1, 1);
		dim3 gridSize3((w-1)/(2*blockSize.x)+1, 1);

		//Copy energy's first row to map
		// copyARowKernel<<<gridSize2, blockSize>>>(d_map, d_energy, 0, w);
		// cudaDeviceSynchronize();
		// CHECK(cudaGetLastError());


		//calculate seam importance
		seamImportanceCalculator<<<gridSize2, blockSize>>>(d_map, d_backtrack, w, height);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		//CHECK MAP
		// int* mapcheck = (int*)malloc(w*height*sizeof(int));
		// CHECK(cudaMemcpy(mapcheck, d_map, w*height*sizeof(int), cudaMemcpyDeviceToHost));
		// for (int i = 0; i<height; i++){
		// 	for(int j=0; j<w; j++)
		// 		printf("%d ", mapcheck[i*w+j]);
		// 	printf("\n");
		// }
		// free(mapcheck);


		//*****FIND MIN SEAM FROM DEVICE*****
		// int* d_last_row, *d_indices;
		// CHECK(cudaMalloc(&d_last_row, w*sizeof(int)));
		// CHECK(cudaMalloc(&d_indices, w*sizeof(int)));
		// copyARowKernel<<<gridSize2, blockSize>>>(d_last_row, d_map, (height-1)*w, w);
		// createIndices<<<gridSize2, blockSize>>>(d_indices, w);
		// cudaDeviceSynchronize();
		// CHECK(cudaGetLastError());

		// size_t mins_size = gridSize3.x * sizeof(int);
		// int* mins = (int*) malloc(mins_size);
		// int* min_indices = (int*) malloc(mins_size);		
		// int* d_mins;
		// int* d_min_indices;
		// CHECK(cudaMalloc(&d_mins, mins_size));
		// CHECK(cudaMalloc(&d_min_indices, mins_size));
		// findMinsKernel<<<gridSize3, blockSize>>>(d_last_row, d_indices, w, d_mins, d_min_indices);
		// cudaDeviceSynchronize();
		// CHECK(cudaGetLastError());
		// CHECK(cudaMemcpy(mins, d_mins, mins_size, cudaMemcpyDeviceToHost));
		// CHECK(cudaMemcpy(min_indices, d_min_indices, mins_size, cudaMemcpyDeviceToHost));
		// int min_index = findMinIndexFromHost(mins, min_indices, gridSize3.x);

		// *****FIND MIN SEAM FROM HOST*****
		int* last_map_row = (int*)malloc(w*sizeof(int));
		CHECK(cudaMemcpy(last_map_row, &d_map[(height-1)*w], w*sizeof(int), cudaMemcpyDeviceToHost));
		int min_index = findMinIndexFromHost1(last_map_row, w);
		free(last_map_row);

		// Remove seam
		CHECK(cudaMemcpy(backtrack, d_backtrack, w*height*sizeof(int8_t), cudaMemcpyDeviceToHost));
		removeSeamFromDevice(d_inPixels, d_energy, w, height, min_index, backtrack, blockSize);

		//Free
		//Uncomment if use device for finding minimum value
		// free(mins);
		// free(min_indices);
		// CHECK(cudaFree(d_mins));
		// CHECK(cudaFree(d_min_indices));
		// CHECK(cudaFree(d_last_row));
		// CHECK(cudaFree(d_indices));


		CHECK(cudaFree(d_map));
		CHECK(cudaFree(d_backtrack));
		free(backtrack);
	}

	CHECK(cudaMemcpy(inPixels, d_inPixels, 3*new_width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost));

	// Host frees device memories
	CHECK(cudaFree(d_inPixels));
	CHECK(cudaFree(d_grayPixels));
	CHECK(cudaFree(d_energy));
	CHECK(cudaFree(d_xSobel));
	CHECK(cudaFree(d_ySobel));

	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (use device): %f ms\n\n", time);
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n");
}

int main(int argc, char ** argv)
{
	printDeviceInfo();

	// Read input image file
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

    float xSobel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	float ySobel[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};


	float scalingPercentage = 0.8;
	int new_width = width*scalingPercentage;

	// Convert RGB to grayscale using device
	uint8_t *grayPixels = (uint8_t *)malloc(width * height);
	int *energy = (int*)malloc(width * height * sizeof(int));

	dim3 blockSize(32, 32); // Default
	if (argc == 4) {
		blockSize.x = atoi(argv[2]);
		blockSize.y = atoi(argv[3]);
	}

	seamCarving(inPixels, width, height, xSobel, ySobel, grayPixels, energy, blockSize, scalingPercentage);
	char *outFileNameBase = strtok(argv[1], ".");
	writePnm(grayPixels,1, width, height, concatStr("", "device_gray.pnm"));
	writePnmEnergy(energy,1, width, height, concatStr("", "device_sobel.pnm"));
	writePnm(inPixels, 3, new_width, height, concatStr("", "device_result.pnm"));

	// Free memories
	free(inPixels);
	free(grayPixels);
	free(energy);

	return 0;
}