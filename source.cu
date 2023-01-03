#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

__constant__ float dc_xSobel[9];
__constant__ float dc_ySobel[9];

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

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void printMatrix (uint8_t *in, int width, int height){
	for (int i=0; i<height; i++){
		for (int j=0; j<width; j++){
			printf("%d ", in[i*width+j]);
		}
		printf("\n");
	}
}

__global__ void convertRgb2GrayKernel(uint8_t *inPixels, int width, int height,
                                      uint8_t *outPixels) {
  // TODO
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue
  if (r < height && c < width) {
    int i = r * width + c;
    outPixels[i] = 0.299 * inPixels[3 * i] + 0.587 * inPixels[3 * i + 1] +
                   0.114 * inPixels[3 * i + 2];
  }
}

__global__ void energyCalculatorKernel(uint8_t *inPixels, int width, int height,
                               	float *xSobel, float* ySobel,
                               	uint8_t *outPixels) {

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
	outPixels[i] = abs(xvalue) + abs(yvalue);
  }
}

__global__ void seamsImportanceCalculator (uint8_t* map, int8_t* backtrack, int width, int height) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	if (r<height && c<width) {
		int i = r*width+c;
		int j = i - width;

		if (j<0) {
			backtrack[i] = 0;
			return;
		}

		if (c == 0) {
			if (map[j] < map[j+1]){
				map[i] += map[j];
				backtrack[i] = 0;
			}
			else {
				map[i] += map[j+1];
				backtrack[i] = 1;
			}
		}
		else if (c == width-1){
			if (map[j-1] < map[j]){
				map[i] += map[j-1];
				backtrack[i] = -1;
			}
			else {
				map[i] += map[j];
				backtrack[i] = 0;
			}
		}
		else {
			int min = map[j], idx = 0;
			if (map[j-1]<min){
				min = map[j-1];
				idx = -1;
			}
			if (map[j+1]<min){
				min = map[j+1];
				idx = 1;
			}
			map[i] = min;
			backtrack[i] = idx;
		}
	}
}

__global__ void transferDataKernel (uint8_t *dst, uint8_t *src, int width, int height){
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	if (r<height && c<width) {	
		int i = r*width+c;
		dst[i] = src[i];
	}
}

__global__ void copyARowKernel (uint8_t *dst, uint8_t *src, int head, int tail){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i+head < tail)
		dst[i] = src[i+head];
}

__global__ void findMinsKernel(uint8_t * in, int n, uint8_t* mins, uint8_t* min_indices)
{
	__shared__ uint8_t sm_min_index[1];
    int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
    int i = numElemsBeforeBlk + threadIdx.x;
    for (int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride){
			if (in[i] > in[i+stride]){
				in[i] = in[i+stride];
				sm_min_index[0] = i+stride;
			}
			else sm_min_index[0] = i;
		}
        __syncthreads(); // Synchronize within each block
    }

    if (threadIdx.x == 0){
        mins[blockIdx.x] = in[numElemsBeforeBlk];
		min_indices[blockIdx.x] = sm_min_index[0];
	}
}

uint8_t findMinIndexFromHost (uint8_t * mins, uint8_t * indices, int n){
	uint8_t min = mins[0];
	uint8_t min_index = indices[0];

	for (int i=1; i<n; i++){
		if (mins[i] < min){
			min = mins[i];
			min_index = indices[i];
		}
	}
	printf("Min energy: %d\nIts index: %d\n", min, min_index);
	return min_index;
}

// __global__ void removeSeamKernel (uint8_t* energy, uint8_t* inPixels, int width, int height, uint8_t* index, int8_t* backtrack){
// }

void seamCarving(uint8_t * inPixels, int width, int height, int new_width, float * xSobel, float * ySobel,
				uint8_t * grayPixels, uint8_t * energy, bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
		// Host running
	}
	else // use device
	{
		size_t nBytes = width * height * sizeof(uint8_t);

		// Host allocates memories on device
		uint8_t *d_inPixels, *d_grayPixels, *d_energy;
		float *d_xSobel, *d_ySobel;
		CHECK(cudaMalloc(&d_inPixels, nBytes*3));
		CHECK(cudaMalloc(&d_grayPixels, nBytes));
		CHECK(cudaMalloc(&d_energy, nBytes));
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
		CHECK(cudaMemcpy(energy, d_energy, nBytes, cudaMemcpyDeviceToHost));

		uint8_t *map;
		int8_t *backtrack;
		CHECK(cudaMalloc(&map, nBytes));
		CHECK(cudaMalloc(&backtrack, nBytes));

		for (int w = width; w > new_width; w--){
			//initialize map (copy from energy), backtrack
			// transferDataKernel<<<gridSize, blockSize>>>(map, d_energy, w, height);
			// cudaDeviceSynchronize();
			// CHECK(cudaGetLastError());
			CHECK(cudaMemcpy(map, energy, nBytes, cudaMemcpyHostToDevice));

			//Test map correctness
			// uint8_t* testmap = (uint8_t*)malloc(w*height*sizeof(uint8_t));
			// CHECK(cudaMemcpy(testmap, map, nBytes, cudaMemcpyDeviceToHost));
			// printf("\nMAP\n");
			// printMatrix(testmap, w, height);
			// free(testmap);

			//calculate seam importance
			seamsImportanceCalculator<<<gridSize, blockSize>>>(map, backtrack, w, height);
			cudaDeviceSynchronize();
			CHECK(cudaGetLastError());

			//Test importance
			uint8_t* testimportance = (uint8_t*)malloc(w*height*sizeof(uint8_t));
			CHECK(cudaMemcpy(testimportance, map, nBytes, cudaMemcpyDeviceToHost));
			printf("\nMAP\n");
			printMatrix(testimportance, w, height);
			free(testimportance);


			dim3 gridSize1((w-1)/(blockSize.x)+1, 1);
			dim3 gridSize2((w-1)/(2*blockSize.x)+1, 1);

			uint8_t* d_last_row;
			CHECK(cudaMalloc(&d_last_row, w*sizeof(uint8_t)));
			copyARowKernel<<<gridSize1, blockSize>>>(d_last_row, map, (height-1)*width, height*width);
			cudaDeviceSynchronize();
			CHECK(cudaGetLastError());

			uint8_t* row = (uint8_t*)malloc(w*sizeof(uint8_t));
			CHECK(cudaMemcpy(row, d_last_row, w*sizeof(uint8_t), cudaMemcpyDeviceToHost));
			printf("\nLAST SEAM IMPORTANCE ROW\n");
			for (int t = 0; t<w; t++){
				printf("%d ", row[t]);
			}
			printf("\n");
			free(row);

			size_t mins_size = gridSize2.x * sizeof(uint8_t);
			uint8_t* mins = (uint8_t*) malloc(mins_size);
			uint8_t* min_indices = (uint8_t*) malloc(mins_size);		

			uint8_t* d_mins;
			uint8_t* d_min_indices;
			CHECK(cudaMalloc(&d_mins, mins_size));
			CHECK(cudaMalloc(&d_min_indices, mins_size));

			findMinsKernel<<<gridSize2, blockSize>>>(d_last_row, w, d_mins, d_min_indices);
			cudaDeviceSynchronize();
			CHECK(cudaGetLastError());

			CHECK(cudaMemcpy(mins, d_mins, mins_size, cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(min_indices, d_min_indices, mins_size, cudaMemcpyDeviceToHost));

			printf("\nMIN ELEMENT FROM DEVICE\n");
			for (int t = 0; t < gridSize2.x; t++){
				printf("%d ", mins[t]);
			}
			printf("\n");
			for (int t = 0; t < gridSize2.x; t++){
				printf("%d ", min_indices[t]);
			}
			printf("\n");

			printf("\nFIND MIN ENERGY\n");
			findMinIndexFromHost(mins, min_indices, gridSize2.x);

			free(mins);
			free(min_indices);
			CHECK(cudaFree(d_mins));
			CHECK(cudaFree(d_min_indices));
			CHECK(cudaFree(d_last_row));
			
			break;
		}

		// Host frees device memories
		CHECK(cudaFree(d_inPixels));
		CHECK(cudaFree(d_grayPixels));
		CHECK(cudaFree(d_energy));
		CHECK(cudaFree(d_xSobel));
		CHECK(cudaFree(d_ySobel));
		CHECK(cudaFree(map));
		CHECK(cudaFree(backtrack));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", useDevice == true? "use device" : "use host", time);
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

	int new_width = 450;
	printf("\nImage size (width x height): %i x %i\n", width, height);
	// printf("\nEnter a number of new width size to scale: ");
	// scanf("%d", &new_width);

    float xSobel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	float ySobel[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

	// Convert RGB to grayscale using device
	uint8_t *grayPixels = (uint8_t *)malloc(width * height);
	uint8_t *energy;
	cudaMallocHost(&energy, width * height * sizeof(uint8_t));

	dim3 blockSize(32, 32); // Default
	if (argc == 4) {
		blockSize.x = atoi(argv[2]);
		blockSize.y = atoi(argv[3]);
	}

	seamCarving(inPixels, width, height, new_width, xSobel, ySobel, grayPixels, energy, true, blockSize);
	char *outFileNameBase = strtok(argv[1], "."); // Get rid of extension
	writePnm(grayPixels,1, width, height, concatStr(outFileNameBase, "_grayscale.pnm"));
	writePnm(energy,1, width, height, concatStr(outFileNameBase, "_energy.pnm"));

	// Free memories
	free(inPixels);
	free(grayPixels);
	cudaFreeHost(energy);

	return 0;
}