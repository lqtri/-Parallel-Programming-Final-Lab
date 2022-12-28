#include <stdio.h>
#include <stdint.h>
#include <string.h>

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

__global__ void energyCalculatingKernel(uint8_t *inPixels, int width, int height,
                               	float *xSobel, float* ySobel,
                               	uint8_t *outPixels) {

  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < height && c < width) {
    int i = r * width + c;
	float xvalue = 0, yvalue = 0;
    for (int r_sobel = -1; r_sobel <= 1; r_sobel++) {
      for (int c_sobel = -1; c_sobel <= 1; c_sobel++) {

        int i_sobel = (r_sobel+1)*3 + c_sobel+1;
        int r_img, c_img;

        if (r+r_sobel<0)
			r_img = 0;
		else if (r+r_sobel>height-1)
			r_img = height - 1;
		else 
			r_img = r + r_sobel;

		if (c+c_sobel<0)
			c_img = 0;
		else if (c+c_sobel>width-1)
			c_img = width - 1;
		else 
			c_img = c + c_sobel;

        int i_img = r_img * width + c_img;
		
        xvalue += xSobel[i_sobel] * inPixels[i_img];
		yvalue += ySobel[i_sobel] * inPixels[i_img];
      }
    }
	outPixels[i] = abs(xvalue) + abs(yvalue);
  }
}

void seamCarving(uint8_t * inPixels, int width, int height, float * xSobel, float * ySobel,
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
		CHECK(cudaMalloc(&d_inPixels, nBytes*3));
		CHECK(cudaMalloc(&d_grayPixels, nBytes));
		CHECK(cudaMalloc(&d_energy, nBytes));

		// Host copies data to device memories
		CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes*3, cudaMemcpyHostToDevice));

		// Host invokes kernel function to add vectors on device
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
		convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
		cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

		// Host copies result from device memory
		CHECK(cudaMemcpy(grayPixels, d_grayPixels, nBytes, cudaMemcpyDeviceToHost));
		
		energyCalculatingKernel<<<gridSize, blockSize>>>(d_grayPixels, width, height, xSobel, ySobel, d_energy);
		cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

		// Host copies result from device memory
		CHECK(cudaMemcpy(energy, d_energy, nBytes, cudaMemcpyDeviceToHost));

		// Host frees device memories
		CHECK(cudaFree(d_inPixels));
		CHECK(cudaFree(d_grayPixels));
		CHECK(cudaFree(d_energy));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
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

	// Convert RGB to grayscale using device
	uint8_t *grayPixels = (uint8_t *)malloc(width * height);
	uint8_t *energy = (uint8_t *)malloc(width * height);
	dim3 blockSize(32, 32); // Default
	if (argc == 5) {
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}

	//Convert to gray scale by device
	seamCarving(inPixels, width, height, xSobel, ySobel, grayPixels, energy, true, blockSize);
	char *outFileNameBase = strtok(argv[1], "."); // Get rid of extension
	writePnm(grayPixels,1, width, height, concatStr(outFileNameBase, "_grayscale.pnm"));
	writePnm(energy,1, width, height, concatStr(outFileNameBase, "_energy.pnm"));

	// Free memories
	free(inPixels);
	free(grayPixels);
	free(energy);

	return 0;
}