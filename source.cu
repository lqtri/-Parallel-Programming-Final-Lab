#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define FILTER_WIDTH 3
__constant__ float dc_filter[FILTER_WIDTH * FILTER_WIDTH];

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

void convertRgb2Gray(uint8_t * inPixels, int width, int height,uint8_t * outPixels, bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
        // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                int i = r * width + c;
                uint8_t red = inPixels[3 * i];
                uint8_t green = inPixels[3 * i + 1];
                uint8_t blue = inPixels[3 * i + 2];
                outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
            }
        }
	}
	else // use device
	{
		size_t nBytes = width * height * sizeof(uint8_t);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// Host allocates memories on device
		uint8_t *d_inPixels,*d_outPixels;
		CHECK(cudaMalloc(&d_inPixels, nBytes*3));
		CHECK(cudaMalloc(&d_outPixels, nBytes));

		// Host copies data to device memories
		CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes*3, cudaMemcpyHostToDevice));

		// Host invokes kernel function to add vectors on device
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
		convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outPixels);

		// Host copies result from device memory
		CHECK(cudaMemcpy(outPixels, d_outPixels, nBytes, cudaMemcpyDeviceToHost));

		// Host frees device memories
		CHECK(cudaFree(d_inPixels));
		CHECK(cudaFree(d_outPixels));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void sobelOperator(uchar3* inPixels, int width, int height, uchar3 * outPixels, int* verticalSobel, int* horizontalSobel)
{
	int filterR = FILTER_WIDTH/2;
	for (int r = 0; r < height; r++){
		for (int c = 0; c < width; c++){
			int i = r * width + c;
			int Gx = 0;
			int Gy = 0;
			for (int i = -filterR; i <= filterR; i++){
				for (int j =-filterR; j <= filterR; j++){
					int x = c + j;
					int y = r + i;

					if (x < 0) {
						x = 0;
					}
					else if (x >= width) {
						x = width - 1;
					}
					if (y < 0) {
						y = 0;
					}
					else if (y >= height) {
						y = height - 1;
					}

					outX += inPixels[y*width + x].x * filter[(i + filterR) * filterR + j + filterR] ;
				}
			}

			outPixels[i] = abs(Gx) + abs(Gy)
		}
	}
}

void HostSeamCarving(uint8_t * inPixels, int width, int height,
					int* verticalSobel, int* horizontalSobel,
					float scalePercentage = 0.75, dim3 blockSize=dim3(1))
{
	char * outFileNameBase = strtok("gido", "."); // Get rid of extension
	uint8_t* grayOut = (uint8_t *)malloc(width * height * sizeof(uint8_t)); 
	convertRgb2Gray(inPixels, width, height, grayOut);
	writePnm(grayOut,1, width, height, concatStr(outFileNameBase, "_gray.pnm"));
	// uint8_t* SobelOut = (uint8_t *)malloc(width * height * sizeof(uchar3)); 



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

    int filterWidth = FILTER_WIDTH;
    //vertical sobel
    int verticalSobel[] = {-1,0,1,-2,0,2,-1,0,1};
    //horizontal sobel
    int horizontalSobel[]= {-1,-2,-1,0,0,0,1,2,1};


	HostSeamCarving(inPixels, width, height, verticalSobel, horizontalSobel );


	// // Blur input image not using device
	
	
    // // Blur input image using device, kernel 1
    // dim3 blockSize(32, 32); // Default

	// // Free memories
	// free(inPixels);
}