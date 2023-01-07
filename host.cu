#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define FILTER_WIDTH 3

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

void convertRgb2Gray(uint8_t * inPixels, int width, int height,uint8_t * outPixels, bool useDevice=false, dim3 blockSize=dim3(1))
{
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

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void sobelOperator(uint8_t* inPixels, int width, int height, uint8_t * outPixels, \
				int* verticalSobel, int* horizontalSobel)
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

					Gx += inPixels[y*width + x] * verticalSobel[(i + filterR) * FILTER_WIDTH + j + filterR] ;
					Gy += inPixels[y*width + x] * horizontalSobel[(i + filterR) * FILTER_WIDTH + j + filterR] ;
				}
			}

			outPixels[i] = abs(Gx) + abs(Gy);
		}
	}
}

void minimum_energy(uint8_t * inPixels, int width, int height, int8_t * out_backtrack, uint32_t * out_map)
{
	uint32_t * map = (uint32_t *)malloc(width * height * sizeof(uint32_t));
	for (int i = 0; i < width * height; i++){ map[i] = inPixels[i]; }

	int8_t * backtrack	 = (int8_t *)malloc(width * height * sizeof(int8_t));
	for (int i = 0; i < width * height; i++){ backtrack[i] = 0; }
	for (int r = 1 ; r < height; r++)
	{
		for (int c = 0 ; c < width; c++)
		{
			int min = -1;
			int idx = 0;
			for (int iBefore = -1; iBefore <= 1; iBefore++){
				int x = (r-1) * width;
				int y = c + iBefore;
				if (y < 0) y = 0;
				if (y>width-1) y = width-1;
				if (map[x + y] < min or min == -1)
				{
					min = map[x + y];
					idx = y - c;
				}
			}
			map [r * width + c] += min;
			backtrack[r * width + c] = idx;
		}
	}

	for (int i = 0; i < width * height; i++){ out_map[i] = map[i]; }
	free(map);
	for (int i = 0; i < width * height; i++){ out_backtrack[i] = backtrack[i]; }
	free(backtrack);
}


void remove_element(uint8_t *array, int index, int array_length)
{
   int i;
   for(i = index; i < array_length - 1; i++) array[i] = array[i + 1];
}


void remove_element3(uint8_t *array, int index, int array_length)
{
   int i;
   for(i = index; i < array_length - 3; i++) array[i] = array[i + 3];
}

void HostSeamCarving(uint8_t * inPixels, int width, int height,
					int* verticalSobel, int* horizontalSobel,
					float scalePercentage = 0.8, dim3 blockSize=dim3(1))
{
	uint8_t* grayOut = (uint8_t *)malloc(width * height * sizeof(uint8_t)); 
	uint8_t* sobelOut = (uint8_t *)malloc(width * height * sizeof(uint8_t)); 

	GpuTimer timer1;
	timer1.Start();
	convertRgb2Gray(inPixels, width, height, grayOut);
	timer1.Stop();
	float time1 = timer1.Elapsed();
    char filename1[] = "host_gray.pnm";
	writePnm(grayOut,1, width, height, filename1);

	GpuTimer timer2;
	timer2.Start();
	sobelOperator(grayOut, width, height, sobelOut, verticalSobel, horizontalSobel);
	timer2.Stop();
	float time2 = timer2.Elapsed();
    char filename2[] = "host_sobel.pnm";
	writePnm(sobelOut,1, width, height, filename2);

	GpuTimer timer;
	timer.Start();
	int scaling_times = width * (1 - scalePercentage);
	int scaling_width = width;
	for (int time = 0; time < scaling_times; time++)
	{
		int8_t * backtrack	 = (int8_t *)malloc(scaling_width * height * sizeof(int8_t));
		uint32_t* map = (uint32_t *)malloc(width * height * sizeof(uint32_t)); 
		minimum_energy(sobelOut, scaling_width, height, backtrack, map);
		int idx = 0;
		int min = map[(height -1) * scaling_width];

		for (int i = 1 ; i < scaling_width; i++)
		{
			int target = map[(height -1) * scaling_width + i];
			if ( target < min)
			{
				min = target;
				idx = i;
			}
		}
		for (int i = 1 ; i <= height; i++)
		{
			int target_id = (height - i) * scaling_width + idx;
			idx += backtrack[target_id]; 
			remove_element3(inPixels, target_id*3, (height*scaling_width - i + 1)*3);
			remove_element(sobelOut, target_id, height*scaling_width - i + 1);
		}
	
		scaling_width -= 1;
		free(map);
		free(backtrack);
	}
	timer.Stop();
	float time = timer.Elapsed();

	printf("Processing time (use host): %f ms\n\n", time+time1+time2);
    char filename3[] = "host_result.pnm";
	writePnm(inPixels,3, scaling_width, height, filename3);

    free(grayOut);
    free(sobelOut);
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

    //vertical sobel
    int verticalSobel[] = {-1,0,1,-2,0,2,-1,0,1};
    //horizontal sobel
    int horizontalSobel[]= {-1,-2,-1,0,0,0,1,2,1};

	HostSeamCarving(inPixels, width, height, verticalSobel, horizontalSobel);
    return 0;
}