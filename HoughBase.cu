/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "pgm.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

# define M_PI 3.14159265358979323846
const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

//*****************************************************************

#define CUDA_CHECK_RETURN(value) {           \
    cudaError_t _m_cudaStat = value;         \
    if (_m_cudaStat != cudaSuccess) {        \
         fprintf(stderr, "Error %s at line %d in file %s\n",              \
                 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
         exit(1);                                                         \
       } }


// The CPU function returns a pointer to the accumulator
void CPU_HoughTran(unsigned char* pic, int w, int h, int** acc)
{
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            int idx = j * w + i;
            if (pic[idx] > 0)
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                float theta = 0;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++)
                {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
    }
}

//*****************************************************************
__global__ void GPU_HoughTran(unsigned char* pic, int w, int h, int* acc, float rMax, float rScale, const float* d_Cos, const float* d_Sin)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h)
        return; // in case of extra threads in block

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
        }
    }
}

constexpr int top_size = 10;
void paintHough(unsigned char* pixels, int w, int h, int** results) 
{
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    int top[top_size] = { -1 };

    for (int i = 0; i < w; i++) //por cada pixel
        for (int j = 0; j < h; j++) //...
        {
            int idx = j * w + i;
            if (pixels[idx] > 0) //si pasa thresh, entonces lo marca
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;  // y-coord has to be reversed
                float theta = 0;         // actual angle
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
                {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    int current = (*results)[rIdx * degreeBins + tIdx]; 

                    //insertar en el top10
                     for (int topG = 0; topG < top_size; topG++)
                    {
                        if (top[topG] == current)
                            break;
                        if (top[topG] < current)
                        {
                            for (int rep = 9; rep > topG; rep--)
                                top[rep] = top[rep - 1];

                            top[topG] = current;
                            break;
                        }
                    }
                    theta += radInc;
                }
            }
        }

    for (int i = 0; i < w; i++) //por cada pixel
        for (int j = 0; j < h; j++) //...
        {
            int idx = j * w + i;

            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
            {
                float r = xCoord * cos(theta) + yCoord * sin(theta);
                int rIdx = (r + rMax) / rScale;
                int current = (*results)[rIdx * degreeBins + tIdx];
            
                //insertar en el top10
                for (int topG = 0; topG < top_size; topG++)
                    if (top[topG] == current)
                        pixels[idx] = pixels[idx] + 100 > 255 ? 255: pixels[idx] + 100;
                    
                theta += radInc;
            }
            
        }
}

//*****************************************************************
int main(int argc, char** argv)
{
    int i;

    PGMImage inImg(argv[1]);

    int* cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    cudaEvent_t start, stop;
    float elapsedTime;

    // CPU calculation
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    // Pre-compute values to be stored
    float* pcCos = (float*)malloc(sizeof(float) * degreeBins);
    float* pcSin = (float*)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (i = 0; i < degreeBins; i++)
    {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // Copy cosine and sine arrays to device constant memory
    float* d_Cos;
    float* d_Sin;
    cudaMalloc((void**)&d_Cos, sizeof(float) * degreeBins);
    cudaMalloc((void**)&d_Sin, sizeof(float) * degreeBins);
    cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

    // Setup and copy data from host to device
    unsigned char* d_in, * h_in;
    int* d_hough, * h_hough;

    h_in = inImg.pixels;

    h_hough = (int*)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void**)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void**)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // Execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    int blockNum = ceil(w * h / 256);

    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time: %.5f s\n", elapsedTime);

    // Get results from device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    for (i = 0; i < degreeBins * rBins; i++)
    {
        if (cpuht[i] != h_hough[i])
            printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }

    printf("Done!\n");

    paintHough(inImg.pixels, w, h, &h_hough);

    inImg.write("output.pgm");

    // Clean up
    delete[] cpuht;
    free(pcCos);
    free(pcSin);
    free(h_hough);
    cudaFree(d_in);
    cudaFree(d_hough);
    cudaFree(d_Cos);
    cudaFree(d_Sin);

    return 0;
}
