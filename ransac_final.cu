#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h> 
#include "timerc.h"

#define PHI 0x9e3779b9

// int ITERATIONS = INT_MAX/16;
#define ITERATIONS 1024*128
#define THRESHOLD 0.5

#define THREADSPERBLOCK 1024
#define NUMSTREAMS 16

__device__ static uint32_t Q[4096], c = 362436;

__device__ void init_rand(uint32_t x)
{
    int i;

    Q[0] = x;
    Q[1] = x + PHI;
    Q[2] = x + PHI + PHI;

    for (i = 3; i < 4096; i++)
            Q[i] = Q[i - 3] ^ Q[i - 2] ^ PHI ^ i;
}

__device__ uint32_t rand_cmwc(void)
{
    uint64_t t, a = 18782LL;
    static uint32_t i = 4095;
    uint32_t x, r = 0xfffffffe;
    i = (i + 1) & 4095;
    t = a * Q[i] + c;
    c = (t >> 32);
    x = t + c;
    if (x < c) {
            x++;
            c++;
    }
    return (Q[i] = r - x);
}

__device__ int randInRange(int min, int max, uint32_t seed) {
    return min + rand_cmwc() % (max + 1 - min);
}


/* 
* RETURNS: d, distance from  point p to the line Ax + By = C
*/
__host__ __device__ double distanceFromLine(double x, double y, double a, double b, double c) {
    double d = abs((a * x + b * y + c)) / (sqrt(a * a + b * b)); 

    return d;
}


/* 
* RETURNS: [A, B, C] for a line equation
*/
__host__ __device__ double *lineFromPoints(double *out, double x1, double y1, double x2, double y2) {
    out[0] = y1 - y2;
    out[1] = x2 - x1;
    out[2] = (x1-x2)*y1 + (y2-y1)*x1;

    return out;
}


/*
* data – A set of observations.
* lineArr - Container for optimal model parameters outputted by the algorithm
* k – Maximum number of iterations allowed in the algorithm.
* t – threshold value to determine data points that are fit well by model.
* d – Number of close data points required to assert that a model fits well to data.
* seed - Random seed for a RNG on device
* numStreams - Number of streams running this function. Set to 1 for testing multi-thread performance
* stream - Index of the current stream used to offset data and lineArr. Used for debugging
*/
__global__ void ransac_gpu_optimal(double *data, double *lineArr,  int k, int t, int d, uint32_t seed, int numStreams, int stream) {
    init_rand(seed);

    int r, inliers;
    int maxInliers = 0;
    int scopeSize = k / THREADSPERBLOCK / numStreams;
    int offset = 2 * threadIdx.x * scopeSize;

    double bestA, bestB, bestC, x1, y1, x2, y2, dist;

    double *shiftedData = &data[offset];
    double *line = &lineArr[threadIdx.x * 3];
    
    for (int i=0; i < scopeSize; i++) {
        inliers = 0;

        /*******************
        CHOOSING RANDOM LINE
        *******************/

        // Choosing first random point
        r = randInRange(0, scopeSize*2 - 1, seed);

        x1 = shiftedData[r];
        y1 = shiftedData[r+1];

        // Choosing second random point
        r = randInRange(0, scopeSize*2 - 1, seed);

        x2 = shiftedData[r];
        y2 = shiftedData[r+1];

        // Modeling a line between those two points
        line = lineFromPoints(line, x1, y1, x2, y2);

        /***********************
        FINDING INLIERS FOR LINE
        ***********************/
        for (int j=0; j < scopeSize*2; j=j+2) {
            x1 = shiftedData[j];
            y1 = shiftedData[j + 1];
            dist = distanceFromLine(x1, y1, line[0], line[1], line[2]);
            if (dist <= t) {
                inliers++;
            }
        }
        
        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestA = line[0];
            bestB = line[1];
            bestC = line[2];

        }

        if (maxInliers >= ( d / THREADSPERBLOCK)) {
            break;
        }
    }

    // Some reduction
    if (bestA == -bestB) {
        bestA = 1;
        bestB = -1;
    }

    lineArr[threadIdx.x * 3] = bestA;
    lineArr[threadIdx.x * 3 + 1] = bestB;
    lineArr[threadIdx.x * 3 + 2] = bestC;

    // Print out only some of the output to check correctness
    // if (threadIdx.x == 0 && stream % 4 == 0) {
    //     printf("GPU w/ Streams: A=%f | B=%f | C=%f \n", bestA, bestB, bestC);
    // }
}





void ransac_cpu(double *data, double *line, int k, int t, int d){
    srand(time(NULL)); 

    int r, inliers;
    int maxInliers = 0;

    double bestA, bestB, bestC, x1, y1, x2, y2, dist;
    
    for (int i=0; i < k; i++) {
        inliers = 0;

        /*******************
        CHOOSING RANDOM LINE
        *******************/

        // Choosing first random point
        r = 1 + rand() % k;

        x1 = data[r];
        y1 = data[r+1];

        // Choosing second random point
        r = 1 + rand() % k;

        x2 = data[r];
        y2 = data[r+1];

        // Modeling a line between those two points
        line = lineFromPoints(line, x1, y1, x2, y2);


        /***********************
        FINDING INLIERS FOR LINE
        ***********************/
        for (int j=0; j < k; j=j+2) {
            x1 = data[j*2];
            y1 = data[j*2 + 1];
            dist = distanceFromLine(x1, y1, line[0], line[1], line[2]);
            if (dist <= t) {
                inliers++;
            }
        }
        
        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestA = line[0];
            bestB = line[1];
            bestC = line[2];
        }

        if (maxInliers >= d) 
            break;
    }

    // Some reduction
    if (bestA == -bestB) {
        bestA = 1;
        bestB = -1;
    }

    line[0] = bestA;
    line[1] = bestB;
    line[2] = bestC;
}



// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------



int main() {
    uint32_t seed = time(NULL);

    srand(seed); 
    int r;

    int pass = ITERATIONS / 2;

    /*
    * Every two elements corresponds to x,y at time t.
    */
    double *points = (double *) malloc(2*ITERATIONS*sizeof(double));
    double *d_points;

    // Move points with velocity (vx, vy)
    double vx = 100.0;
    double vy = 100.0;


    for (int j=0; j < ITERATIONS; j++) {
        if (j % 10 == 0) {
            r = 0 + rand() % ITERATIONS;
            points[j*2] = r;
            r = 0 + rand() % ITERATIONS;
            points[j*2+1] = r;
        } else {
            points[j*2] = j-1 + vx;
            points[j*2+1] = j-1 + vy; 
        }
    }
    
    // Shell to be used for outputting results in the form of line equation
    double *line = (double *) malloc(3*sizeof(double));

    // Copy points to file
    FILE *fp;
    fp = fopen("p.txt", "w+");
    
    for (int i=0; i < 2*ITERATIONS; i++) {
        fprintf(fp,"%f ", points[i]);
    }
    fclose(fp);



    float cpu_time;
    cstart();
    ransac_cpu(points, line, ITERATIONS, THRESHOLD, pass);
    cend(&cpu_time);

    printf("CPU: A=%f | B=%f | C=%f \n", line[0], line[1], line[2]);
    puts("***");

    

    

    cudaMalloc((void **) &d_points, (2*ITERATIONS*sizeof(double)));
    cudaMemcpy(d_points, points, (2*ITERATIONS*sizeof(double)), cudaMemcpyHostToDevice);

    double *lineArr;
    double *d_lineArr;

    // Each thread will need it's own line equation container
    lineArr = (double *) malloc(3*THREADSPERBLOCK*sizeof(double));

    cudaMalloc((void **) &d_lineArr, (3*THREADSPERBLOCK*sizeof(double)));
    cudaMemcpy(d_lineArr, lineArr, (3*THREADSPERBLOCK*sizeof(double)), cudaMemcpyHostToDevice);

    float gpu_multi_thread_time;
    gstart();
    ransac_gpu_optimal<<<1,THREADSPERBLOCK>>>(d_points, d_lineArr, ITERATIONS, THRESHOLD, pass / THREADSPERBLOCK, seed, 1, 0);
    gend(&gpu_multi_thread_time);

    cudaMemcpy(lineArr, d_lineArr, (3*THREADSPERBLOCK*sizeof(double)), cudaMemcpyDeviceToHost);


    double avgA = 0;
    double avgB = 0;
    double avgC = 0;
    for (int i=0; i<3*THREADSPERBLOCK; i=i+3) {
        avgA = avgA + lineArr[i];
        avgB = avgB + lineArr[i+1];
        avgC = avgC + lineArr[i+2];
    }
    avgA = avgA / THREADSPERBLOCK;
    avgB = avgB / THREADSPERBLOCK;
    avgC = avgC / THREADSPERBLOCK;

    printf("GPU w/Threads: A=%f | B=%f | C=%f \n", avgA, avgB, avgC);

    puts("***");
    cudaDeviceSynchronize();





    double *lineStreamArr;
    double *d_lineStreamArr;

    // Each thread will need it's only line equation container
    lineStreamArr = (double *) malloc(3*NUMSTREAMS*THREADSPERBLOCK*sizeof(double));
    cudaMalloc((void **) &d_lineStreamArr, (3*NUMSTREAMS*THREADSPERBLOCK*sizeof(double)));

    int streamSize = (2 * ITERATIONS) / NUMSTREAMS;
    int streamBytes = streamSize * sizeof(double);

    cudaStream_t stream[NUMSTREAMS];

    for (int i = 0; i < NUMSTREAMS; ++i)
        cudaStreamCreate(&stream[i]);

    float gpu_stream_time;
    gstart();
    for (int i=0; i < NUMSTREAMS; i++) {
        int offset = i * streamSize;
        int lineOffset = 3 * i * THREADSPERBLOCK;
        cudaMemcpyAsync(&d_points[offset], &points[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_lineStreamArr[lineOffset], &lineStreamArr[lineOffset], 3*THREADSPERBLOCK*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
        ransac_gpu_optimal<<<1, THREADSPERBLOCK, 0, stream[i]>>>(&d_points[offset], &d_lineStreamArr[lineOffset], ITERATIONS, THRESHOLD, pass / THREADSPERBLOCK, seed, NUMSTREAMS, i);
    }
    gend(&gpu_stream_time);

    cudaDeviceSynchronize();

    for (int i = 0; i < NUMSTREAMS; ++i)
        cudaStreamDestroy(stream[i]);

    cudaMemcpy(lineStreamArr, d_lineStreamArr, (3*NUMSTREAMS*THREADSPERBLOCK*sizeof(double)), cudaMemcpyDeviceToHost);

    avgA = 0;
    avgB = 0;
    avgC = 0;
    for (int i=0; i<3*NUMSTREAMS*THREADSPERBLOCK; i=i+3) {
        avgA = avgA + lineStreamArr[i];
        avgB = avgB + lineStreamArr[i+1];
        avgC = avgC + lineStreamArr[i+2];
    }
    avgA = avgA / THREADSPERBLOCK / NUMSTREAMS;
    avgB = avgB / THREADSPERBLOCK / NUMSTREAMS;
    avgC = avgC / THREADSPERBLOCK / NUMSTREAMS;

    printf("GPU w/Streams: A=%f | B=%f | C=%f \n", avgA, avgB, avgC);

    // for (int b=0; b<9; b=b+3) {
    //     printf("GPU w/Streams: A=%f | B=%f | C=%f \n", lineStreamArr[b], lineStreamArr[b+1], lineStreamArr[b+2]);
    // }
    puts("***\n");
    cudaDeviceSynchronize();




    printf("CPU time: %f\n",cpu_time);
    printf("GPU w/ Multi-thread time: %f\n", gpu_multi_thread_time);
    printf("GPU w/ Streams time: %f\n", gpu_stream_time);



    cudaFree(d_points);
    cudaFree(d_lineArr);
    cudaFree(d_lineStreamArr);
    free(points);
    free(line);
    free(lineArr);
    free(lineStreamArr);

    return 0;
}