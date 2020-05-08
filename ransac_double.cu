#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h> 
#include "timerc.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define PHI 0x9e3779b9

// INT_MAX = 2147483647
// int ITERATIONS = 2147483647/16;
#define ITERATIONS 2048*2048
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

    // printf("x=%f | y=%f | A=%f | B=%f | C=%f | distance: %f \n", x, y, a, b, c ,d);

    return d;
}


/* 
* RETURNS: [A, B, C] for a line equation
*/
__host__ __device__ double *lineFromPoints(double *out, double x1, double y1, double x2, double y2) {
    out[0] = y1 - y2;
    out[1] = x2 - x1;
    out[2] = (x1-x2)*y1 + (y2-y1)*x1;

    // printing values to check correctness
    // for (int i=0; i < 3; i++) {
    //     printf("%f ", out[i]);
    // }
    // printf("\n");

    return out;
}


/*
* data – A set of observations.
* k – Maximum number of iterations allowed in the algorithm.
* t – threshold value to determine data points that are fit well by model.
* d – Number of close data points required to assert that a model fits well to data.
*/
__global__ void ransac_gpu(double *data, double *line,  int k, int t, int d, uint32_t seed){
    init_rand(seed);

    int r, inliers;
    int maxInliers = 0;

    double bestA, bestB, bestC, x1, y1, x2, y2, dist;
    
    for (int i=0; i < k; i++) {
        inliers = 0;

        /*******************
        CHOOSING RANDOM LINE
        *******************/

        // Choosing first random point
        r = randInRange(0, k*2, seed);
        // printf("r = %d\n", r);

        x1 = data[r];
        y1 = data[r+1];

        // Choosing second random point
        r = randInRange(0, k*2, seed);

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
            printf("yay");
    }

    if (bestA == -bestB) {
        bestA = 1;
        bestB = -1;
    }

    printf("GPU: A=%f | B=%f | C=%f \n", bestA, bestB, bestC);
}





__global__ void ransac_gpu_multi_thread(double *data, double *lineArr,  int k, int t, int d, uint32_t seed){
    init_rand(seed);

    int r, inliers;
    int maxInliers = 0;
    int scopeSize = k / THREADSPERBLOCK;
    int offset = 2 * threadIdx.x * scopeSize;

    double *shiftedData = &data[offset];

    // Test to see if data is offset
    // printf("first point is (%d, %d)\n", shiftedData[0], shiftedData[1]);

    double bestA, bestB, bestC, x1, y1, x2, y2, dist;

    double *line = &lineArr[threadIdx.x * 3];

    
    for (int i=0; i < scopeSize; i++) {
        inliers = 0;

        /*******************
        CHOOSING RANDOM LINE
        *******************/

        // Choosing first random point
        r = randInRange(0, scopeSize*2 - 1, seed);
        // printf("r = %d\n", r);

        x1 = shiftedData[r];
        y1 = shiftedData[r+1];

        // Choosing second random point
        r = randInRange(0, scopeSize*2 - 1 , seed);

        x2 = shiftedData[r];
        y2 = shiftedData[r+1];

        // Modeling a line between those two points
        line = lineFromPoints(line, x1, y1, x2, y2);

        // printf("A: %d, B:%d, C:%d\n", line[0], line[1], line[2]);

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

    if (bestA == -bestB) {
        bestA = 1;
        bestB = -1;
    }

    if (threadIdx.x % 256 == 0) {
        printf("GPU w/ Threads: A=%f | B=%f | C=%f \n", bestA, bestB, bestC);
    }
}





__global__ void ransac_gpu_stream(double *data, double *lineArr,  int k, int t, int d, uint32_t seed, int stream) {
    init_rand(seed);

    int r, inliers;
    int maxInliers = 0;
    int scopeSize = k / THREADSPERBLOCK / NUMSTREAMS;
    int offset = 2 * threadIdx.x * scopeSize;

    double *shiftedData = &data[offset];

    // Test to see if data is offset
    // printf("first point is (%d, %d)\n", shiftedData[0], shiftedData[1]);

    double bestA, bestB, bestC, x1, y1, x2, y2, dist;

    double *line = &lineArr[threadIdx.x * 3];
    
    for (int i=0; i < scopeSize; i++) {
        inliers = 0;

        /*******************
        CHOOSING RANDOM LINE
        *******************/

        // Choosing first random point
        r = randInRange(0, scopeSize*2 - 1, seed);
        // printf("r = %d\n", r);

        x1 = shiftedData[r];
        y1 = shiftedData[r+1];

        // Choosing second random point
        r = randInRange(0, scopeSize*2 - 1, seed);

        x2 = shiftedData[r];
        y2 = shiftedData[r+1];

        // Modeling a line between those two points
        line = lineFromPoints(line, x1, y1, x2, y2);

        // printf("A: %d, B:%d, C:%d\n", line[0], line[1], line[2]);

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

    if (bestA == -bestB) {
        bestA = 1;
        bestB = -1;
    }

    if (threadIdx.x == 0 && stream % 4 == 0) {
        printf("GPU w/ Streams: A=%f | B=%f | C=%f \n", bestA, bestB, bestC);
    }
}





void ransac(double *data, double *line, int k, int t, int d){
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

    if (bestA == -bestB) {
        bestA = 1;
        bestB = -1;
    }

    printf("CPU: A=%f | B=%f | C=%f \n", bestA, bestB, bestC);
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

    /*
    * Every two elements corresponds to x,y at time t.
    */
    double *points = (double *) malloc(2*ITERATIONS*sizeof(double));

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

    int pass = ITERATIONS / 2;




    float cpu_time;
    cstart();
    // ransac(points, line, ITERATIONS, THRESHOLD, pass);
    cend(&cpu_time);
    
    puts("***");

    

    double *d_points;
    // double *d_line;

    cudaMalloc((void **) &d_points, (2*ITERATIONS*sizeof(double)));
    cudaMemcpy(d_points, points, (2*ITERATIONS*sizeof(double)), cudaMemcpyHostToDevice);

    // cudaMalloc((void **) &d_line, (3*sizeof(double)));
    // cudaMemcpy(d_line, line, (3*sizeof(double)), cudaMemcpyHostToDevice);

    // float gpu_time;
    // gstart();
    // ransac_gpu<<<1,1>>>(d_points, d_line, ITERATIONS, THRESHOLD, pass, seed);
    // gend(&gpu_time);

    // puts("***");
    // cudaDeviceSynchronize();



    double *lineArr;
    double *d_lineArr;

    // Each thread will need it's only line equation container
    lineArr = (double *) malloc(3*THREADSPERBLOCK*sizeof(double));

    gpuErrchk( cudaMalloc((void **) &d_lineArr, (3*THREADSPERBLOCK*sizeof(double))) );
    gpuErrchk( cudaMemcpy(d_lineArr, lineArr, (3*THREADSPERBLOCK*sizeof(double)), cudaMemcpyHostToDevice) );

    float gpu_multi_thread_time;
    gstart();
    ransac_gpu_multi_thread<<<1,THREADSPERBLOCK>>>(d_points, d_lineArr, ITERATIONS, THRESHOLD, pass / THREADSPERBLOCK, seed);
    gend(&gpu_multi_thread_time);

    

    // gpuErrchk( cudaMemcpy(lineArr, d_lineArr, (3*THREADSPERBLOCK*sizeof(double)), cudaMemcpyDeviceToHost) );

    // for (int b=0; b<3*THREADSPERBLOCK; b=b+3) {
    //     printf("GPU: A=%f | B=%f | C=%f \n", lineArr[b], lineArr[b+1], lineArr[+2]);
    // }
    puts("***");
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));


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
        ransac_gpu_stream<<<1, THREADSPERBLOCK, 0, stream[i]>>>(&d_points[offset], &d_lineStreamArr[lineOffset], ITERATIONS, THRESHOLD, pass / THREADSPERBLOCK, seed, i);
    }
    gend(&gpu_stream_time);

    for (int i = 0; i < NUMSTREAMS; ++i)
        cudaStreamDestroy(stream[i]);

    puts("***\n");

    printf("CPU time: %f\n",cpu_time);
    // printf("GPU time: %f\n", gpu_time);
    printf("GPU w/ Multi-thread time: %f\n", gpu_multi_thread_time);
    printf("GPU w/ Streams time: %f\n", gpu_stream_time);



    // puts("\n***\n\nStreams Test");
    // for (int s=1; s <=16; s++) {
    //     nStreams = s;
    //     streamSize = ITERATIONS / nStreams;
    //     streamBytes = 2 * streamSize * sizeof(int);

    //     cudaStream_t stream[nStreams];

    //     for (int i = 0; i < nStreams; ++i)
    //         cudaStreamCreate(&stream[i]);

    //     float gpu_stream_time;
    //     gstart();
    //     for (int i=0; i < nStreams; i++) {
    //         int offset = 2 * i * streamSize;
    //         cudaMemcpyAsync(&d_points[offset], &points[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    //         ransac_gpu_stream<<<1, 1, 0, stream[i]>>>(&d_points[offset], streamSize, THRESHOLD, PASS, seed, offset);
    //     }
    //     gend(&gpu_stream_time);

    //     printf("%d Streams, time: %f\n", nStreams, gpu_stream_time);
    // }


    cudaDeviceSynchronize();

    cudaFree(d_points);
    cudaFree(d_lineArr);
    cudaFree(d_lineStreamArr);
    free(points);
    free(line);
    free(lineArr);
    free(lineStreamArr);

    return 0;
}