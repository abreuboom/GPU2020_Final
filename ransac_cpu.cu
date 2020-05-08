#include <stdio.h>
#include <stdlib.h> 
#include<math.h> 
#include "timerc.h"

#define W 100
#define H 100
#define NUM_POINTS 300
#define ITERATIONS 100

void printChild(int x) {
    if (x==2) {
        printf("O ");
    } else if (x==1) {
        printf("* ");
    } else {
        printf(". ");
    }
}

void printBoard(int *a) {
    for (int i=0; i < W; i++) {
        for (int j=0; j < H; j++) {
            printChild(a[i*W + j]);
        }
        printf("\n");
    }
}

void printToFile(int *a) {
    FILE *fp;
    fp = fopen("space.txt", "w+");
    for (int i=0; i < W; i++) {
        for (int j=0; j < H; j++) {
            // char *num = a[i*W + j] + " ";
            fprintf(fp,"%d ", a[i*W + j]);
        }
        fputs("\n", fp);
    }
    fclose(fp);
}

/* 
* RETURNS: d, distance from  point p to the line Ax + By = C
*/
double distanceFromLine(double *p, double a, double b, double c) {
    double x = p[0];
    double y = p[1];

    double d = abs((a * x + b * y + c)) / (sqrt(a * a + b * b)); 

    // printf("x=%f | y=%f | A=%f | B=%f | C=%f | distance: %f \n", x, y, a, b, c ,d);

    return d;
}


/* 
* RETURNS: [A, B, C] for a line equation
*/
double *lineFromPoints(double x1, double y1, double x2, double y2) {
    double *out = (double *) malloc(3*sizeof(double));

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
* RETURNS: [x, y] to represent point n in 2D space
*/
double *convertIntToPoint(int n) {
    double *p = (double *) malloc(2*sizeof(double));

    // x, y
    p[0] = (double) (n % W);
    p[1] = (double) (n / W);

    return p;
}




// ------------------------------------------------------
// ------------------------------------------------------
// ------------------------------------------------------


int main() {
    srand(time(NULL)); 
    int r;

    int *space = (int *) malloc(W*H*sizeof(int));
    int *points = (int *) malloc(NUM_POINTS*sizeof(int));


    // Create empty sample sapce
    for (int i=0; i < W*H; i++) 
        space[i] = 0;

    // Generating random points
    for (int i=0; i < NUM_POINTS; i++)  {
        r = 2500 + rand() % 3000;
        space[r] = 1;
        points[i] = r;
    }
    printBoard(space);

    FILE *fp;
    fp = fopen("points.txt", "w+");
    
    for (int i=0; i < NUM_POINTS; i++) {
        fprintf(fp,"%d ", points[i]);
    }
    fclose(fp);

    

    double *p;
    double *temp;
    double *line;
    
    double x1, y1, x2, y2, dist;
    double thres = 0.5;
    

    double bestA, bestB, bestC, maxInliers, inliers;
    maxInliers = 0;

    for (int i=0; i < ITERATIONS; i++) {
        inliers = 0;

        /*******************
        CHOOSING RANDOM LINE
        *******************/

        // Choosing first random point
        r = 1 + rand() % (NUM_POINTS+1);
        p = convertIntToPoint(points[r]);

        x1 = p[0];
        y1 = p[1];

        // Choosing second random point
        r = 1 + rand() % (NUM_POINTS+1);
        p = convertIntToPoint(points[r]);

        x2 = p[0];
        y2 = p[1];

        // printf("x1=%f | y1=%f | x2=%f | y2=%f \n", x1, y1, x2, y2);

        line = lineFromPoints(x1, y1, x2, y2);


        /***********************
        FINDING INLIERS FOR LINE
        ***********************/
        for (int j=0; j < NUM_POINTS; j++) { 
            temp = convertIntToPoint(points[j]);
            dist = distanceFromLine(temp, line[0], line[1], line[2]);
            if (dist <= thres) {
                inliers++;
            }
        }
        
        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestA = line[0];
            bestB = line[1];
            bestC = line[2];
        }
    }


    /****************
    DRAWING BEST LINE
    ****************/
    for (int j=0; j < NUM_POINTS; j++) { 
        temp = convertIntToPoint(points[j]);
        dist = distanceFromLine(temp, bestA, bestB, bestC);
        if (dist <= thres) {
            space[points[j]] = 2; 
        }
    }

    printBoard(space);


    printf("A=%f | B=%f | C=%f \n", bestA, bestB, bestC);
    printToFile(space);



    free(p);
    free(temp);
    free(line);
    free(space);
    free(points);
    
    return 0;
}