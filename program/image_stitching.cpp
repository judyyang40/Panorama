#include <iostream>
#include <cmath>
#include <time.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include "image_stitching.h"

using namespace std;

//RANSAC
stitIn* stitch_pic(matchSet &matchset){
    stitIn* stitching = new stitIn();
    int i, j, pair = 0, count = 0;

    double** affineMat1 = new double*[matchset.size];
    double** affineMat2 = new double*[matchset.size];
    
    for(i = 0; i < matchset.size; i++){
        affineMat1[i] = new double[2];
        affineMat2[i] = new double[2];
    }

    for(i = 0; i < matchset.size; i++){
        keyPoint point1 = matchset.pair[i][0];
        keyPoint point2 = matchset.pair[i][1];
        affineMat1[i][0] = point1.x;
        affineMat1[i][1] = point1.y;
        affineMat2[i][0] = point2.x;
        affineMat2[i][1] = point2.y;
    }
    srand(time(NULL));

    for(i = 0; i < 400; i++){
        int a, b, c;
        a = b = c = rand()%matchset.size;
        
        while(b == a)
            b = rand()%matchset.size;
        while(c == b || c == a)
            c = rand()%matchset.size;

        CvPoint2D32f* first = new CvPoint2D32f[3];
        first[0] = cvPoint2D32f(affineMat2[a][0], affineMat2[a][1]);
        first[1] = cvPoint2D32f(affineMat2[b][0], affineMat2[b][1]);
        first[2] = cvPoint2D32f(affineMat2[c][0], affineMat2[c][1]);
        
        CvPoint2D32f* second = new CvPoint2D32f[3];
        second[0] = cvPoint2D32f(affineMat1[a][0], affineMat1[a][1]);
        second[1] = cvPoint2D32f(affineMat1[b][0], affineMat1[b][1]);
        second[2] = cvPoint2D32f(affineMat1[c][0], affineMat1[c][1]);

        CvMat* affineMat = cvCreateMat(2, 3, CV_32F);
        cvGetAffineTransform(first, second, affineMat);

        float rotation[4] = {CV_MAT_ELEM(*affineMat, float, 0, 0), CV_MAT_ELEM(*affineMat, float, 0, 1 ), CV_MAT_ELEM(*affineMat, float, 1, 0), CV_MAT_ELEM(*affineMat, float, 1, 1)};
        CvMat Rotate = cvMat(2, 2, CV_32F, rotation);

        float translation[2] = {CV_MAT_ELEM(*affineMat, float, 0, 2), CV_MAT_ELEM(*affineMat, float, 1, 2)};
        CvMat Trans = cvMat(2, 1, CV_32F, translation);
        
        count = 0;
        for(j = 0; j < matchset.size; j++){
            float A[2] = {affineMat2[j][0], affineMat2[j][1]};
            float B[2] = {affineMat1[j][0], affineMat1[j][1]};
            
            CvMat AA = cvMat(2, 1, CV_32FC1, A);
            CvMat* BB = cvCreateMat(2, 1, CV_32F);
            cvMatMulAdd(&Rotate, &AA, &Trans, BB);

            float dx = CV_MAT_ELEM(*BB, float, 0, 0) - B[0];
            float dy = CV_MAT_ELEM(*BB, float, 1, 0) - B[1];
            float error = dx*dx + dy*dy;
            
            if(error < 50) 
                count++;
        }
        if(count > pair){
            pair = count;
            stitching->affineMat = affineMat;
            stitching->delta_x = translation[0];
            stitching->delta_y = translation[1];
        }
    }
    return stitching;
}

//drift and blending
IplImage* drift(int width, int height, CvMat* final_affine, IplImage* image){
    float x_y[3] = {width, 0.0, 1.0};
    CvMat XY = cvMat(3, 1, CV_32F, x_y);
    CvMat* new_xy = cvCreateMat(2, 1, CV_32F);
    cvMatMul(final_affine, &XY, new_xy);

    int new_width = sqrt(CV_MAT_ELEM(*new_xy, float, 0, 0)*CV_MAT_ELEM(*new_xy, float, 0, 0)+CV_MAT_ELEM(*new_xy, float, 1, 0)*CV_MAT_ELEM(*new_xy,float,1,0));
    IplImage* mapImage = cvCreateImage(cvSize(new_width, height), IPL_DEPTH_8U, 3);

    float x_y2[3] = {width, height, 1.0};
    CvMat XY2 = cvMat(3, 1, CV_32F, x_y2);
    CvMat* new_xy2 = cvCreateMat(2, 1, CV_32F);
    cvMatMul(final_affine, &XY2, new_xy2);

    CvPoint2D32f* first = new CvPoint2D32f[4];
    first[0] = cvPoint2D32f(0.0, 0.0);
    first[1] = cvPoint2D32f(0.0, height);
    first[2] = cvPoint2D32f(CV_MAT_ELEM(*new_xy, float, 0, 0),CV_MAT_ELEM(*new_xy, float, 1, 0));
    first[3] = cvPoint2D32f(CV_MAT_ELEM(*new_xy2, float ,0 ,0),CV_MAT_ELEM(*new_xy2, float, 1, 0));

    CvPoint2D32f* second = new CvPoint2D32f[4];
    second[0] = cvPoint2D32f(0.0, 0.0);
    second[1] = cvPoint2D32f(0.0, height);
    second[2] = cvPoint2D32f(new_width, 0.0);
    second[3] = cvPoint2D32f(new_width, height);

    CvMat* affineMat = cvCreateMat(3, 3, CV_32F);
    cvGetPerspectiveTransform(first, second, affineMat);
    cvWarpPerspective(image, mapImage, affineMat, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cvScalarAll(0));

    return mapImage;
}
