#include <iostream>
#include <cmath>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include "feature.h"

using namespace std;

bool newImage::load_image(char *name, double *focal_len){
    image_original = cvLoadImage(name, CV_LOAD_IMAGE_GRAYSCALE); //load pic in grayscale

    //cylindrical projection
    double width = (double)image_original->width/2, height = (double)image_original->height/2;
    double x = focal_len[0] * atan(width/focal_len[0]), y = height;
    unsigned char* p1;
    unsigned char* p2;

    IplImage* project = cvCreateImage(cvSize((int)x*2, (int)y*2), IPL_DEPTH_8U, 1);

    for(int j = 0; j<image_original->height; j++){
        for(int i = 0; i<image_original->width; i++){
            int yy = (int)(focal_len[0]*(j-height)/sqrt((i-width)*(i-width)+focal_len[0]*focal_len[0])+y);
            int xx = (int)(focal_len[0]*atan((i-width)/focal_len[0])+x);

            p1 = &CV_IMAGE_ELEM(image_original, unsigned char, j, i);
            p2 = &CV_IMAGE_ELEM(project, unsigned char, yy, xx); 
            p2[0] = p1[0];
        }
    }
    image[0] = project;
    if(image[0] == 0)
        return false;

    size[0] = cvGetSize(image[0]);
    IplImage* temp;

    for(int i = 1; i < 5; i++){ //smooth and resize into every scale
        size[i] = cvSize(size[i-1].width/2, size[i-1].height/2);
        if(size[i].height == 0 || size[i].width == 0)
            continue;

        temp = cvCreateImage(size[i-1], IPL_DEPTH_8U, 1);
        cvSmooth(image[i-1], temp, CV_GAUSSIAN, 0, 0, 1.0); //param = 1.0
        image[i] = cvCreateImage(size[i], IPL_DEPTH_8U, 1);
        cvResize(temp, image[i], CV_INTER_CUBIC);
        cvReleaseImage(&temp);
    }
    return true;
}

keyPointSet newImage::getKeyPointSet(){
    keyPointSet out;

    for(int s = 4 ; s >= 0 ; s--){
        if(size[s].height==0 || size[s].width==0) break;

        IplImage *grad[2] , *blur[2] , *show[2];

        for(int i = 0; i < 2; i++){
            grad[i] = cvCreateImage(cvGetSize(image[s]), IPL_DEPTH_32F, 1);
            blur[i] = cvCreateImage(cvGetSize(image[s]), IPL_DEPTH_32F, 1);
            show[i] = cvCreateImage(cvGetSize(image[s]), IPL_DEPTH_8U, 1);
        }

        IplImage* src = cvCreateImage(cvGetSize(image[s]), IPL_DEPTH_32F, 1);
        cvConvertScale(image[s], src, 1.0);

        float data[2] = { -1 , 1 };
        CvMat xd = cvMat(1, 2, CV_32FC1, data);
        CvMat yd = cvMat(2, 1, CV_32FC1, data);
        
        //get the gradient map
        cvFilter2D(src, grad[0] , &xd, cvPoint(0,0));
        cvFilter2D(src, grad[1] , &yd, cvPoint(0,0));

        //smooth
        cvSmooth(grad[0], blur[0], CV_GAUSSIAN, 0, 0, 1.0);
        cvSmooth(grad[1], blur[1], CV_GAUSSIAN, 0, 0, 1.0);

        cvConvertScaleAbs(blur[0] ,show[0]);
        cvConvertScaleAbs(blur[1] ,show[1]);

        IplImage *M[3];
        for(int i = 0; i < 3; i++)
            M[i] = cvCreateImage(size[s], IPL_DEPTH_32F, 1);

        for(int y = 0; y < size[s].height; y++){
            for(int x = 0; x < size[s].width; x++){
                float a = CV_IMAGE_ELEM(blur[0], float, y, x);
                float b = CV_IMAGE_ELEM(blur[1], float, y, x);
                CV_IMAGE_ELEM(M[0], float, y, x) = a*a;
                CV_IMAGE_ELEM(M[1], float, y, x) = a*b;
                CV_IMAGE_ELEM(M[2], float, y, x) = b*b;
            }
        }
        
        //smooth
        IplImage *H[3];
        for(int i = 0; i < 3; i++){
            H[i] = cvCreateImage(size[s], IPL_DEPTH_32F, 1);
            cvSmooth(M[i], H[i], CV_GAUSSIAN, 0, 0, 1.5); //param = 1.5
        }

        //calculate tra, det, and FHM
        FHM[s] = cvCreateImage(size[s], IPL_DEPTH_32F, 1);
        for(int y = 1; y < size[s].height-1; y++){
            for(int x = 1; x < size[s].width-1; x++){
                float T = CV_IMAGE_ELEM(H[0], float, y, x) + CV_IMAGE_ELEM(H[2], float, y, x);
                float D = CV_IMAGE_ELEM(H[0], float, y, x) * CV_IMAGE_ELEM(H[2], float, y, x) - CV_IMAGE_ELEM(H[1], float, y, x) * CV_IMAGE_ELEM(H[1], float, y, x);
                CV_IMAGE_ELEM(FHM[s], float, y, x) = D / (T+0.00001);
            }
        }
        
        //smooth
        cvSmooth(grad[0], blur[0],CV_GAUSSIAN, 0, 0, 4.5);
        cvSmooth(grad[1], blur[1],CV_GAUSSIAN, 0, 0, 4.5); //param = 4.5

        //search for the corner
        for(int y = 0; y < size[s].height; y++){
            for(int x = 0; x < size[s].width; x++){
                if(CV_IMAGE_ELEM(FHM[s], float, y, x) < 10.0) //threshold = 10
                    continue;

                bool max = true;
                for(int Y = -1; Y <= 1 && max; Y++){
                    for(int X = -1; X <= 1; X++){
                        if((X==0 && Y==0) || (y+Y >= size[s].height) || (y+Y<0) || (x+X >= size[s].width) || (x+X<0))
                            continue;

                        if(CV_IMAGE_ELEM(FHM[s], float, y, x) <= CV_IMAGE_ELEM(FHM[s], float, y+Y, x+X)){
                            max = false;
                            break;
                        }
                    }
                }

                if(max)
                    out.add(x, y, s, atan2(CV_IMAGE_ELEM(blur[1], float, y, x), CV_IMAGE_ELEM(blur[0], float, y, x)), CV_IMAGE_ELEM(FHM[s], float, y, x));
            }
        }

    }
    out.sort(); //sort for FHM from largest to smallest
    return out;
}

bool keypointcompare(const keyPoint &a, const keyPoint &b){
    return (b < a);
}

void keyPointSet::sort(){
    std::sort(point, point+size, keypointcompare);
}

bool keyPointSet::add(const keyPoint &point){
    return add(point.x, point.y, point.s, point.angle, point.value);
}

bool keyPointSet::add(const double &x, const double &y, const int &s, const double &angle, const float &value){
    if(size == 50000)
        return false;

    point[size].x = x;
    point[size].y = y;
    point[size].s = s;
    point[size].angle = angle;
    point[size].value = value;
    size++;
    return true;
}

double keyPoint::dis(const keyPoint &point) const {
    return ((x-point.x)*(x-point.x) + (y-point.y)*(y-point.y));
}

float keyPoint::operator- (const keyPoint &point) const{
    for(int i = 0; i < 3; i++)
        if(fabs(wave[i] - point.wave[i]) > 16.0)
            return 1000.0;

    float answer = 0;
    for(int i = 0; i < 64; i++)
        answer += fabs(descriptor[i] - point.descriptor[i]);
    return answer;
}

keyPointSet keyPointSet::supress(){
    keyPointSet result;
    bool *supressed = new bool[size];
    double square = 10000.0;
    
    if(size < 500) 
        return *this;
    
    while(result.size < 500){
        result.size = 0;
        memset(supressed, 0, size * sizeof(bool));
        for(int i=0; i<size; i++){
            if(supressed[i]) continue;
            for(int j=i+1; j<size; j++){
                if(point[i].s == point[j].s && point[i].dis(point[j]) < square)
                    supressed[j] = true;
            }
            result.add(point[i]);
        }
        square = square*0.8;
    }
    return result;
}

void newImage::subpixacc(keyPointSet result){
    for(int i=0; i<result.size; i++){
        keyPoint p = result[i];
        IplImage* fimage = FHM[result[i].s];
        float F[3][3], dF[2], dF2[4], delta[2];
        int m, n;

        for(int j=0; j<3; j++){
            for(int k=0; k<3; k++){
                m=p.x-1+j;
                n=p.y-1+k;
                F[j][k] = CV_IMAGE_ELEM(fimage, float, n, m);
            }
        }

        dF[0] = (F[2][1] - F[0][1])/2.0;
        dF[1] = (F[1][2] - F[1][0])/2.0;
        dF2[0] = F[2][1]-2*F[1][1]+F[0][1];
        dF2[1] = (F[0][0]+F[2][2]-F[0][2]-F[2][0])/4.0;
        dF2[2] = (F[0][0]+F[2][2]-F[0][2]-F[2][0])/4.0;
        dF2[3] = F[1][2]-2*F[1][1]+F[1][0];

        CvMat dFmat = cvMat(2, 1, CV_32F, dF);
        CvMat dF2mat = cvMat(2, 2, CV_32F, dF2);
        CvMat deltamat = cvMat(2, 1, CV_32F, delta);
        CvMat *Inmat = cvCreateMat(2, 2, CV_32F);
        cvInvert(&dF2mat, Inmat);
        cvGEMM(Inmat, &dFmat, -1.0, 0, 1.0, &deltamat);

        result[i].x += delta[0];
        result[i].y += delta[1];
    }
}

void newImage::make_descriptor(keyPointSet result){
    int i, a, b;
    IplImage *temp;
    IplImage *sample = cvCreateImage(cvSize(40, 40), IPL_DEPTH_8U, 1);
    IplImage *patch = cvCreateImage(cvSize(8, 8), IPL_DEPTH_8U, 1);
    CvMat *tMat = cvCreateMat(2, 3, CV_32F);

    for(i = 0; i < result.size; i++){
        float cos_theta = cos(result[i].angle), sin_theta = sin(result[i].angle);
        CV_MAT_ELEM(*tMat, float, 0, 0) = cos_theta;
        CV_MAT_ELEM(*tMat, float, 0, 1) = -sin_theta;
        CV_MAT_ELEM(*tMat, float, 1, 0) = sin_theta;
        CV_MAT_ELEM(*tMat, float, 1, 1) = cos_theta;
        CV_MAT_ELEM(*tMat, float, 0, 2) = result[i].x;
        CV_MAT_ELEM(*tMat, float, 1, 2) = result[i].y;

        cvGetQuadrangleSubPix(image[result[i].s], sample, tMat);
        cvResize(sample, patch, CV_INTER_AREA);
        result[i].descriptor = new float[64];

        float SUM = 0.0;
        for(a = 0; a < 8; a++){
            for(b = 0; b < 8; b++){
                result[i].descriptor[a*8+b] = CV_IMAGE_ELEM(patch, unsigned char, b, a);
                SUM += CV_IMAGE_ELEM(patch, unsigned char, b, a);
            }
        }

        float average = SUM/64.0;
        SUM = 0.0;
        for(a = 0; a < 64; a++){
            result[i].descriptor[a] -= average;
            SUM += result[i].descriptor[a] * result[i].descriptor[a];
        }

        float sigma = sqrt(SUM/64.0);
        for(a = 0; a < 64; a++)
            result[i].descriptor[a] /= sigma;

        result[i].wave[0] = 0.0;
        result[i].wave[1] = 0.0;
        result[i].wave[2] = 0.0;

        for(a = 0; a < 32; a++)
            result[i].wave[0] += result[i].descriptor[a];

        for(a = 32; a < 64; a++)
            result[i].wave[1] += result[i].descriptor[a];

        for(a = 0; a < 4; a++)
            for(b = 0; b < 4; b++)
                result[i].wave[2] += result[i].descriptor[a*8+b];

        for(a = 4; a < 8; a++)
            for(b = 4; b < 8; b++)
                result[i].wave[2] += result[i].descriptor[a*8+b];
    }
}

void feature_match(keyPointSet &set, keyPointSet &set2, matchSet &result){
    int *best = new int[set.size];
    int *best2 = new int[set2.size];
    float temp, min;
    int i, j, a;

    for(i = 0; i < set.size; i++){
        min = 20.0;
        best[i] = -1;

        for(j = 0; j < set2.size; j++){
            if(set[i].s == set2[j].s){
                temp = 0;
                for(a = 0; a < 3; a++){
                    if(fabs(set[i].wave[a]-set2[j].wave[a]) > 16.0){
                        temp = 20.0;
                        break;
                    }
                }
                if(temp < 20.0){
                    temp = 0;
                    for(a =0; a < 64 && temp < min; a++)
                        temp += fabs(set[i].descriptor[a] - set2[j].descriptor[a]);
                }
                if(temp < min){
                    min = temp;
                    best[i] = j;
                }
            }
        }
    }

    for(i = 0; i < set2.size; i++){
        min = 20.0;
        best2[i] = -1;

        for(j = 0; j < set.size; j++){
            if(set2[i].s == set[j].s){
                temp = 0;
                for(a = 0; a < 3; a++)
                    if(fabs(set2[i].wave[a] - set[j].wave[a]) > 16.0){
                        temp = 20.0;
                        break;
                    }
                if(temp < 20.0){
                    temp = 0;
                    for(a = 0; a < 64 && temp < min; a++)
                        temp += fabs(set2[i].descriptor[a] - set[j].descriptor[a]);
                }
                if(temp < min){
                    min = temp;
                    best2[i] = j;
                }
            }
        }
    }
    for(i = 0; i < set.size; i++){
        if(best[i] != -1 && best2[best[i]] == i){
            j = best[i];
            result.add_pair(set[i] , set2[j]);
        }
    }
}
