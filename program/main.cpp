#include <iostream>
#include <cmath>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include "feature.h"
#include "image_stitching.h"

using namespace std;

int main(){
    IplImage **image;
    newImage *picture;

    double *focal_len;
    int i, j, k, nPic, original, count;
    char **picName;

    scanf("%d", &nPic);
    picture = new newImage[nPic];
    image = new IplImage *[nPic];
    focal_len = new double [nPic];
    picName = new char *[nPic];

    for(i = 0; i < nPic; i++){
        picName[i] = new char[1024];
        scanf("%s %lf", picName[i], &focal_len[i]);
        printf("loading: %s\n", picName[i]);       
        picture[i].load_image(picName[i], &focal_len[i]);
        image[i] = cvLoadImage(picName[i]);
    }
    original = image[0]->width;

    //cylindrical projection
    for(i = 0; i < nPic; i++){
        double w = (double)image[i]->width/2, h = (double)image[i]->height/2;
        double m_x = focal_len[i] * atan(w/focal_len[i]), m_y = h;
        unsigned char *p1;
        unsigned char *p2;

        IplImage *projection = cvCreateImage(cvSize((int)m_x*2, (int)m_y*2), IPL_DEPTH_8U, 3);

        int x, y, num;
        for(j = 0; j < image[i]->height; j++){
            for(k = 0; k < image[i]->width; k++){
                x = (int)(focal_len[i] * atan((k-w)/focal_len[i]) + m_x);
                y = (int)(focal_len[i]*(j-h)/sqrt((k-w)*(k-w)+focal_len[i]*focal_len[i])+m_y);

                p1 = &CV_IMAGE_ELEM(image[i], unsigned char, j, k*3);
                p2 = &CV_IMAGE_ELEM(projection, unsigned char, y, x*3);
                
                for(num = 0; num < 3; num++)
                    p2[num] = p1[num];
            }
        }
    }

    //get key points
    printf("calculating key points\n");
    keyPointSet *feature_points = new keyPointSet[nPic];
    for(i = 0; i < nPic; i++)
        feature_points[i] = picture[i].getKeyPointSet();

    //non-maximal suppression
    printf("non-maximal suppression\n");
    keyPointSet *out = new keyPointSet[nPic];
    for(i = 0; i < nPic; i++)
        out[i] = feature_points[i].supress();

    //sub-pixel accuracy
    printf("sub-pixel accuracy\n");
    for(i = 0; i < nPic; i++)
        picture[i].subpixacc(out[i]);

    for(i = 0; i < nPic; i++)
        picture[i].make_descriptor(out[i]);

    //freature matching
    printf("feature matching\n");
    matchSet *match = new matchSet[nPic-1];
    for(i = 0; i < nPic-1; i++)
        feature_match(out[i] , out[i+1] , match[i]);

    //stitching
    printf("image stitching\n");
    stitIn **stitchingNow = new stitIn*[nPic-1];

    for(i = 0; i < nPic-1; i++){
        stitchingNow[i] = stitch_pic(match[i]);

        if(i != nPic-2){
            for(count = 0; count < match[i+1].size; count++){
                float xy[3] = {match[i+1].pair[count][0].x, match[i+1].pair[count][0].y, 1.0};
                CvMat XY = cvMat(3, 1, CV_32F, xy);

                CvMat *new_xy = cvCreateMat(2, 1, CV_32F);
                cvMatMul(stitchingNow[i]->affineMat, &XY, new_xy);
                match[i+1].pair[count][0].x = CV_MAT_ELEM(*new_xy, float, 0, 0);
                match[i+1].pair[count][0].y = CV_MAT_ELEM(*new_xy, float, 1, 0);
            }
        }
    }
    
    //output match points
    IplImage *image_match = cvCreateImage(cvSize(image[0]->width+image[1]->width, image[0]->height), IPL_DEPTH_8U, 3);
    for(i = 0; i < image_match->height; i++){
        for(count = 0; count < image_match->width; count++){
            if(count < image[0]->width){
                if(i <image[0]->height)
                    for(j = 0; j < 3; j++)
                        CV_IMAGE_ELEM(image_match, unsigned char, i, count*3+j) =  CV_IMAGE_ELEM(image[0], unsigned char, i, count*3+j) ;
            }
            else{
                if(i < image[0]->height)
                    for(j = 0; j < 3; j++)
                        CV_IMAGE_ELEM(image_match, unsigned char, i, count*3+j) =  CV_IMAGE_ELEM(image[1], unsigned char, i , (count-image[0]->width)*3+j) ;
            }
        }
    }

    for(i = 0; i < match[0].size; i++){
        count = (2*match[0].pair[i][0].s);
        cvCircle(image_match, cvPoint(match[0].pair[i][0].x*count, match[0].pair[i][0].y*count), 4*count, CV_RGB(0, 0, 255), 1);
        cvCircle(image_match, cvPoint(match[0].pair[i][1].x*count+image[1]->width, match[0].pair[i][1].y*count), 4*count, CV_RGB(0, 0, 255), 1);
        cvLine(image_match, cvPoint(match[0].pair[i][0].x*count, match[0].pair[i][0].y*count), cvPoint(match[0].pair[i][1].x*count+image[1]->width, match[0].pair[i][1].y*count), CV_RGB(255, 0, 0));
    }

    //output final picture
    IplImage *mapImage = cvCloneImage(image[0]);
    for(count = 0; count < nPic-1; count++){
        IplImage* mapImage2 = cvCloneImage(image[count+1]);
        double Dx = stitchingNow[count]->delta_x, Dy = stitchingNow[count]->delta_y;

        IplImage* newPic = cvCreateImage(cvSize(mapImage2->width+original+Dx, mapImage2->height+Dy), IPL_DEPTH_8U, 3);
        cvWarpAffine(mapImage2, newPic, stitchingNow[count]->affineMat, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
        mapImage2 = newPic;

        int b_x = max(mapImage->width, mapImage2->width), b_y = max(mapImage->height, mapImage2->height);
        IplImage* stitchImage = cvCreateImage(cvSize(b_x , b_y), IPL_DEPTH_8U, 3);

        for(j = 0; j < mapImage->height; j++)
            for(i = 0; i < mapImage->width; i++)
                for(k = 0; k < 3; k++)
                    CV_IMAGE_ELEM(stitchImage, unsigned char, j, i*3+k) =  CV_IMAGE_ELEM(mapImage, unsigned char, j, i*3+k);

        for(j = 0; j < mapImage2->height; j++){
            for(i = 0; i < mapImage2->width; i++){
                for(k = 0; k < 3; k++){
                    if(CV_IMAGE_ELEM(mapImage2, unsigned char, j , i*3+k))
                        CV_IMAGE_ELEM(stitchImage, unsigned char, j, i*3+k) =  CV_IMAGE_ELEM(mapImage2, unsigned char, j, i*3+k);
                }
            }
        }

        unsigned char *ptr;
        unsigned char ptr_m;
        int b_y2 = min(mapImage->height, mapImage2->height);
        for(j = 0; j < b_y2; j++){
            int a, b;
            for(a = 0; a < mapImage2->width; a++){
                ptr = &CV_IMAGE_ELEM(mapImage2, unsigned char, j, a*3);
                ptr_m = ptr[0]/3 + ptr[1]/3 + ptr[2]/3;
                if(ptr_m > 5) 
                    break;
            }
            for(b = mapImage->width-1; b >= 0; b--){
                ptr = &CV_IMAGE_ELEM(mapImage, unsigned char, j, b*3);
                ptr_m = ptr[0]/3 + ptr[1]/3 + ptr[2]/3;
                if(ptr_m > 5) 
                    break;
            }
            if(a < b){
                for(i = a; i <= b; i++){
                    double ratio = (double)(i-a)/(b-a+1);
                    for(k = 0; k < 3; k++)
                        CV_IMAGE_ELEM(stitchImage, unsigned char, j, i*3+k) = (unsigned char)(ratio*CV_IMAGE_ELEM(mapImage2,unsigned char, j, i*3+k) + ((1-ratio)*CV_IMAGE_ELEM(mapImage, unsigned char, j, i*3+k)));
                }
            }
        }
        mapImage = cvCloneImage(stitchImage);
    }

    //drift final picture
    printf("drifting\n");
    int trimming[4] = {0, 0, 20, 20};
    IplImage *driftImage = drift(image[0]->width, image[0]->height, stitchingNow[nPic-2]->affineMat, mapImage);
    IplImage *drift_final = cvCreateImage(cvSize(driftImage->width-trimming[2]-trimming[3], driftImage->height-trimming[0]-trimming[1]), IPL_DEPTH_8U, 3);

    for(j = 0; j < drift_final->height; j++)
        for(i = 0; i < drift_final->width; i++)
            for(k = 0; k < 3; k++)
                CV_IMAGE_ELEM(drift_final, unsigned char, j, i*3+k) =  CV_IMAGE_ELEM(driftImage, unsigned char, j+trimming[0], (i+trimming[2])*3+k);

    cvSaveImage("match_result.jpg", image_match);
    cvSaveImage("output.jpg" , drift_final); //output result
    printf("done\n");

    return 0;
}
