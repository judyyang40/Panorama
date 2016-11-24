#ifndef image_stitching_H
#define image_stitching_H

#include "feature.h"

class stitIn{
public:
    CvMat* affineMat;
    double delta_x;
    double delta_y;
    stitIn(){affineMat = cvCreateMat(2, 3, CV_32F);}

};

stitIn* stitch_pic(matchSet &matchset);
IplImage* drift(int width, int height, CvMat* final_affine, IplImage* image);

#endif
