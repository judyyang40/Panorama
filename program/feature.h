#ifndef feature_H
#define feature_H

class keyPoint{
public:
	keyPoint(double ax = 0.0, double ay = 0.0, int as = 0, double aa = 0.0, float avalue = 0.0, float *des = NULL) 
		: x(ax), y(ay), s(as), angle(aa), value(avalue), descriptor(des) {	}
	double x, y, angle;
	int s;
	float value, wave[3], *descriptor;

	bool operator< (const keyPoint &point) const {
		return (value < point.value);
	}
	float operator- (const keyPoint &point) const;
	double dis(const keyPoint &point) const;
};

class keyPointSet{
public:
	keyPoint *point;
	int size;

	keyPointSet(){	
		size = 0;
		point = new keyPoint[50000];	
	}

	keyPoint& operator[](int x)	{return point[x];}
	bool add(const keyPoint &point);
	bool add(const double &x, const double &y, const int &s, const double &angle, const float &value);
    void sort();
	keyPointSet supress();
};

class newImage{ 
public:
	IplImage *image_original, *image[5], *FHM[5];
	CvSize size[5];

    keyPointSet getKeyPointSet();
	bool load_image(char *name, double *focal_len);
	void subpixacc(keyPointSet result);
	void make_descriptor(keyPointSet result);
};

class matchSet{
public:
	matchSet(){size = 0;}
	keyPoint pair[1000][2];
	int size;
	
    void add_pair(const keyPoint& x, const keyPoint &y){
		pair[size][0] = x;
		pair[size][1] = y;
		size++;
	}
};

void feature_match(keyPointSet &set, keyPointSet &set2, matchSet &result);

#endif
