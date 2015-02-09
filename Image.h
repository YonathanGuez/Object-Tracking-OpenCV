/*Image .h */
#ifndef IMAGE_H
#define IMAGE_H
#include <stdlib.h>
#include<iostream>
#include "opencv2\highgui\highgui.hpp"//bibliotheque de videocapture,namedWindow,imshow,waitkey
#include "opencv2\video\background_segm.hpp"//bibliotheque de backgroundsubstractor
#include "opencv2/video/tracking.hpp"//bibliotheque kalmanFilter
#include "opencv2\imgproc\imgproc.hpp"//biliotheque de findContours et drawContours
#include "opencv2\core\core.hpp"//bibliotheque qui contien le type Size
#include<vector>//biblioteque qui va contenir nos coordonner de contours

using namespace std;
using namespace cv;

#define drawCross( img, center, color, d )\
line(img, Point(center.x - d, center.y - d), Point(center.x + d, center.y + d), color, 2, CV_AA, 0);\
line(img, Point(center.x + d, center.y - d), Point(center.x - d, center.y + d), color, 2, CV_AA, 0 )\
//dessine une croix  sur la martrice img par les coordonner des  Points
//line ==dessine le segment de ligne (PT1, PT2) dans l'image: (CV_IN_OUT Mat& img, Point pt1, Point pt2, const Scalar& color,int thickness=1, int lineType=8, int shift=0);

class Image
{		
public : 
	//initialisation de la capture de webcam
	Mat Orig;
	Mat plant1;
	Mat plant2;
	Size size;
	//Constructeur 
	Image();
	//creation d un element qui va detecter le fond et le plant part un Algorithm de Segmentation  basée sur Mixture Gaussian du Font / Premier plan (MOG/Gaussian Mixture-based Background/Foreground Segmentation Algorithm)
	BackgroundSubtractorMOG bg;	
	/*Creation d un element dont La classe implémente l'algorithme suivant:"Un modèle adaptatif de mélange de fond
	   amélioré pour le suivi en temps réel avec détection de l'ombre"(Algorithm de Segmentation  basée sur Mixture Gaussian du Font / Premier plan)*/
	BackgroundSubtractorMOG2 bg2;
	
	int MenuVideo();
	int AppFiltre();
	Mat FonctionFiltre(int l,Mat &plant1,Mat &plant2,Size size);//on fait apple au adresse de plant1 et plant2 pour applique les filtre
	Mat FindDrawContour(Mat plant2,Mat Orig);
	void Regarder(Mat plant1,Mat Orig);
	void Destruction(Mat plan1,Mat plant2,Mat Orig,BackgroundSubtractorMOG bg,BackgroundSubtractorMOG2 bg2);
	void fonctionKalman();
	void fonctionKalmanBackground();
	
};


#endif