/*Image.cpp*/
#include "Image.h"
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

Image::Image()
{ //initialisation a 0 pour suprimer l'image enregistre apres application d un  test
	Orig=0;
	plant1=0;
	plant2=0;
}
int Image::MenuVideo()//fonction qui demande quelle Menu video on veut choisir
{
	int k=0;
	cout<<"Les images sont saisir par votre webcam"<<endl;
	cout<<"Quatre choix sont possible pour visualiser les contours apres filtrage:"<<endl<<endl;
	cout<<"Taper 1:La premier image saisi est considerer comme statique "<<endl<<"      (tout difference de la premiere image  est un objet en mouvement) "<<endl;
	cout<<"Taper 2:Considerer l'objet statique une fois immobile au bout d'un certain temps "<<"     (Rafrechisement de l'image statique,capte toute modification d'image )"<<endl<<endl;
	cout<<"Taper 3:Application du Filtre de Kalman sans isoler le Font "<<endl;
	cout<<"Taper Autre: Application du Filtre de Kalman avec Isolation du Font"<<endl;
	cin>>k;
	return k;
}

int Image::AppFiltre()//fonction qui demande quelle filtre on veut appliquer 
{
	int k=0;
	cout<<"Menu des differents filtres et applications:"<<endl;
	
	cout<<"Taper 1 : Canny(detection de Bort)"<<endl;
	cout<<"Taper 2 : Filtre de Dilatation"<<endl;
	cout<<"Taper 3 : Filtre Gaussien "<<endl;
	cout<<"Taper 4 : Filtre d'Erode "<<endl;
	cout<<"Taper 5 : Filtre Median"<<endl;
	cout<<"Taper 6 : Filtre Morphologique"<<endl;
	cout<<"Taper 7 : Application de plusieur Filtre"<<endl;
	cout<<"Taper une autre touche:Sans Filtre"<<endl;
	cin>>k;
	return k;
}

Mat Image::FindDrawContour(Mat plant2,Mat Orig)//cette fonction trouve les points de contour et les traces dans l image d'origine
{
	//les contours vont etre enregistrer dans un Tableau 
	vector<vector<Point> > contours;
	//trouver le contour du 1er plant et les enregistre dans un vector contours
	findContours(plant2,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	//dessine le contours par des point sur l image d origine 
	drawContours(Orig,contours,-1,Scalar(0,0,255),2);
	contours.~vector();//destruction du tableau contours une fois le nouvelle Orig enregistrer  
	return Orig;
}

void Image::Regarder(Mat plant1,Mat Orig)//fonction qui cree et ouvre des fenetres
{
	namedWindow("WEBCAM",1);//creation d une fenetre WEBCAM
	namedWindow("ISOLATION du FONT/PLANT",1);//creation d un fenetre pour visualiser l isolation du font et du plant 
	
	//projection de la capture 
	imshow("ISOLATION du FONT/PLANT",plant1);

	imshow("WEBCAM",Orig);
}
void Image::Destruction(Mat plan1,Mat plant2,Mat Orig,BackgroundSubtractorMOG bg,BackgroundSubtractorMOG2 bg2)//appel des destructeurs
{
	plant2.~Mat();
	plant1.~Mat();
	Orig.~Mat();
	bg2.~BackgroundSubtractorMOG2();
	bg.~BackgroundSubtractorMOG();
}

Mat Image::FonctionFiltre(int l,Mat &plant1,Mat &plant2,Size size)//fonction qui applique les filtres
{
  	switch(l)//le filtre selectionner est appliquer 
				{
				case 1:cv::Canny(plant1,plant2,2,4);//Capte les  contours du plant
					cout<<"vous appliquer le Filtre de Canny"<<endl;
					break;
				case 2:
					cv::dilate(plant1,plant2,Mat());//Dilate le plant Bouche les trous les plus petits
					cout<<"vous appliquer le Filtre de Dilatation"<<endl;
					break;
				case 3:
					cv::GaussianBlur(plant1,plant2,size,2);//Capte par methode de Gauss les points du plant
					cout<<"vous appliquer le Filtre de Gauss"<<endl;
					break;
				case 4:
					cv::erode(plant1,plant2,Mat());//Erode le plant ( il fait l inverse de la dilatation Elimine les composantes connexes )
					cout<<"vous appliquer le Filtre de Erode"<<endl;
					break;
				case 5:
					cv::medianBlur(plant1,plant2,3);//capte le plant en faisent une moyenne
					cout<<"vous appliquer le Filtre Median"<<endl;
					break;
				case 6:
					cv::morphologyEx(plant1,plant2,1,Mat());//une morphology est considerer comme :dilatations et érosions pour des éléments structurants «plats» et symétriques
					cout<<"vous appliquer le Filtre de Morphologique"<<endl;
					break;
				case 7:
					cout<<"vous apppliquer 2 fois le filtre d erode pour éliminer les points blanc en trops,une fois Filtre de Morphologique pour se consacrer au contour,et le Filtre de Gauss pour des formes courber "<<endl;
					cv::erode(plant1,plant2,Mat());
					cv::erode(plant2,plant2,Mat());
					cv::morphologyEx(plant2,plant2,1,Mat());
					cv::GaussianBlur(plant2,plant2,size,2);//le filtre gaussien donne c est forme approcher au contour de l objet 
					break;
				default:
					plant2=plant1;//Aucun filtre n est appliquer le plant1 = le plant2 
					cout<<"vous n appliquer aucun Filtre"<<endl;
					break;
				}
	return plant2;
}

void Image::fonctionKalman()
{
	////////initialisation de Kalman///////////
  Mat frame, thresh_frame;//matrice image ,matrice seuil cadre
  vector<Mat> channels;//tableau de canneau 
  VideoCapture capture(0);
  vector<Vec4i> hierarchy;//tableau vecteur hierrarchy
  vector<vector<Point> > contours;//tableau Point (coordoner x et y )

  KalmanFilter KF(4, 2, 0);//creation d un objet Kalman
  Mat_<float> state(4, 1);
  Mat_<float> processNoise(4, 1, CV_32F);
  Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));

  KF.statePre.at<float>(0) = 0;
  KF.statePre.at<float>(1) = 0;
  KF.statePre.at<float>(2) = 0;
  KF.statePre.at<float>(3) = 0;

  KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1); // Including velocity
  KF.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,  0,0,0,0.3);

  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
  setIdentity(KF.errorCovPost, Scalar::all(.1));

  while((char)waitKey(1) != 'q' )//temps que q n est pas saisi on continu a appliquer le filtre 
    {
      capture.retrieve(frame);//enregistre l image saisi dans frame (meme chose que cap<<orig )c est une autre methode d enrefistrement 

      split(frame, channels);//copie les image de frame dans un tableau  de matrice channel apres les avoir binairiser 
      add(channels[0], channels[1], channels[1]);//ajoute vector 0,1l enregistre dans 1
      subtract(channels[2], channels[1], channels[2]);//soustrai 2 a 1 enregistre dans 2
     threshold(channels[2], thresh_frame, 50, 255, CV_THRESH_BINARY);//applique seuil fixe pour l'image
      medianBlur(thresh_frame, thresh_frame, 5);//adoucit l'image en utilisant un filtre médian.

      findContours(thresh_frame, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));//récupère les contours et l'information hiérarchique de l'image noir-n-blanc.

      Mat drawing = Mat::zeros(thresh_frame.size(), CV_8UC1);//initialisation de matrice
      for(size_t i = 0; i < contours.size(); i++)//boucle temps que la longeure de la sequence n est pas effectuer
        {
//          cout << contourArea(contours[i]) << endl;
          if(contourArea(contours[i]) > 500)//si le calcule de contour est superieur a 500 
            drawContours(drawing, contours, i, Scalar::all(255), CV_FILLED, 8, vector<Vec4i>(), 0, Point());//on dessine le contour dans drawing, avec une couleur au plus 255(le rouge)
        }
      thresh_frame = drawing;//la matrice thresh _frame == drawing 

// Get the moments:defini les moments
      vector<Moments> mu(contours.size() );
      for( size_t i = 0; i < contours.size(); i++ )
        { mu[i] = moments( contours[i], false ); }

//  Get the mass centers:defini le centre 
      vector<Point2f> mc( contours.size() );
      for( size_t i = 0; i < contours.size(); i++ )
        { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

      Mat prediction = KF.predict();
      Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

      for(size_t i = 0; i < mc.size(); i++)
        {
          drawCross(frame, mc[i], Scalar(255, 0, 0), 5);//on fait une croix en bleu dans frame des points predit
          measurement(0) = mc[i].x;//enregistre dans mesure 0==coordoner mc[i].x
          measurement(1) = mc[i].y;//enregistre dans mesure 1==coordonner mc[i]y
        }

      Point measPt(measurement(0),measurement(1));//un point nomer measPtcontenent les coordonner de mc[i].x,et mc[i].y

      Mat estimated = KF.correct(measurement);//met à jour l'état ​​prédit à partir de la mesure(measurement)
      Point statePt(estimated.at<float>(0),estimated.at<float>(1));//creation d un point contenent les mise a jour des points 

      drawCross(frame, statePt, Scalar(0, 255, 255), 5);//on fait une croix (jaune ) au point estimer 

      vector<vector<Point> > contours_poly( contours.size() );//creation d un tableau contour poly de taille la taille des nombres de point dans contours
      vector<Rect> boundRect( contours.size() );//cration d un tabeau boundRect de taille contour (il nous donnera les cooredonner des rectangle )
      for( size_t i = 0; i < contours.size(); i++ )//boucle tenst que la taille de I< au nobre de contours 
       { 
		   approxPolyDP( Mat(contours[i]), contours_poly[i], 50, true );//se rapproche de contour ou une courbe en utilisant l'algorithme Douglas-Peucker
		   boundRect[i] = boundingRect( Mat(contours_poly[i]) );//calcule le rectangle de délimitation pour un contour
       }

      for( size_t i = 0; i < contours.size(); i++ )
       {
         rectangle( frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0 );
       }//dessin du rectange dans fram ,coordonner de x du tableau ,coordonner d y du tableua ,de couleur vert , epaisseur 2 


      imshow("Video", frame);//affichage de la video normal 
      imshow("operation sur 3 image", channels[2]);//affichage de channel[2] apres que l image (0 +1)-2=2
      imshow("Binary", thresh_frame);
	  
    }
}

void Image::fonctionKalmanBackground()
{

  Mat frame, thresh_frame,plant1,plant2,plant3;
  vector<Mat> channels;
  VideoCapture capture(0);
  vector<Vec4i> hierarchy;
  vector<vector<Point> > contours;
  BackgroundSubtractorMOG2 bg;//objet qui permet d isoler le font du plant suivant un model gaussien 

 /////////////// //initialisation du filtre de Kalman/////////////////////////

  KalmanFilter KF(4, 2, 0);//objet filtre de kalman du nom de KF initialiser (4D,2 element de parametre mesurer ,parametre de controle toujour 0)
  Mat_<float> state(4, 1);//matrice etat( dynamique 4 ,taille1)
  Mat_<float> processNoise(4, 1, CV_32F);//le traitement de la matrice de covariance de bruit (Q)
  Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));//matrice mesure(dynamique 2,taille 1);définit une partie des éléments de matrice à scalair 0, en fonction du masque
 
  //état prédit (x '(k)): x (k) = A x * (k-1) + B * u (k) ou A et B des matrice 
  KF.statePre.at<float>(0) = 0;
  KF.statePre.at<float>(1) = 0;
  KF.statePre.at<float>(2) = 0;
  KF.statePre.at<float>(3) = 0;
  //effectue la transition de la matrice entrer
  KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1); // Including velocity
  //effectue l entrer de la matrice de covariane de bruit 
  KF.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,  0,0,0,0.3);

  //changement des valeur par Seteur
  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
  setIdentity(KF.errorCovPost, Scalar::all(.1));

  ///////Application //////

  while((char)waitKey(1) != 'q' )//temps que q n est pas presser l action continu
    {
     capture>>frame;//capture des image de l image de la webcam dans la matrice frame
	 split(frame, channels);//copie chaque image dans le tableau de matrice channels a la suite
	
	 //isolation du font du plant 
	 bg.operator()(channels[0],plant1);
	 bg.operator()(channels[1],plant2);
	 bg.operator()(channels[2],plant3);
	
	 //manipule les image par addition et soustraction de matrice pour les prediction 

	subtract(plant1,plant2,plant2);//soustrai la matrice channels 0 et channels 1 enregistrer dans channels 1
	add(plant2,plant3, plant3);//ajoute  2 a 1 enregistre dans 2
    threshold(plant3, thresh_frame, 50, 255, CV_THRESH_BINARY);//applique seuil fixe pour l'image
	medianBlur(plant3, thresh_frame, 5);//adoucit l'image en utilisant un filtre médian.

	 findContours(thresh_frame, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));//ecrit les points de thresh_frame dans un tableau de point contours,et un tableau de vecteur hyerarchie

      Mat drawing = Mat::zeros(thresh_frame.size(), CV_8UC1);//initialisation de matrice de taille le nombre de point de thresh_frame
      for(size_t i = 0; i < contours.size(); i++)//boucle temps que le nombre de contours n est pas egal a i
        {
//          cout << contourArea(contours[i]) << endl;
          if(contourArea(contours[i]) > 500)//si le calcule de contour est superieur a 500 
            drawContours(drawing, contours, i, Scalar::all(255), CV_FILLED, 8, vector<Vec4i>(), 0, Point());//on dessine le contour dans drawing, avec une couleur au plus 255
        }
      thresh_frame = drawing;//la matrice thresh _frame == drawing 

// Get the moments        //les moment vont etre calculer a partir des point contours
      vector<Moments> mu(contours.size() );//tableau de moment de taille le nombre de points contours
      for( size_t i = 0; i < contours.size(); i++ )
        { mu[i] = moments( contours[i], false ); }//calcule des moments de la forme pixellisée ou un vecteur de points

//  Get the mass centers:
      vector<Point2f> mc( contours.size() );//tableau de point2f de taille le nombre de points contours
      for( size_t i = 0; i < contours.size(); i++ )
        { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

      Mat prediction = KF.predict();
      Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

      for(size_t i = 0; i < mc.size(); i++)
        {
          drawCross(frame, mc[i], Scalar(255, 0, 0), 5);//dessiner croix en bleu dans frame se sont les point predit
          measurement(0) = mc[i].x;//enregistre dans mesure 0==coordoner mc[i].x
          measurement(1) = mc[i].y;//enregistre dans mesure 1==coordonner mc[i]y
        }

      Point measPt(measurement(0),measurement(1));//un point nomer measPtcontenent les coordonner de mc[i].x,et mc[i].y

      Mat estimated = KF.correct(measurement);//met à jour l'état ​​prédit à partir de la mesure(measurement)
      Point statePt(estimated.at<float>(0),estimated.at<float>(1));//creation d un point contenent les mise a jour des points 

      drawCross(frame, statePt, Scalar(255, 0, 255), 5);//on trace une croix violette

      vector<vector<Point> > contours_poly( contours.size() );//creation d un tableau contour poly de taille la taille des nombres de point dans contours
      vector<Rect> boundRect( contours.size() );//cration d un tabeau boundRect de taille contour (il nous donnera les cooredonner des rectangle )
     

	  for( size_t i = 0; i < contours.size(); i++ )//boucle tenst que la taille de I< au nobre de contours 
       {
		   approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );//se rapproche de contour ou une courbe en utilisant l'algorithme Douglas-Peucker
		   boundRect[i] = boundingRect( Mat(contours_poly[i]) );//calcule le rectangle de délimitation pour un contour
		
       }
	

      for( size_t i = 0; i < contours.size(); i++ )
       {
         rectangle( frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0 );
       }//dessin du rectange dans fram ,coordonner de x du tableau ,coordonner d y du tableau ,de couleur vert , epaisseur 2 


      imshow("Video", frame);//affichage de la video normal 
      imshow("Image font plant Isoler et manipulation de l image ", thresh_frame);//affichage  apres que l image (0 -1)+2=2
	  
    }
}