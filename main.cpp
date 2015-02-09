/*main.cpp*/
#include "Image.h"//Bibliotheque Image cree contenent la class Image 
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


int main(int argc,char* argv[])
{
	Image image;//creation d'un Objet image 
	VideoCapture cap(0);//ouverture et initialisation du port de capture webcam
	int touchesortie=INT_MAX;//touche de sortie 

	cout<<"////////////////  Sujet :Suivi de Contour d Objet en Mouvement  ////////////////"<<endl<<endl<<endl;
	
	do//temps que la touche -1 n'est pas selectionner on redemande à l'utilisateur de faire une nouvelle application 
	{
		int k =image.MenuVideo();//on Choisi le mode d'isolation plant /Font que l' on souaite 1=Premier image enregister est le font de reference ,2=il effectue un rafrechisement du font
		
		if(k==1||k==2)//si k = 1 ou 2 
		{
			int l =image.AppFiltre();//on Choisi le Filtre que l on veut appliquer 
			for(;;)//on fait une boucle infini pour avoir les images en temps Reel
			{
				Image();//le constructeur initialise plant1,plant2 et  orig  apres chaque application de teste pour rafrechir le background (BackgroundSubtractorMOG) 
				cap>>image.Orig;//on enregiste l image de la webcam dans une matrice d'Origine
				if(k==1)//si le resulta du MenuVideo est 1 on va prendre le Background dont la premiere image sera pris comme reference de font
				{
					image.bg.operator()(image.Orig,image.plant1);//on applique la fonction qui isole le font du plant
					image.FonctionFiltre( l, image.plant1, image.plant2, image.size);//applique le filtre selection par image.AppFiltre
					image.FindDrawContour(image.plant2,image.Orig);//on applique les 2 fonctions qui vont trouver les points et les tracer 
				}
				else  //si le resulta du MenuVideo est 2 on va prendre le Background qui va effectuer un rafréchisement du font
				{
					image.bg2.operator()(image.Orig,image.plant1);//on applique la fonction qui isole le font du plant(avec Rafrechisement du font)
					image.FonctionFiltre( l, image.plant1, image.plant2, image.size);//applique le filtre selection par image.AppFiltre
					image.FindDrawContour(image.plant2,image.Orig);//on applique les 2 fonctions qui vont trouver les points et les tracer 
				}
				image.Regarder(image.plant1,image.Orig);//on cree et  ouvre les fenetres qui permetron de visualiser l isolation du font/plant et le resulta final appliquer a notre webcam
				if(cv::waitKey(30) >= 0) break;//si une toucher est presser dans une temps sup ou = 0 on sort de la boucle 
			}
			destroyAllWindows();//destruction de toute les fenetres (pour pouvoir commencer une nouvelle application ou sortir et video la memoire)
			cout<<"Exit?(No=0, Yes=-1)"<<endl;//demander a l utilisateur si il veut recommencer ou finir le programme
			cin>>touchesortie;
		}
		else if (k==3)//si k=3 je vais applique kalman sans isolation du font
		{
			cout<<"Sortir :Taper q "<<endl;
			image.fonctionKalman();
			destroyAllWindows();//destruction de toute les fenetres (pour pouvoir commencer une nouvelle application ou sortir et video la memoire)
			cout<<"Exit?(No=0, Yes=-1)"<<endl;//demander a l utilisateur si il veut recommencer ou finir le programme
			cin>>touchesortie;
		}
		else//si autre touche je vais applique kalmant avec isolation du font 
		{
			cout<<"Sortir :Taper q "<<endl;
			image.fonctionKalmanBackground();
			destroyAllWindows();//destruction de toute les fenetres (pour pouvoir commencer une nouvelle application ou sortir et video la memoire)
			cout<<"Exit?(No=0, Yes=-1)"<<endl;//demander a l utilisateur si il veut recommencer ou finir le programme
			cin>>touchesortie;
		}

	}
	while(touchesortie!=-1);//temps que la touche -1 n est pas saisir on refait l'application par le "do"
	image.Destruction(image.plant1,image.plant2,image.Orig,image.bg,image.bg2);//destructeur de la class Image 
	cap.release();//destructeur de videocapture cap
	system("pause");
	return 0;
}