Facial Paralysis Detection on Images Using Key Point Analysis
============================================================================================
Software to automatically compute facial measures from previously extracted facial landmarks.
Required packages:
+ OpenCV 
This project was created with opencv 4.12.0
+ Dlib
This project was created with dlib 20.0.1
+ Python
This project was created with python 3.13.9

Default:
+ input data must be stored at "test_data/"
+ results will be stored at "test_results/"
+   MAIN FILE is gpmain.cpp

Input data:
+ images should be in ".jpg" format and named as "test_image_XXX.jpg", where XXX are consecutive number 000, 001,...,XXX
+ facial landmarks must be stored on a ".csv" file and named as "test_image_XXX.csv", where XXX are consecutive numbers 000, 001,...,XXX

Output data:
+ images showing the 51 facial landmarks will be named as "test_image_XXX_facereg_keypts_out.jpg"
+ symmetry measures will be named as "smfeat_XXX_out.csv"

If you use this software please cite the following paper:
+ Parra-Dominguez, G.S.; Sanchez-Yanez, R.E.; Garcia-Capulin, C.H. Facial Paralysis Detection on Images Using Key Point Analysis. Appl. Sci. 2021, 11, 2435. https://doi.org/10.3390/app11052435
+ Contact info: gemmasparra at gmail dot com

===============================

If you wish to extrac the facial landmarks using the mee_shape_predictor_68_face_landmarks, you will need to run "command_line.py",
which is a python script developed by Diego L.Guarin and modified by Gemma S. Parra-Dominguez.
Required packages:
+ dlib
+ OpenCV

Default:
+input data must be stored at "test_data/"
+results will be stored at "test_data/" (landmark files) and "test_results/" (images)

If you use this software please cite the following paper:
Guarin, Diego L., Joseph Dusseldorp, Tessa A. Hadlock, and Nate Jowett. "A machine learning approach for automated facial measurements in facial palsy." JAMA facial plastic surgery (2018).

===============================

If you wish to create ARFF files to test the methodology, you can run function "gp_data_prepare_to_arff(path, vec_labels, cvs_name_dataset, arff_name_dataset)"

Other software:
+ Weka, we use the 3.8.4 version.
