Facial Paralysis Detection on Images Using Key Point Analysis
============================================================================================
Software to autmatically compute facial measures from previously extracted facial landmarks.
Required packages:
+ OpenCV 
This project was created with opencv 4.2.0 and the additional modules that can be obtained from https://github.com/opencv/opencv_contrib

Default:
+input data must be stored at "test_data/"
+results will be stored at "test_results/"

Input data:
+images should be in ".jpg" format and named as "test_image_XXX.jpg", where XXX are consecutive number 000, 001,...,XXX
+facial landmarks must be stored on a ".csv" file and named as "test_image_XXX.csv", where XXX are consecutive numbers 000, 001,...,XXX

Output data:
+images showing the 51 facial landmarks will be named as "test_image_XXX_keypts_out.jpg"
+symmetry measures will be named as "smfeat_XXX_out.csv"

If you use this software please cite the following paper:
+ Parra-Dominguez, G.S.; Sanchez-Yanez, R.E.; Garcia-Capulin, C.H. Facial Paralysis Detection on Images Using Key Point Analysis. Appl. Sci. 2021, 11, 2435. https://doi.org/10.3390/app11052435

================================================================================================================================

If you wish to extrac the facial landmarks using the mee_shape_predictor_68_face_landmarks, you will need to run "command_line.py",
which is a python script developed by Diego L.Guarin and modified by Gemma S. Parra-Dominguez.
Required packages:
+ dlib
+ OpenCV

Executing the code: Download and install python (I used 3.6), I recomend to install anaconda (https://www.anaconda.com/download/) and use Spider to edit and excute the code. Once conda is installed you can add anaconda to your enviromental variables (in windows) and excute the following commands in the command line (cmd)

    conda install -c menpo opencv3
    conda install -c menpo dlib

this will allow you to use openCV and dlib in your python installation. There are ways to compile your own version of dlib and OpenCV, google is your friend.
Default:
+input data must be stored at "test_data/"
+results will be stored at "test_data/"

If you use this software please cite the following paper:
Guarin, Diego L., Joseph Dusseldorp, Tessa A. Hadlock, and Nate Jowett. "A machine learning approach for automated facial measurements in facial palsy." JAMA facial plastic surgery (2018).

====================================================================================================================

If you wish to create ARFF files to test the methodology, you can run the matlab script "gp_data_prepare_to_arff.m"

Other software:
+ Matlab, we use Matlab R2017a but other versions should work.
+ Weka, we use the 3.8.4 version.
