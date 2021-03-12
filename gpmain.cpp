#include "gpmisc.h"
#include "gp1SIERRA_main.h"
/**/
// @author Gemma S. Parra-Dominguez gemmasparra at gmail.com
//
//  This project was created to compute 29 facial symmetry features from a set of 68 extracted landmarks,
//  additionally, the input image is draw to visualize the 51 facial key points.
//
//  Input data:
//      +images should be in ".jpg" format and named as "test_image_XXX.jpg", where XXX are consecutive number 000, 001,...,XXX
//      +facial landmarks must be stored on a ".csv" file and named as "test_image_XXX.csv", where XXX are consecutive numbers 000, 001,...,XXX
//
//  Output data:
//      +images showing the 51 facial landmarks will be named as "test_image_XXX_keypts_out.jpg"
//      +symmetry measures will be named as "smfeat_XXX_out.csv"
//
//  Default:
//      +input data must be stored at "test_data/"
//      +results will be stored at "test_results/"
//      +variable num_subjects refers to the number of images to be processed, by default is set to 5, modified as required
//
// If you use this code, please cite:
//      Parra-Dominguez, G.S.; Sanchez-Yanez, R.E.; Garcia-Capulin, C.H.
//      Facial Paralysis Detection on Images Using Key Point Analysis.
//      Appl. Sci. 2021, 11, 2435. https://doi.org/10.3390/app11052435
/**/

int main()
{
    Mat data_keypts, img_color, img_rot, mat_landmarks;
    int num_subjects = 5;
    /* image paths */
    string image_root("test_image_");
    string test_image_root = test_image_path + image_root;
    /* processing */
    gpComputeSRM(num_subjects, test_image_root, 0);
    waitKey(0);

    return 0;
}
