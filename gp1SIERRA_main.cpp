#include "gpmisc.h"
#include "gp1SIERRA_main.h"

void gpComputeSRM(int num_test_images, string test_image_root, int init_img)
{
    Mat img_color, img_lbf, mat_landmarks;
    Mat data_keypts, coord_vector, norm_vector, smr_vector;
    string file_name, img_name;
    char cbuffs[20];
    int no_img_found = 0;
    int face_found = 0;
    /* Processing */
    for (int ik = init_img; ik < num_test_images; ik++)
    {
        if (ik < 1000)
            sprintf (cbuffs, "%03d", ik);
        else
            sprintf (cbuffs, "%04d", ik);
        img_name = test_image_root + cbuffs + image_ext;
        file_name = test_image_root + cbuffs + csv_name;
        cout << file_name << endl;
        mat_landmarks = readCSV(file_name);
        img_color = imread(img_name);
        if (mat_landmarks.empty())
        {
            no_img_found++;
            cout << "no data found..." << endl;
            smr_vector = Mat::ones(1,1,CV_32FC1)*(-1);
            file_name = test_results_path + "smfeat_" + "test_image_" + cbuffs + csv_name;
            writeCSV(file_name, smr_vector);
        }
        else
        {
            /* matrix rotation */
            img_lbf = gpFaceReg(img_color, mat_landmarks, data_keypts);
            for (int jk = FIRST_LANDMARK; jk < data_keypts.rows; jk++)
            {
                circle(img_lbf, Point(data_keypts.at<float>(jk,0), data_keypts.at<float>(jk,1)), 6, Scalar(0, 0, 255),FILLED);
            }
            file_name = test_results_path + "test_image_" + cbuffs + "_keypts_out" + image_ext;
            imwrite(file_name, img_lbf);
            face_found++;
            /* Processing */
            gpPtsExt(data_keypts, coord_vector);
            /* SMR */
            smr_vector = Mat::zeros(TOTAL_SMR, 1, CV_32FC1);
            gpGetSMR(coord_vector, smr_vector);
            gpGetFMM(coord_vector, smr_vector);
            file_name = test_results_path + "smfeat_" + "test_image_" + cbuffs + csv_name;
            writeCSV(file_name, smr_vector);
        }
    }
    cout << "Number of files not found: " << no_img_found << endl;
    cout << "Number of files found: " << face_found << endl;
    cout << "Total of supposed data: " << (num_test_images-init_img) << endl;
}

Mat gpFaceReg(Mat &image_face, Mat &data_keypts, Mat &mat_landmarks)
{
    Mat rot_face;
    double angle = 0;
    float delta_x = data_keypts.at<float>(0,0) - data_keypts.at<float>(16,0);
    float delta_y = data_keypts.at<float>(0,1) - data_keypts.at<float>(16,1);
    if(isnan(atan2(delta_y, delta_x)) == 0)
        angle = atan2(delta_y, delta_x)*180/CV_PI;
    if (angle > 0)
        angle -=180;
    else
        angle +=180;
    Mat rot_mat = getRotationMatrix2D(Point2f(data_keypts.at<float>(0,0),data_keypts.at<float>(0,1)),angle,1.0);
    warpAffine(image_face, rot_face, rot_mat, image_face.size());
    /**/
    rot_mat.convertTo(rot_mat,CV_32FC1);
    mat_landmarks = data_keypts.clone();
    for (int ik = 0; ik < data_keypts.rows; ik++)
    {
        mat_landmarks.at<float>(ik,0) = data_keypts.at<float>(ik,0)*rot_mat.at<float>(0,0)+data_keypts.at<float>(ik,1)*rot_mat.at<float>(0,1)+rot_mat.at<float>(0,2);
        mat_landmarks.at<float>(ik,1) = data_keypts.at<float>(ik,0)*rot_mat.at<float>(1,0)+data_keypts.at<float>(ik,1)*rot_mat.at<float>(1,1)+rot_mat.at<float>(1,2);
    }

    return rot_face;
}

float gpEuclideanDist (float x_land, float y_land, float x_key, float y_key)
{
    float aux_x, aux_y;
    float p = 2;

    aux_x = pow((x_land-x_key),p);
    aux_y = pow((y_land-y_key),p);

    return sqrt(aux_x + aux_y);
}

void gpPtsExt(Mat &mat_landmarks, Mat &coord_vector)
{
    int cont = 0;
    coord_vector = Mat::zeros(TOTAL_LANDMARK, 2, CV_32FC1);
    //  Eyebrow
    for (int ik = 17; ik <= 26; ik++)
    {
        coord_vector.at<float>(cont,0) = mat_landmarks.at<float>(ik,0);
        coord_vector.at<float>(cont,1) = mat_landmarks.at<float>(ik,1);
        cont++;
    }
    // Eye
    for (int ik = 36; ik <= 47; ik++)
    {
        coord_vector.at<float>(cont,0) = mat_landmarks.at<float>(ik,0);
        coord_vector.at<float>(cont,1) = mat_landmarks.at<float>(ik,1);
        cont++;
    }
    // Nose
    for (int ik = 30; ik <= 35; ik++)
    {
        coord_vector.at<float>(cont,0) = mat_landmarks.at<float>(ik,0);
        coord_vector.at<float>(cont,1) = mat_landmarks.at<float>(ik,1);
        cont++;
    }
    // Mouth
    for (int ik = 48; ik <= 67; ik++)
    {
        coord_vector.at<float>(cont,0) = mat_landmarks.at<float>(ik,0);
        coord_vector.at<float>(cont,1) = mat_landmarks.at<float>(ik,1);
        cont++;
    }
    // Jaw
    coord_vector.at<float>(cont,0) = mat_landmarks.at<float>(0,0);
    coord_vector.at<float>(cont,1) = mat_landmarks.at<float>(0,1);
    cont++;
    coord_vector.at<float>(cont,0) = mat_landmarks.at<float>(16,0);
    coord_vector.at<float>(cont,1) = mat_landmarks.at<float>(16,1);
    cont++;
    coord_vector.at<float>(cont,0) = mat_landmarks.at<float>(8,0);
    coord_vector.at<float>(cont,1) = mat_landmarks.at<float>(8,1);
}

void gpGetFMM(Mat &norm_vector, Mat &face_smr_data)
{
    // distance using the first and last jaw point
    float a_dist = gpEuclideanDist(norm_vector.at<float>(49,0), norm_vector.at<float>(49,1), norm_vector.at<float>(48,0), norm_vector.at<float>(48,1));
    // left out point of the mouth to the middle mouth
    float b_dist = gpEuclideanDist(norm_vector.at<float>(37,0), norm_vector.at<float>(37,1), norm_vector.at<float>(28,0), norm_vector.at<float>(28,1));
    // right out point of the mouth to the middle mouth
    float c_dist = gpEuclideanDist(norm_vector.at<float>(37,0), norm_vector.at<float>(37,1), norm_vector.at<float>(34,0), norm_vector.at<float>(34,1));
    // distance from left eye brown to middle mouth
    float d_dist = gpEuclideanDist(norm_vector.at<float>(37,0), norm_vector.at<float>(37,1), norm_vector.at<float>(2,0), norm_vector.at<float>(2,1));
    float e_dist = gpEuclideanDist(norm_vector.at<float>(37,0), norm_vector.at<float>(37,1), norm_vector.at<float>(3,0), norm_vector.at<float>(3,1));
    // distance from right eye brown to middle mouth
    float f_dist = gpEuclideanDist(norm_vector.at<float>(6,0), norm_vector.at<float>(6,1), norm_vector.at<float>(37,0), norm_vector.at<float>(37,1));
    float g_dist = gpEuclideanDist(norm_vector.at<float>(7,0), norm_vector.at<float>(7,1), norm_vector.at<float>(37,0), norm_vector.at<float>(37,1));
    // mouth width
    float h_dist = gpEuclideanDist(norm_vector.at<float>(34,0), norm_vector.at<float>(34,1), norm_vector.at<float>(28,0), norm_vector.at<float>(28,1));
    // distance nose to mouth
    float i_dist = gpEuclideanDist(norm_vector.at<float>(31,0), norm_vector.at<float>(31,1), norm_vector.at<float>(25,0), norm_vector.at<float>(25,1));
    // left mouth height
    float sl_dist = gpEuclideanDist(norm_vector.at<float>(29,0), norm_vector.at<float>(29,1), norm_vector.at<float>(39,0), norm_vector.at<float>(39,1));
    float su_dist = gpEuclideanDist(norm_vector.at<float>(30,0), norm_vector.at<float>(30,1), norm_vector.at<float>(38,0), norm_vector.at<float>(38,1));
    // right mouth height
    float tl_dist = gpEuclideanDist(norm_vector.at<float>(33,0), norm_vector.at<float>(33,1), norm_vector.at<float>(35,0), norm_vector.at<float>(35,1));
    float tu_dist = gpEuclideanDist(norm_vector.at<float>(32,0), norm_vector.at<float>(32,1), norm_vector.at<float>(36,0), norm_vector.at<float>(36,1));
    // mouth perimeter
    float ml_dist = gpEuclideanDist(norm_vector.at<float>(28,0), norm_vector.at<float>(28,1), norm_vector.at<float>(29,0), norm_vector.at<float>(29,1));
    ml_dist += gpEuclideanDist(norm_vector.at<float>(30,0), norm_vector.at<float>(30,1), norm_vector.at<float>(29,0), norm_vector.at<float>(29,1));
    ml_dist += gpEuclideanDist(norm_vector.at<float>(30,0), norm_vector.at<float>(30,1), norm_vector.at<float>(31,0), norm_vector.at<float>(31,1));
    ml_dist += gpEuclideanDist(norm_vector.at<float>(37,0), norm_vector.at<float>(37,1), norm_vector.at<float>(31,0), norm_vector.at<float>(31,1));
    ml_dist += gpEuclideanDist(norm_vector.at<float>(38,0), norm_vector.at<float>(38,1), norm_vector.at<float>(37,0), norm_vector.at<float>(37,1));
    ml_dist += gpEuclideanDist(norm_vector.at<float>(38,0), norm_vector.at<float>(38,1), norm_vector.at<float>(39,0), norm_vector.at<float>(39,1));
    ml_dist += gpEuclideanDist(norm_vector.at<float>(28,0), norm_vector.at<float>(28,1), norm_vector.at<float>(39,0), norm_vector.at<float>(39,1));
    //
    float mr_dist = gpEuclideanDist(norm_vector.at<float>(32,0), norm_vector.at<float>(32,1), norm_vector.at<float>(31,0), norm_vector.at<float>(31,1));
    mr_dist += gpEuclideanDist(norm_vector.at<float>(32,0), norm_vector.at<float>(32,1), norm_vector.at<float>(33,0), norm_vector.at<float>(33,1));
    mr_dist += gpEuclideanDist(norm_vector.at<float>(34,0), norm_vector.at<float>(34,1), norm_vector.at<float>(33,0), norm_vector.at<float>(33,1));
    mr_dist += gpEuclideanDist(norm_vector.at<float>(34,0), norm_vector.at<float>(34,1), norm_vector.at<float>(35,0), norm_vector.at<float>(35,1));
    mr_dist += gpEuclideanDist(norm_vector.at<float>(36,0), norm_vector.at<float>(36,1), norm_vector.at<float>(35,0), norm_vector.at<float>(35,1));
    mr_dist += gpEuclideanDist(norm_vector.at<float>(36,0), norm_vector.at<float>(36,1), norm_vector.at<float>(37,0), norm_vector.at<float>(37,1));
    mr_dist += gpEuclideanDist(norm_vector.at<float>(37,0), norm_vector.at<float>(37,1), norm_vector.at<float>(31,0), norm_vector.at<float>(31,1));
    /* FMM */
    // f18
    if (b_dist > c_dist)
        face_smr_data.at<float>(18,0) = b_dist/a_dist;
    else
        face_smr_data.at<float>(18,0) = c_dist/a_dist;
    // f19
    if(sl_dist > tl_dist)
        face_smr_data.at<float>(19,0) = sl_dist/h_dist;
    else
        face_smr_data.at<float>(19,0) = tl_dist/h_dist;
    // f20
    if (su_dist > tu_dist)
        face_smr_data.at<float>(20,0) = su_dist/h_dist;
    else
        face_smr_data.at<float>(20,0) = tu_dist/h_dist;
    // f21
    if (ml_dist > mr_dist)
        face_smr_data.at<float>(21,0) = ml_dist/h_dist;
    else
        face_smr_data.at<float>(21,0) = mr_dist/h_dist;
    // f25
    if (d_dist > g_dist)
        face_smr_data.at<float>(25,0) = d_dist/a_dist;
    else
        face_smr_data.at<float>(25,0) = g_dist/a_dist;
    // f26
    if (e_dist > f_dist)
        face_smr_data.at<float>(26,0) = e_dist/a_dist;
    else
        face_smr_data.at<float>(26,0) = f_dist/a_dist;
    // f28
    face_smr_data.at<float>(28,0) = i_dist/a_dist;

}

void gpGetSMR(Mat &norm_vector, Mat &face_smr_data)
{
    float a_dist, b_dist, c_dist, d_dist, h_dist, i_dist, e_dist, f_dist, gl_dist, gr_dist, j_dist, k_dist, m_dist, n_dist, o_dist, p_dist;
    float l_dist, ql_dist, qr_dist, rl_dist, rr_dist, sl_dist, su_dist, tl_dist, tu_dist;
    float q_avg, r_avg;
    float delta_x, delta_y;
    float mineval = 0.0001;
    // distance using the first and last jaw point
    b_dist = gpEuclideanDist(norm_vector.at<float>(49,0), norm_vector.at<float>(49,1), norm_vector.at<float>(48,0), norm_vector.at<float>(48,1));
    // distance from outer eye point to middle mouth
    e_dist = gpEuclideanDist(norm_vector.at<float>(37,0), norm_vector.at<float>(37,1), norm_vector.at<float>(10,0), norm_vector.at<float>(10,1));
    // distance from outer eye point to middle mouth
    f_dist = gpEuclideanDist(norm_vector.at<float>(19,0), norm_vector.at<float>(19,1), norm_vector.at<float>(37,0), norm_vector.at<float>(37,1));
    // eye width
    gl_dist = gpEuclideanDist(norm_vector.at<float>(13,0), norm_vector.at<float>(13,1), norm_vector.at<float>(10,0), norm_vector.at<float>(10,1));
    gr_dist = gpEuclideanDist(norm_vector.at<float>(19,0), norm_vector.at<float>(19,1), norm_vector.at<float>(16,0), norm_vector.at<float>(16,1));
    // distance eye corner to head side
    j_dist = gpEuclideanDist(norm_vector.at<float>(10,0), norm_vector.at<float>(10,1), norm_vector.at<float>(48,0), norm_vector.at<float>(48,1));
    k_dist = gpEuclideanDist(norm_vector.at<float>(49,0), norm_vector.at<float>(49,1), norm_vector.at<float>(19,0), norm_vector.at<float>(19,1));
    // distance mouth to jaw
    l_dist = gpEuclideanDist(norm_vector.at<float>(50,0), norm_vector.at<float>(50,1), norm_vector.at<float>(37,0), norm_vector.at<float>(37,1));
    // distance from outer eye to nose tip
    m_dist = gpEuclideanDist(norm_vector.at<float>(10,0), norm_vector.at<float>(10,1), norm_vector.at<float>(23,0), norm_vector.at<float>(23,1));
    n_dist = gpEuclideanDist(norm_vector.at<float>(19,0), norm_vector.at<float>(19,1), norm_vector.at<float>(27,0), norm_vector.at<float>(27,1));
    // distance from middle mouth to nose tip
    o_dist = gpEuclideanDist(norm_vector.at<float>(23,0), norm_vector.at<float>(23,1), norm_vector.at<float>(37,0), norm_vector.at<float>(37,1));
    p_dist = gpEuclideanDist(norm_vector.at<float>(27,0), norm_vector.at<float>(27,1), norm_vector.at<float>(37,0), norm_vector.at<float>(37,1));
    // eye height
    ql_dist = gpEuclideanDist(norm_vector.at<float>(11,0), norm_vector.at<float>(11,1), norm_vector.at<float>(15,0), norm_vector.at<float>(15,1));
    qr_dist = gpEuclideanDist(norm_vector.at<float>(12,0), norm_vector.at<float>(12,1), norm_vector.at<float>(14,0), norm_vector.at<float>(14,1));
    q_avg = (abs(norm_vector.at<float>(11,1)-norm_vector.at<float>(15,1)) + abs(norm_vector.at<float>(12,1)-norm_vector.at<float>(14,1)))/2.0;
    if (q_avg <= mineval)
        q_avg = 0;
    // eye height
    rl_dist = gpEuclideanDist(norm_vector.at<float>(17,0), norm_vector.at<float>(17,1), norm_vector.at<float>(21,0), norm_vector.at<float>(21,1));
    rr_dist = gpEuclideanDist(norm_vector.at<float>(18,0), norm_vector.at<float>(18,1), norm_vector.at<float>(20,0), norm_vector.at<float>(20,1));
    r_avg = (abs(norm_vector.at<float>(17,1)-norm_vector.at<float>(21,1)) + abs(norm_vector.at<float>(18,1)-norm_vector.at<float>(20,1)))/2.0;
    if (r_avg <= mineval)
        r_avg = 0;
    // mouth height
    sl_dist = gpEuclideanDist(norm_vector.at<float>(29,0), norm_vector.at<float>(29,1), norm_vector.at<float>(39,0), norm_vector.at<float>(39,1));
    su_dist = gpEuclideanDist(norm_vector.at<float>(30,0), norm_vector.at<float>(30,1), norm_vector.at<float>(38,0), norm_vector.at<float>(38,1));
    // mouth height
    tl_dist = gpEuclideanDist(norm_vector.at<float>(33,0), norm_vector.at<float>(33,1), norm_vector.at<float>(35,0), norm_vector.at<float>(35,1));
    tu_dist = gpEuclideanDist(norm_vector.at<float>(32,0), norm_vector.at<float>(32,1), norm_vector.at<float>(36,0), norm_vector.at<float>(36,1));
    // eyebrow height
    i_dist = norm_vector.at<float>(0,1) + norm_vector.at<float>(1,1) + norm_vector.at<float>(2,1) + norm_vector.at<float>(3,1) + norm_vector.at<float>(4,1);
    i_dist /= 5;
    a_dist = norm_vector.at<float>(5,1) + norm_vector.at<float>(6,1) + norm_vector.at<float>(7,1) + norm_vector.at<float>(8,1) + norm_vector.at<float>(9,1);
    a_dist /= 5;
    // eyebrow distances
    c_dist = abs(norm_vector.at<float>(9,1) - norm_vector.at<float>(0,1))/abs(norm_vector.at<float>(9,0) - norm_vector.at<float>(0,0));
    d_dist = abs(norm_vector.at<float>(7,1) - norm_vector.at<float>(2,1))/abs(norm_vector.at<float>(7,0) - norm_vector.at<float>(2,0));
    h_dist = abs(norm_vector.at<float>(5,1) - norm_vector.at<float>(4,1))/abs(norm_vector.at<float>(5,0) - norm_vector.at<float>(4,0));
    /* Symmetry features */
    // f0
    delta_x = norm_vector.at<float>(0,0) - norm_vector.at<float>(9,0);
    delta_y = norm_vector.at<float>(0,1) - norm_vector.at<float>(9,1);
    if(isnan(atan2(delta_y, delta_x)) == 0)
        face_smr_data.at<float>(0,0) = abs(atan2(delta_y, delta_x)*180/CV_PI);
    // f1
    delta_x = norm_vector.at<float>(2,0) - norm_vector.at<float>(7,0);
    delta_y = norm_vector.at<float>(2,1) - norm_vector.at<float>(7,1);
    if(isnan(atan2(delta_y, delta_x)) == 0)
        face_smr_data.at<float>(1,0) = abs(atan2(delta_y, delta_x)*180/CV_PI);
    // f2
    delta_x = norm_vector.at<float>(4,0) - norm_vector.at<float>(5,0);
    delta_y = norm_vector.at<float>(4,1) - norm_vector.at<float>(5,1);
    if(isnan(atan2(delta_y, delta_x)) == 0)
        face_smr_data.at<float>(2,0) = abs(atan2(delta_y, delta_x)*180/CV_PI);
    // f3
    if (a_dist < i_dist)
        face_smr_data.at<float>(3,0) = a_dist/i_dist;
    else
        face_smr_data.at<float>(3,0) = i_dist/a_dist;
    // f4
    face_smr_data.at<float>(4,0) = c_dist;
    // f5
    face_smr_data.at<float>(5,0) = d_dist;
    // f6
    face_smr_data.at<float>(6,0) = h_dist;
    // f7
    delta_x = norm_vector.at<float>(19,0) - norm_vector.at<float>(10,0);
    delta_y = norm_vector.at<float>(19,1) - norm_vector.at<float>(10,1);
    if(isnan(atan2(delta_y, delta_x)) == 0)
        face_smr_data.at<float>(7,0) = abs(atan2(delta_y, delta_x)*180/CV_PI);
    // f8
    if (gl_dist < gr_dist)
        face_smr_data.at<float>(8,0) = gl_dist/gr_dist;
    else
        face_smr_data.at<float>(8,0) = gr_dist/gl_dist;
    // f9
    if (j_dist < k_dist)
        face_smr_data.at<float>(9,0) = j_dist/k_dist;
    else
        face_smr_data.at<float>(9,0) = k_dist/j_dist;
    // f10
    if (m_dist < n_dist)
        face_smr_data.at<float>(10,0) = m_dist/n_dist;
    else
        face_smr_data.at<float>(10,0) = n_dist/m_dist;
    // f11
    if (q_avg < r_avg)
        face_smr_data.at<float>(11,0) = q_avg/r_avg;
    else
        face_smr_data.at<float>(11,0) = r_avg/q_avg;
    // f12
    if (ql_dist < rr_dist)
        face_smr_data.at<float>(12,0) = ql_dist/rr_dist;
    else
        face_smr_data.at<float>(12,0) = rr_dist/ql_dist;
    // f13
    if (qr_dist < rl_dist)
        face_smr_data.at<float>(13,0) = qr_dist/rl_dist;
    else
        face_smr_data.at<float>(13,0) = rl_dist/qr_dist;
    // f14
    delta_x = norm_vector.at<float>(34,0) - norm_vector.at<float>(28,0);
    delta_y = norm_vector.at<float>(34,1) - norm_vector.at<float>(28,1);
    if(isnan(atan2(delta_y, delta_x)) == 0)
        face_smr_data.at<float>(14,0) = abs(atan2(delta_y, delta_x)*180/CV_PI);
    // f15
    if (e_dist < f_dist)
        face_smr_data.at<float>(15,0) = e_dist/f_dist;
    else
        face_smr_data.at<float>(15,0) = f_dist/e_dist;
    // f16
    if(sl_dist < tl_dist)
        face_smr_data.at<float>(16,0) = sl_dist/tl_dist;
    else
        face_smr_data.at<float>(16,0) = tl_dist/sl_dist;
    // f17
    if (su_dist < tu_dist)
        face_smr_data.at<float>(17,0) = su_dist/tu_dist;
    else
        face_smr_data.at<float>(17,0) = tu_dist/su_dist;
    // f18
    // f19
    // f20
    // f21
    // f22
    delta_x = norm_vector.at<float>(27,0) - norm_vector.at<float>(23,0);
    delta_y = norm_vector.at<float>(27,1) - norm_vector.at<float>(23,1);
    if(isnan(atan2(delta_y, delta_x)) == 0)
        face_smr_data.at<float>(22,0) = abs(atan2(delta_y, delta_x)*180/CV_PI);
    // f23
    delta_x = norm_vector.at<float>(37,0) - norm_vector.at<float>(22,0);
    delta_y = norm_vector.at<float>(37,1) - norm_vector.at<float>(22,1);
    if(isnan(atan2(delta_y, delta_x)) == 0)
        face_smr_data.at<float>(23,0) = abs(atan2(delta_y, delta_x)*180/CV_PI);
    // f24
    if (o_dist < p_dist)
        face_smr_data.at<float>(24,0) = o_dist/p_dist;
    else
        face_smr_data.at<float>(24,0) = p_dist/o_dist;
    // f27
    if (b_dist != 0)
        face_smr_data.at<float>(27,0) = l_dist/b_dist;
    // f28
}
