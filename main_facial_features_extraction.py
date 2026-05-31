import cv2
import numpy as np
import math
import os
import glob

from config import *
from utils import readCSV, writeCSV, gpEuclideanDist
from run_prepare_data_to_arff import gp_data_prepare_to_arff

def gpFaceReg(image_face, data_keypts):
    """
    Aligns and rotates the face to normalize its position based on jaw landmarks.

    Calculates the angle between the first and last jaw points (index 0 and 16)
    and applies an affine rotation matrix to both the original image and the
    corresponding landmark coordinates.

    Args:
        image_face (np.ndarray): The input face image (BGR matrix).
        data_keypts (np.ndarray): A (68, 2) array of facial landmark coordinates.

    Returns:
        tuple:
            - rot_face (np.ndarray): The aligned, rotated face image.
            - mat_landmarks (np.ndarray): The updated coordinates matching the rotated image.
    """
    angle = 0
    delta_x = data_keypts[0, 0] - data_keypts[16, 0]
    delta_y = data_keypts[0, 1] - data_keypts[16, 1]

    if not math.isnan(math.atan2(delta_y, delta_x)):
        angle = math.atan2(delta_y, delta_x) * 180 / math.pi

    if angle > 0:
        angle -= 180
    else:
        angle += 180

    center = (int(data_keypts[0, 0]), int(data_keypts[0, 1]))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_face = cv2.warpAffine(image_face, rot_mat, (image_face.shape[1], image_face.shape[0]))

    mat_landmarks = data_keypts.copy()
    for ik in range(data_keypts.shape[0]):
        mat_landmarks[ik, 0] = round(data_keypts[ik, 0] * rot_mat[0, 0] + data_keypts[ik, 1] * rot_mat[0, 1] + rot_mat[0, 2], 4)
        mat_landmarks[ik, 1] = round(data_keypts[ik, 0] * rot_mat[1, 0] + data_keypts[ik, 1] * rot_mat[1, 1] + rot_mat[1, 2], 4)

    return rot_face, mat_landmarks

def gpPtsExt(mat_landmarks):
    """
    Extracts a specific subset of 51 relevant facial landmarks from the full 68-point set.

    The subset includes specific points from the eyebrows, eyes, nose, mouth, and jaw.

    Args:
        mat_landmarks (np.ndarray): The (68, 2) array of all facial landmarks.

    Returns:
        np.ndarray: A (51, 2) array containing the selected coordinates for feature measurement.
    """
    coord_vector = np.zeros((TOTAL_LANDMARK, 2), dtype=np.float32)
    cont = 0

    # Eyebrow
    for ik in range(17, 27):
        coord_vector[cont] = mat_landmarks[ik]
        cont += 1

    # Eye
    for ik in range(36, 48):
        coord_vector[cont] = mat_landmarks[ik]
        cont += 1

    # Nose
    for ik in range(30, 36):
        coord_vector[cont] = mat_landmarks[ik]
        cont += 1

    # Mouth
    for ik in range(48, 68):
        coord_vector[cont] = mat_landmarks[ik]
        cont += 1

    # Jaw
    coord_vector[cont] = mat_landmarks[0]
    cont += 1

    coord_vector[cont] = mat_landmarks[16]
    cont += 1

    coord_vector[cont] = mat_landmarks[8]

    return coord_vector

def gpGetFMM(norm_vector, face_smr_data):
    """
    Computes advanced Facial Movement Measures (FMM) based on facial dimensions.

    Calculates various Euclidean distances across the face (e.g., mouth width,
    eye to mouth, jaw width) and derives specific symmetry/proportion ratios.
    These computed features are updated in-place into the `face_smr_data` array.

    Args:
        norm_vector (np.ndarray): A (51, 2) array of normalized extracted landmarks.
        face_smr_data (np.ndarray): A 2D column vector storing the final SMR metrics.
                                    (Modified in place).
    """
    a_dist = gpEuclideanDist(norm_vector[49, 0], norm_vector[49, 1], norm_vector[48, 0], norm_vector[48, 1])
    b_dist = gpEuclideanDist(norm_vector[37, 0], norm_vector[37, 1], norm_vector[28, 0], norm_vector[28, 1])
    c_dist = gpEuclideanDist(norm_vector[37, 0], norm_vector[37, 1], norm_vector[34, 0], norm_vector[34, 1])
    d_dist = gpEuclideanDist(norm_vector[37, 0], norm_vector[37, 1], norm_vector[2, 0], norm_vector[2, 1])
    e_dist = gpEuclideanDist(norm_vector[37, 0], norm_vector[37, 1], norm_vector[3, 0], norm_vector[3, 1])
    f_dist = gpEuclideanDist(norm_vector[6, 0], norm_vector[6, 1], norm_vector[37, 0], norm_vector[37, 1])
    g_dist = gpEuclideanDist(norm_vector[7, 0], norm_vector[7, 1], norm_vector[37, 0], norm_vector[37, 1])
    h_dist = gpEuclideanDist(norm_vector[34, 0], norm_vector[34, 1], norm_vector[28, 0], norm_vector[28, 1])
    i_dist = gpEuclideanDist(norm_vector[31, 0], norm_vector[31, 1], norm_vector[25, 0], norm_vector[25, 1])

    sl_dist = gpEuclideanDist(norm_vector[29, 0], norm_vector[29, 1], norm_vector[39, 0], norm_vector[39, 1])
    su_dist = gpEuclideanDist(norm_vector[30, 0], norm_vector[30, 1], norm_vector[38, 0], norm_vector[38, 1])
    tl_dist = gpEuclideanDist(norm_vector[33, 0], norm_vector[33, 1], norm_vector[35, 0], norm_vector[35, 1])
    tu_dist = gpEuclideanDist(norm_vector[32, 0], norm_vector[32, 1], norm_vector[36, 0], norm_vector[36, 1])

    ml_dist = gpEuclideanDist(norm_vector[28, 0], norm_vector[28, 1], norm_vector[29, 0], norm_vector[29, 1])
    ml_dist += gpEuclideanDist(norm_vector[30, 0], norm_vector[30, 1], norm_vector[29, 0], norm_vector[29, 1])
    ml_dist += gpEuclideanDist(norm_vector[30, 0], norm_vector[30, 1], norm_vector[31, 0], norm_vector[31, 1])
    ml_dist += gpEuclideanDist(norm_vector[37, 0], norm_vector[37, 1], norm_vector[31, 0], norm_vector[31, 1])
    ml_dist += gpEuclideanDist(norm_vector[38, 0], norm_vector[38, 1], norm_vector[37, 0], norm_vector[37, 1])
    ml_dist += gpEuclideanDist(norm_vector[38, 0], norm_vector[38, 1], norm_vector[39, 0], norm_vector[39, 1])
    ml_dist += gpEuclideanDist(norm_vector[28, 0], norm_vector[28, 1], norm_vector[39, 0], norm_vector[39, 1])

    mr_dist = gpEuclideanDist(norm_vector[32, 0], norm_vector[32, 1], norm_vector[31, 0], norm_vector[31, 1])
    mr_dist += gpEuclideanDist(norm_vector[32, 0], norm_vector[32, 1], norm_vector[33, 0], norm_vector[33, 1])
    mr_dist += gpEuclideanDist(norm_vector[34, 0], norm_vector[34, 1], norm_vector[33, 0], norm_vector[33, 1])
    mr_dist += gpEuclideanDist(norm_vector[34, 0], norm_vector[34, 1], norm_vector[35, 0], norm_vector[35, 1])
    mr_dist += gpEuclideanDist(norm_vector[36, 0], norm_vector[36, 1], norm_vector[35, 0], norm_vector[35, 1])
    mr_dist += gpEuclideanDist(norm_vector[36, 0], norm_vector[36, 1], norm_vector[37, 0], norm_vector[37, 1])
    mr_dist += gpEuclideanDist(norm_vector[37, 0], norm_vector[37, 1], norm_vector[31, 0], norm_vector[31, 1])

    if b_dist > c_dist:
        face_smr_data[18] = b_dist / a_dist
    else:
        face_smr_data[18] = c_dist / a_dist

    if sl_dist > tl_dist:
        face_smr_data[19] = sl_dist / h_dist
    else:
        face_smr_data[19] = tl_dist / h_dist

    if su_dist > tu_dist:
        face_smr_data[20] = su_dist / h_dist
    else:
        face_smr_data[20] = tu_dist / h_dist

    if ml_dist > mr_dist:
        face_smr_data[21] = ml_dist / h_dist
    else:
        face_smr_data[21] = mr_dist / h_dist

    if d_dist > g_dist:
        face_smr_data[25] = d_dist / a_dist
    else:
        face_smr_data[25] = g_dist / a_dist

    if e_dist > f_dist:
        face_smr_data[26] = e_dist / a_dist
    else:
        face_smr_data[26] = f_dist / a_dist

    face_smr_data[28] = i_dist / a_dist

def gpGetSMR(norm_vector, face_smr_data):
    """
    Computes standard Symmetry Measure Ratios (SMR) from extracted landmarks.

    This function analyzes left-to-right asymmetries across multiple facial
    components (angles, eye heights, distances from center) and updates the
    SMR feature vector `face_smr_data` in place.

    Args:
        norm_vector (np.ndarray): A (51, 2) array of normalized extracted landmarks.
        face_smr_data (np.ndarray): A 2D column vector storing the final SMR metrics.
                                    (Modified in place).
    """
    b_dist = gpEuclideanDist(norm_vector[49, 0], norm_vector[49, 1], norm_vector[48, 0], norm_vector[48, 1])
    e_dist = gpEuclideanDist(norm_vector[37, 0], norm_vector[37, 1], norm_vector[10, 0], norm_vector[10, 1])
    f_dist = gpEuclideanDist(norm_vector[19, 0], norm_vector[19, 1], norm_vector[37, 0], norm_vector[37, 1])
    gl_dist = gpEuclideanDist(norm_vector[13, 0], norm_vector[13, 1], norm_vector[10, 0], norm_vector[10, 1])
    gr_dist = gpEuclideanDist(norm_vector[19, 0], norm_vector[19, 1], norm_vector[16, 0], norm_vector[16, 1])
    j_dist = gpEuclideanDist(norm_vector[10, 0], norm_vector[10, 1], norm_vector[48, 0], norm_vector[48, 1])
    k_dist = gpEuclideanDist(norm_vector[49, 0], norm_vector[49, 1], norm_vector[19, 0], norm_vector[19, 1])
    l_dist = gpEuclideanDist(norm_vector[50, 0], norm_vector[50, 1], norm_vector[37, 0], norm_vector[37, 1])
    m_dist = gpEuclideanDist(norm_vector[10, 0], norm_vector[10, 1], norm_vector[23, 0], norm_vector[23, 1])
    n_dist = gpEuclideanDist(norm_vector[19, 0], norm_vector[19, 1], norm_vector[27, 0], norm_vector[27, 1])
    o_dist = gpEuclideanDist(norm_vector[23, 0], norm_vector[23, 1], norm_vector[37, 0], norm_vector[37, 1])
    p_dist = gpEuclideanDist(norm_vector[27, 0], norm_vector[27, 1], norm_vector[37, 0], norm_vector[37, 1])

    ql_dist = gpEuclideanDist(norm_vector[11, 0], norm_vector[11, 1], norm_vector[15, 0], norm_vector[15, 1])
    qr_dist = gpEuclideanDist(norm_vector[12, 0], norm_vector[12, 1], norm_vector[14, 0], norm_vector[14, 1])
    q_avg = (abs(norm_vector[11, 1] - norm_vector[15, 1]) + abs(norm_vector[12, 1] - norm_vector[14, 1])) / 2.0
    if q_avg <= MIN_EVAL: q_avg = 0

    rl_dist = gpEuclideanDist(norm_vector[17, 0], norm_vector[17, 1], norm_vector[21, 0], norm_vector[21, 1])
    rr_dist = gpEuclideanDist(norm_vector[18, 0], norm_vector[18, 1], norm_vector[20, 0], norm_vector[20, 1])
    r_avg = (abs(norm_vector[17, 1] - norm_vector[21, 1]) + abs(norm_vector[18, 1] - norm_vector[20, 1])) / 2.0
    if r_avg <= MIN_EVAL: r_avg = 0

    sl_dist = gpEuclideanDist(norm_vector[29, 0], norm_vector[29, 1], norm_vector[39, 0], norm_vector[39, 1])
    su_dist = gpEuclideanDist(norm_vector[30, 0], norm_vector[30, 1], norm_vector[38, 0], norm_vector[38, 1])
    tl_dist = gpEuclideanDist(norm_vector[33, 0], norm_vector[33, 1], norm_vector[35, 0], norm_vector[35, 1])
    tu_dist = gpEuclideanDist(norm_vector[32, 0], norm_vector[32, 1], norm_vector[36, 0], norm_vector[36, 1])

    i_dist = norm_vector[0, 1] + norm_vector[1, 1] + norm_vector[2, 1] + norm_vector[3, 1] + norm_vector[4, 1]
    i_dist /= 5
    a_dist = norm_vector[5, 1] + norm_vector[6, 1] + norm_vector[7, 1] + norm_vector[8, 1] + norm_vector[9, 1]
    a_dist /= 5

    c_dist = abs(norm_vector[9, 1] - norm_vector[0, 1]) / max(abs(norm_vector[9, 0] - norm_vector[0, 0]), 1e-6)
    d_dist = abs(norm_vector[7, 1] - norm_vector[2, 1]) / max(abs(norm_vector[7, 0] - norm_vector[2, 0]), 1e-6)
    h_dist = abs(norm_vector[5, 1] - norm_vector[4, 1]) / max(abs(norm_vector[5, 0] - norm_vector[4, 0]), 1e-6)

    delta_x = norm_vector[0, 0] - norm_vector[9, 0]
    delta_y = norm_vector[0, 1] - norm_vector[9, 1]
    if not math.isnan(math.atan2(delta_y, delta_x)):
        face_smr_data[0] = abs(math.atan2(delta_y, delta_x) * 180 / math.pi)

    delta_x = norm_vector[2, 0] - norm_vector[7, 0]
    delta_y = norm_vector[2, 1] - norm_vector[7, 1]
    if not math.isnan(math.atan2(delta_y, delta_x)):
        face_smr_data[1] = abs(math.atan2(delta_y, delta_x) * 180 / math.pi)

    delta_x = norm_vector[4, 0] - norm_vector[5, 0]
    delta_y = norm_vector[4, 1] - norm_vector[5, 1]
    if not math.isnan(math.atan2(delta_y, delta_x)):
        face_smr_data[2] = abs(math.atan2(delta_y, delta_x) * 180 / math.pi)

    if a_dist < i_dist:
        face_smr_data[3] = a_dist / i_dist
    else:
        face_smr_data[3] = i_dist / a_dist

    face_smr_data[4] = c_dist
    face_smr_data[5] = d_dist
    face_smr_data[6] = h_dist

    delta_x = norm_vector[19, 0] - norm_vector[10, 0]
    delta_y = norm_vector[19, 1] - norm_vector[10, 1]
    if not math.isnan(math.atan2(delta_y, delta_x)):
        face_smr_data[7] = abs(math.atan2(delta_y, delta_x) * 180 / math.pi)

    if gl_dist < gr_dist:
        face_smr_data[8] = gl_dist / gr_dist
    else:
        face_smr_data[8] = gr_dist / gl_dist

    if j_dist < k_dist:
        face_smr_data[9] = j_dist / k_dist
    else:
        face_smr_data[9] = k_dist / j_dist

    if m_dist < n_dist:
        face_smr_data[10] = m_dist / n_dist
    else:
        face_smr_data[10] = n_dist / m_dist

    if q_avg < r_avg and r_avg != 0:
        face_smr_data[11] = q_avg / r_avg
    elif q_avg != 0:
        face_smr_data[11] = r_avg / q_avg
    else:
        face_smr_data[11] = 0

    if ql_dist < rr_dist:
        face_smr_data[12] = ql_dist / rr_dist
    else:
        face_smr_data[12] = rr_dist / ql_dist

    if qr_dist < rl_dist:
        face_smr_data[13] = qr_dist / rl_dist
    else:
        face_smr_data[13] = rl_dist / qr_dist

    delta_x = norm_vector[34, 0] - norm_vector[28, 0]
    delta_y = norm_vector[34, 1] - norm_vector[28, 1]
    if not math.isnan(math.atan2(delta_y, delta_x)):
        face_smr_data[14] = abs(math.atan2(delta_y, delta_x) * 180 / math.pi)

    if e_dist < f_dist:
        face_smr_data[15] = e_dist / f_dist
    else:
        face_smr_data[15] = f_dist / e_dist

    if sl_dist < tl_dist:
        face_smr_data[16] = sl_dist / tl_dist
    else:
        face_smr_data[16] = tl_dist / sl_dist

    if su_dist < tu_dist:
        face_smr_data[17] = su_dist / tu_dist
    else:
        face_smr_data[17] = tu_dist / su_dist

    delta_x = norm_vector[27, 0] - norm_vector[23, 0]
    delta_y = norm_vector[27, 1] - norm_vector[23, 1]
    if not math.isnan(math.atan2(delta_y, delta_x)):
        face_smr_data[22] = abs(math.atan2(delta_y, delta_x) * 180 / math.pi)

    delta_x = norm_vector[37, 0] - norm_vector[22, 0]
    delta_y = norm_vector[37, 1] - norm_vector[22, 1]
    if not math.isnan(math.atan2(delta_y, delta_x)):
        face_smr_data[23] = abs(math.atan2(delta_y, delta_x) * 180 / math.pi)

    if o_dist < p_dist:
        face_smr_data[24] = o_dist / p_dist
    else:
        face_smr_data[24] = p_dist / o_dist

    if b_dist != 0:
        face_smr_data[27] = l_dist / b_dist

def gpComputeSRM(num_test_images, test_image_root):
    """
    Main extraction loop that reads images/landmarks, normalizes them, and extracts SMRs.

    For each valid subject file:
      1. Loads the image and landmark coordinates.
      2. Aligns and rotates the face.
      3. Plots 51 selected keypoints on the image and saves it.
      4. Calculates SMR/FMM features and exports them to a CSV.

    Args:
        num_test_images (int): The total number of subject images to process.
        test_image_root (str): The common directory/filename prefix for test images.
        init_img (int): The starting index for sequential image numbering.
    """
    no_img_found = 0
    face_found = 0

    for ik in range(num_test_images):
        if ik < 1000: cbuffs = f"{ik:03d}"
        else: cbuffs = f"{ik:04d}"

        img_name = test_image_root + cbuffs + image_ext
        file_name = test_image_root + cbuffs + csv_name

        print(file_name)
        data_keypts = readCSV(file_name)
        img_color = cv2.imread(img_name)

        if data_keypts.size == 0 or img_color is None:
            no_img_found += 1
            print("no data found...")
            smr_vector = np.array([[-1.0]])
            file_name_out = test_results_path + sm_feat + image_root + cbuffs + csv_name
            writeCSV(file_name_out, smr_vector)
        else:
            img_lbf, mat_landmarks = gpFaceReg(img_color, data_keypts)
            for jk in range(FIRST_LANDMARK, data_keypts.shape[0]):
                cv2.circle(img_lbf, (int(mat_landmarks[jk, 0]), int(mat_landmarks[jk, 1])), 6, (0, 0, 255), -1)

            file_name_out = test_results_path + image_root + cbuffs + "_facereg_keypts_out" + image_ext
            cv2.imwrite(file_name_out, img_lbf)
            face_found += 1

            coord_vector = gpPtsExt(mat_landmarks)
            smr_vector = np.zeros(TOTAL_SMR, dtype=np.float32)
            gpGetSMR(coord_vector, smr_vector)
            gpGetFMM(coord_vector, smr_vector)

            file_name_out = test_results_path + sm_feat + image_root + cbuffs + csv_name
            writeCSV(file_name_out, smr_vector)

    print(f"Number of images: {num_test_images}")
    print(f"Number of files not found: {no_img_found}")
    print(f"Number of files found: {face_found}")

if __name__ == '__main__':
    # Setting the paths
    test_image_path = os.getcwd() + os.path.sep + test_image_path + os.path.sep
    test_results_path = os.getcwd() + os.path.sep + test_results_path + os.path.sep
    test_image_root = test_image_path + image_root

    # Read the number of files in test_image_root
    jpg_files = glob.glob(test_image_path + '*' + image_ext)
    num_subjects = len(jpg_files)

    # Process the test images
    print(f"Processing images from {test_image_root}")
    gpComputeSRM(num_subjects, test_image_root)

    # Create the ARFF file (example)    
    print(f"Preparing ARFF dataset...")
    # Example labels for the 5 subjects, vec_labes must be adjust according to the dataset    
    vec_labels = [[0], [0], [1], [0], [1]]
    gp_data_prepare_to_arff(test_results_path, vec_labels, cvs_name_dataset, arff_name_dataset)
    print("Done.")
