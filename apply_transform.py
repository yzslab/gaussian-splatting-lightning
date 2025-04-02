import csv
import os.path as osp
import numpy as np
import subprocess

output_folder_path = '/media/rishabh/SSD_1/Data/Table_vid_reg/project_2_images'
path_2_trained_GS = '/home/rishabh/projects/gaussian-splatting/output/table_2/point_cloud/iteration_30000/point_cloud.ply'

# Initialize variables to hold matrices and Euler angles
scale = []
global_euler_zyx = []
fine_euler_zyx = []

global_translations = []
fine_translations = []

global_csv = osp.join(output_folder_path, 'global_reg_result.csv')
fine_csv = osp.join(output_folder_path, 'fine_reg_result.csv')

# Open the CSV file for reading
with open(global_csv, mode='r') as file:
    reader = csv.reader(file)
    rows = list(reader)

    # Initialize variables to hold extracted data
    # scale = None
    result_T_m2_m1 = None

    # Iterate over the rows and extract "scale" and "result_T_m2_m1"
    for i, row in enumerate(rows):
        if row and row[0] == "scale":
            # Extract the scale value (next row contains the scalar value)
            scale = float(rows[i + 1][0])  # The scale is the only element in that row
        elif row and row[0] == "result_T_m2_m1":
            # Extract the matrix "result_T_m2_m1"
            matrix = []
            for j in range(1, 5):  # Skip header and read the next 4 rows for the matrix
                matrix.append(list(map(float, rows[i + j])))
            result_T_m2_m1 = np.array(matrix)
        elif row and row[0] == "euler_angles_zyx":
            # Extract the Euler angles (the next row contains the angles)
            global_euler_zyx = list(map(float, rows[i + 1]))
            break  # Once we have found all, exit the loop

# Extract the top-left 3x3 submatrix and the [3,3] element
if result_T_m2_m1 is not None:
    global_translations = result_T_m2_m1[:3,3]

input = path_2_trained_GS
output = input.split('.')[0]+'_scale.ply'

scale_command = 'python gaussian_transform.py {} {} --scale {}'.format(input, output, scale)
scale_result = subprocess.run(scale_command, shell=True, capture_output=True, text=True)
print("STDOUT:", scale_result.stdout)

tx,ty,tz = global_translations
rz,ry,rx = global_euler_zyx

input = output
output = input.split('.')[0]+'_global_reg.ply'

global_reg_command = 'python gaussian_transform.py {} {} --tx {} --ty {} --tz {} --rx {} --ry {} --rz {}'.format(input, output, tx, ty, tz, rx, ry, rz)
global_reg_result = subprocess.run(global_reg_command, shell=True, capture_output=True, text=True)
print("STDOUT:", global_reg_result.stdout)


# Open the CSV file for reading
with open(fine_csv, mode='r') as file:
    reader = csv.reader(file)
    rows = list(reader)

    # Initialize variables to hold extracted data
    # scale = None
    fine_T_m2_m1 = None

    # Iterate over the rows and extract "scale" and "fine_T_m2_m1"
    for i, row in enumerate(rows):
        if row and row[0] == "icp_result.transformation":
            # Extract the matrix "fine_T_m2_m1"
            matrix = []
            for j in range(1, 5):  # Skip header and read the next 4 rows for the matrix
                matrix.append(list(map(float, rows[i + j])))
            fine_T_m2_m1 = np.array(matrix)
        elif row and row[0] == "euler_angles_zyx":
            # Extract the Euler angles (the next row contains the angles)
            fine_euler_zyx = list(map(float, rows[i + 1]))
            break  # Once we have found all, exit the loop

# Extract the top-left 3x3 submatrix and the [3,3] element
if fine_T_m2_m1 is not None:
    fine_translations = fine_T_m2_m1[:3,3]

tx_fine,ty_fine,tz_fine = fine_translations
rz_fine,ry_fine,rx_fine = fine_euler_zyx

input = output
output = input.split('.')[0]+'_icp.ply'

fine_reg_command = 'python gaussian_transform.py {} {} --tx {} --ty {} --tz {} --rx {} --ry {} --rz {}'.format(input, output, tx_fine, ty_fine, tz_fine, rx_fine, ry_fine, rz_fine)
fine_reg_result = subprocess.run(fine_reg_command, shell=True, capture_output=True, text=True)
print("STDOUT:", fine_reg_result.stdout)

print("Yay")