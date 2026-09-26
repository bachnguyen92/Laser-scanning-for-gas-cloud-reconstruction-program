import numpy as np
import math
from PIL import Image
from io import BytesIO

from file_var import ClassLaserRayParams

def fig_to_array(fig):
    """Chuyển Figure thành mảng ảnh"""
    buf = BytesIO()
    fig.savefig(buf, format='png')  # Lưu Figure vào bộ nhớ tạm
    buf.seek(0)
    image = Image.open(buf)
    return np.asarray(image)

def gas_concentration(intersections, width, height, x_divisions, y_divisions, cell_list):
    if not intersections:
        return

    # Calculate cell index for each intersection point
    laser_ray_parameters = ClassLaserRayParams()
    for i in range(len(intersections) - 1):
        mid_x = (intersections[i][0] + intersections[i + 1][0]) / 2
        mid_y = ((intersections[i][1] + intersections[i + 1][1]) / 2)
        cell_x = int(mid_x // (width / x_divisions))
        cell_y = int(mid_y // (height / y_divisions))
        if cell_x < x_divisions and cell_y < y_divisions:
            laser_ray_parameters.index_plength.append((cell_x, cell_y))

    # Calculate distance between two points
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # Distance between consecutive points
    for i in range(len(intersections) - 1):
        dist = distance(intersections[i], intersections[i + 1])
        laser_ray_parameters.value_plength.append(dist)
        # print(f"Distance between point {i+1} and point {i+2} in cell {path_length.index[i]}: {dist}")

    if not cell_list:
        return

    # methane concentration in cell
    cell_cons = np.zeros(x_divisions * y_divisions + 1)
    for cell in cell_list:
        x = cell[0]
        y = cell[1]
        cell_cons[int((y_divisions - y - 1) * x_divisions + x + 1)] = 30

    # Print path-integral concentration
    PIC = 0
    for i, dist in enumerate(laser_ray_parameters.value_plength):
        y = laser_ray_parameters.index_plength[i][1]
        x = laser_ray_parameters.index_plength[i][0]
        PIC = PIC + dist * cell_cons[int((y_divisions - y - 1) * x_divisions + x + 1)]
    laser_ray_parameters.integral_concentration = PIC

    return laser_ray_parameters


def create_cell_list(box, x_divisions, y_divisions):
    cell_list_pos = []
    for cell_num in box:
        x = (cell_num - 1) % x_divisions
        y = (y_divisions - ((cell_num - 1) // x_divisions + 1))
        cell_list_pos.append((x, y))
    return cell_list_pos


def compute_angle(laser_position, width, height, x_divisions):
    if laser_position < width / (2 * x_divisions) or laser_position > (2 * x_divisions - 1) * width / (
            2 * x_divisions):
        start_angle = 45
        end_angle = 45
    else:
        if laser_position == width / (2 * x_divisions):
            start_angle = 90
        else:
            start_angle = np.degrees(
                np.arctan(height / (laser_position - width / (2 * x_divisions))))  # góc bắt đầu
        if laser_position == width - width / (2 * x_divisions):
            end_angle = 90
        else:
            end_angle = 180 - np.degrees(
                np.arctan(height / (width - laser_position - width / (2 * x_divisions))))

    return start_angle, end_angle


def creat_YLVmatrix(laser_rays_parameters, x_divisions, y_divisions):
    num_equations = len(laser_rays_parameters)
    num_cell = x_divisions * y_divisions
    L_matrix = np.zeros((num_equations, num_cell))  # weight coefficients matrix or forward projection
    Y_matrix = np.zeros(num_equations)  # path-integral concentrations
    V_matrix = np.zeros((x_divisions, y_divisions))

    # weight coefficients matrix or projection matrix
    for index, laser_ray_parameters in enumerate(laser_rays_parameters):
        for i, cell_index in enumerate(laser_ray_parameters.index_plength):
            x = cell_index[0]
            y = cell_index[1]
            second_index = int((y_divisions - y - 1) * x_divisions + x)
            L_matrix[index][second_index] = laser_ray_parameters.value_plength[i]
            V_matrix[x][y] = V_matrix[x][y] + 1
            Y_matrix[index] = laser_ray_parameters.integral_concentration

    V_matrix = np.flipud(np.transpose(V_matrix))

    return Y_matrix, L_matrix, V_matrix

