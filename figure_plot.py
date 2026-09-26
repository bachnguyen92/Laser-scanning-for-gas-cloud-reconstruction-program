from PyQt5.QtWidgets import QApplication, QMainWindow, QFileSystemModel, QFileDialog, QMessageBox, QWidget
from PyQt5.QtCore import QDir, QModelIndex, Qt, QStandardPaths
import pandas as pd
import os
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # type: ignore
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from New_window import Ui_MainWindow
import support_function as sf
import rec_method as rm


class FigurePlot(QMainWindow):

    def __init__(self, self_instance):
        super(FigurePlot, self).__init__()
        self.ui: Ui_MainWindow = self_instance.ui

        self.ui.spinBox_width.valueChanged.connect(lambda: self.init_window())
        self.ui.spinBox_hight.valueChanged.connect(lambda: self.init_window())
        self.ui.spinBox_x_divisions.valueChanged.connect(lambda: self.init_window())
        self.ui.spinBox_y_divisions.valueChanged.connect(lambda: self.init_window())
        self.ui.spinBox_CloudSize.valueChanged.connect(lambda: self.init_window())
        self.ui.spinBox_positionX.valueChanged.connect(lambda: self.init_window())
        self.ui.spinBox_positionY.valueChanged.connect(lambda: self.init_window())
        self.ui.spinBox_number_laser.valueChanged.connect(lambda: self.init_window())
        self.ui.spinBox_num_ray_laser.valueChanged.connect(lambda: self.init_window())
        self.ui.spinBox_distance_laser.valueChanged.connect(lambda: self.init_window())

        self.ui.radioButton_show.toggled.connect(lambda: self.init_window())

        self.ui.comboBox_method.currentIndexChanged.connect(lambda: self.callback_method_changed())

        self.ui.pushButton_draw.clicked.connect(lambda: self.call_draw())
        self.ui.pushButton_reconstruct.clicked.connect(lambda: self.call_reconstruct())

        self.ui.pushButton_save_graph.clicked.connect(self.save_graph)

        self.ui.pushButton_toggleWidth.clicked.connect(self.toggleWidth)
        self.ui.pushButton_information.clicked.connect(self.show_information)
        # ============================================================ local var
        self.fig_orig, self.axes_orig = plt.subplots()
        self.canvas_orig = FigureCanvas(self.fig_orig)
        self.ui.frame_orig.addWidget(self.canvas_orig)  # add to frame_orig

        self.fig_laser, self.axes_laser = plt.subplots()
        self.canvas_laser = FigureCanvas(self.fig_laser)
        self.ui.frame_laser.addWidget(self.canvas_laser)  # add to frame_laser

        self.fig_rec, self.axes_rec = plt.subplots()
        self.canvas_rec = FigureCanvas(self.fig_rec)
        self.ui.frame_rec.addWidget(self.canvas_rec)  # add to frame_rec
        self.colorbar = self.colorbar = self.fig_rec.colorbar(mappable=None, ax=self.axes_rec)  # taọ đối tượng colorbar
        self.colorbar.set_label('Concentration (ppm)')

        self.total_lasers = 0
        self.current_laser = 0
        self.cell_list = None
        self.laser_rays_parameters = []
        self.w_constant = 50
        self.show = 1 if self.ui.radioButton_show.isChecked() else 0

        self.max_width = self.ui.frame_signal_file.maximumWidth()
        self.max_width_imf = self.ui.frame_information.maximumWidth()
        # ============================================================
        self.init_window()

    def show_information(self):
        # toggle Width
        current_width = self.ui.frame_information.maximumWidth()
        if current_width != 0:
            new_width = 0
            self.ui.pushButton_information.setText("<<")
        else:
            new_width = self.max_width_imf
            self.ui.pushButton_information.setText(">>")
        self.ui.frame_information.setFixedWidth(new_width)

    def toggleWidth(self):
        # toggle Width
        current_width = self.ui.frame_signal_file.maximumWidth()
        if current_width != 0:
            new_width = 0
            self.ui.pushButton_toggleWidth.setText(">>")
        else:
            new_width = self.max_width
            self.ui.pushButton_toggleWidth.setText("<<")

        # set Width of QFrame
        self.ui.frame_signal_file.setFixedWidth(new_width)

    def save_graph(self):
        # Get the filename/path using QFileDialog
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, 'Save Origin gas cloud', '',
                                                   'PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)')

        if file_path:  # If user selected a file
            if file_path.endswith('.png'):
                self.fig_orig.savefig(file_path, format='png')
            elif file_path.endswith('.pdf'):
                self.fig_orig.savefig(file_path, format='pdf')

        # ==================================================================
        file_path, _ = file_dialog.getSaveFileName(self, 'Save Reconstruction process', '',
                                                   'PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)')

        if file_path:  # If user selected a file
            if file_path.endswith('.png'):
                self.fig_laser.savefig(file_path, format='png')
            elif file_path.endswith('.pdf'):
                self.fig_laser.savefig(file_path, format='pdf')

        # ==================================================================
        file_path, _ = file_dialog.getSaveFileName(self, 'Save Reconstruction image', '',
                                                   'PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)')

        if file_path:  # If user selected a file
            if file_path.endswith('.png'):
                self.fig_rec.savefig(file_path, format='png')
            elif file_path.endswith('.pdf'):
                self.fig_rec.savefig(file_path, format='pdf')

    def draw_rectangle_cloud(self):
        # draw_rectangle in origin window
        self.draw_rectangle(self.axes_orig, self.fig_orig, self.canvas_orig)
        self.draw_cloud(self.axes_orig, self.fig_orig, self.canvas_orig)

        # draw_rectangle in reconstruction window
        self.draw_rectangle(self.axes_laser, self.fig_laser, self.canvas_laser, index_cell=0)
        self.draw_cloud(self.axes_laser, self.fig_laser, self.canvas_laser)

    def init_window(self):

        self.draw_rectangle_cloud()

        self.draw_rectangle(self.axes_rec, self.fig_rec, self.canvas_rec, index_cell=0)   # draw rectangle in reconstruction image
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))  # reset colorbar
        self.colorbar.update_normal(sm)

        self.show = 1 if self.ui.radioButton_show.isChecked() else 0    # update show value

        # Set giá trị 0 và ẩn progressBar
        self.ui.progressBar.setValue(0)
        self.ui.pushButton_draw.setEnabled(True)

        self.ui.pushButton_reconstruct.setEnabled(False)
        self.ui.comboBox_method.setEnabled(False)

        self.ui.tabWidget.setCurrentIndex(1)  # hiện tab 2

    def draw_rectangle(self, ax, fig, canvas, index_cell=1):
        ax.clear()

        x_divisions = self.ui.spinBox_x_divisions.value()
        y_divisions = self.ui.spinBox_y_divisions.value()
        height = self.ui.spinBox_hight.value()
        width = self.ui.spinBox_width.value()

        rect = plt.Rectangle((0, 0), width, height, linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
        ax.add_patch(rect)

        # Draw black dividing lines`1
        for i in range(0, x_divisions + 1):
            x = (i * width / x_divisions)
            ax.plot([x, x], [0, height], color='black', linewidth=1, linestyle='--')

        for i in range(0, y_divisions + 1):
            y = (i * height / y_divisions)
            ax.plot([0, width], [y, y], color='black', linewidth=1, linestyle='--')

        # Add index for columns on the first row
        for i in range(0, x_divisions):
            x = (i * width / x_divisions)
            ax.text(x, -1.5, str(i), fontsize=8, ha='left')
        # Add index for rows on the first column
        for i in range(0, y_divisions):
            y = (i * height / y_divisions)
            ax.text(-1, y, str(i), fontsize=8, ha='center')

        # Add index for cell
        if index_cell == 1:
            for i in range(0, y_divisions * x_divisions):
                x_index = (i) % x_divisions
                x = (x_index * (width / x_divisions))
                y = ((y_divisions - ((i) // x_divisions + 1)) * (height / y_divisions))
                ax.text(x, y, str(i + 1), fontsize=8, ha='left')

        ax.autoscale()

        # Adjust the size of the FigureCanvas
        fig.tight_layout()
        # Redraw the canvas
        canvas.draw()

    def draw_cloud_rec(self, X_matrix, ax, fig, canvas):
        x_divisions = self.ui.spinBox_x_divisions.value()
        y_divisions = self.ui.spinBox_y_divisions.value()
        height = self.ui.spinBox_hight.value()
        width = self.ui.spinBox_width.value()

        norm = plt.Normalize(np.min(X_matrix), np.max(X_matrix))
        cmap = plt.get_cmap('viridis')

        # Highlight cells whose coordinates are in the cell_list list
        for i in range(0, y_divisions * x_divisions):
            x = (i) % x_divisions
            y = y_divisions - ((i) // x_divisions + 1)
            x_start = x * (width / x_divisions)
            y_start = y * (height / y_divisions)
            x_end = (x + 1) * (width / x_divisions)
            y_end = (y + 1) * (height / y_divisions)

            # Chuẩn hóa giá trị và lấy màu tương ứng
            value = X_matrix[i]
            color = cmap(norm(value))

            rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, color=color)
            ax.add_patch(rect)

        # add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        self.colorbar.update_normal(sm)

        # Adjust the size of the FigureCanvas
        fig.tight_layout()
        # Redraw the canvas
        canvas.draw()

    def draw_cloud(self, ax, fig, canvas):
        x_divisions = self.ui.spinBox_x_divisions.value()
        y_divisions = self.ui.spinBox_y_divisions.value()
        height = self.ui.spinBox_hight.value()
        width = self.ui.spinBox_width.value()

        width_fild = self.ui.spinBox_CloudSize.value()  # kích thước khối khi

        # kich thước đầu tiên là chiều dài, thứ 2 là rộng của khối khí
        i, j = np.meshgrid(np.arange(width_fild), np.arange(width_fild), indexing='ij')

        x_start = self.ui.spinBox_positionX.value()
        y_start = self.ui.spinBox_positionY.value()
        box2 = (y_start * x_divisions + x_start + 1) + x_divisions * i + j

        box2 = box2.flatten()
        self.cell_list = sf.create_cell_list(box2, x_divisions, y_divisions)

        x_divisions = int(x_divisions)
        y_divisions = int(y_divisions)
        # Highlight cells whose coordinates are in the cell_list list
        for cell in self.cell_list:
            x_start = cell[0] * (width / x_divisions)
            y_start = cell[1] * (height / y_divisions)
            x_end = (cell[0] + 1) * (width / x_divisions)
            y_end = (cell[1] + 1) * (height / y_divisions)
            rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                 color=(255 / 255, 150 / 255, 0 / 255))
            ax.add_patch(rect)

        # Adjust the size of the FigureCanvas
        fig.tight_layout()
        # Redraw the canvas
        canvas.draw()

    def draw_lase_line(self, ax, fig, canvas, laser_pos, alpha, show=0):
        x_divisions = self.ui.spinBox_x_divisions.value()
        y_divisions = self.ui.spinBox_y_divisions.value()
        height = self.ui.spinBox_hight.value()
        width = self.ui.spinBox_width.value()

        intersections = []  # List to store the coordinates of intersection points

        # Draw laser at A position
        # ax.add_patch(plt.Rectangle((a-20, height), 20*2, -15, color=(255/255, 100/255, 100/255)))

        # Calculate and draw intersection points of the line with dividing lines and border
        if alpha == 90:
            # Draw a vertical line at x coordinate 'a'
            if show == 1:
                ax.plot([laser_pos, laser_pos], [height, 0], color=(255 / 255, 0 / 255, 0 / 255), linewidth=1)

            for i in range(0, y_divisions + 1):
                y = (i * height / y_divisions)
                x_intersect = (laser_pos)
                if 0 <= x_intersect <= width:
                    if show == 1:
                        ax.scatter(x_intersect, y, s=20, c=[(0, 0, 255 / 255)])
                    intersections.append((x_intersect, y))
        else:
            m = math.tan(math.radians(alpha))  # Slope of the line
            b = height - m * laser_pos
            x2 = -b / m
            if x2 < 0:
                x2 = 0
                y2 = height - m * laser_pos
            elif x2 > width:
                x2 = width
                y2 = m * width + b
            else:
                y2 = 0
            if show == 1:
                ax.plot([laser_pos, x2], [height, y2], color='r', linewidth=1)

            for i in range(0, x_divisions + 1):
                x = (i * width / x_divisions)
                y_intersect = (m * x + b)
                if 0 <= y_intersect <= height:
                    if show == 1:
                        ax.scatter(x, y_intersect, s=20, c=[(0, 0, 255 / 255)])
                    intersections.append((x, y_intersect))

            for i in range(0, y_divisions + 1):
                y = (i * height / y_divisions)
                x_intersect = ((y - b) / m)
                if 0 <= x_intersect <= width:
                    if show == 1:
                        ax.scatter(x_intersect, y, s=20, c=[(0, 0, 255 / 255)])
                    intersections.append((x_intersect, y))
        intersections = sorted(intersections, key=lambda x: x[0])

        # Remove duplicate points in the intersections list
        intersections_round = np.round(intersections, 5)
        unique_intersections = [intersections[0]]
        for i in range(1, len(intersections)):
            if not np.array_equal(intersections_round[i], intersections_round[i - 1]):
                unique_intersections.append(intersections[i])

        # Adjust the size of the FigureCanvas
        fig.tight_layout()
        # Redraw the canvas
        canvas.draw()

        return unique_intersections

    def laser_scan(self):
        x_divisions = self.ui.spinBox_x_divisions.value()
        y_divisions = self.ui.spinBox_y_divisions.value()
        height = self.ui.spinBox_hight.value()
        width = self.ui.spinBox_width.value()

        num_ray_laser = self.ui.spinBox_num_ray_laser.value()
        number_laser = self.ui.spinBox_number_laser.value()
        distance_laser = self.ui.spinBox_distance_laser.value()
        distance_laser = distance_laser * 2

        step = width / (2 * x_divisions)  # 1/2 kích thước của ô

        first_position = (width - (number_laser - 1) * distance_laser * step) / 2
        end_position = width - first_position

        # Cập nhật tiến trình trong progressBar
        self.start_progressBar()
        # ====================================================    chiếu
        for laser_pos in np.arange(width / (2 * x_divisions), width, width / x_divisions):
            alpha = 90
            intersections = self.draw_lase_line(self.axes_laser, self.fig_laser, self.canvas_laser, laser_pos, alpha,
                                                show=self.show)
            laser_ray_parameters = sf.gas_concentration(intersections, width, height, x_divisions, y_divisions,
                                                        self.cell_list)
            self.laser_rays_parameters.append(laser_ray_parameters)

            # start_progressBar
            self.update_progressBar()

        for laser_pos in np.linspace(first_position, end_position, number_laser):
            start_angle, end_angle = sf.compute_angle(laser_pos, width, height, x_divisions)
            for alpha in np.linspace(start_angle, end_angle, num_ray_laser):
                intersections = self.draw_lase_line(self.axes_laser, self.fig_laser, self.canvas_laser, laser_pos,
                                                    alpha,
                                                    show=self.show)
                laser_ray_parameters = sf.gas_concentration(intersections, width, height, x_divisions, y_divisions,
                                                            self.cell_list)
                self.laser_rays_parameters.append(laser_ray_parameters)

                # Cập nhật tiến trình trong progressBar
                self.update_progressBar()

    def start_progressBar(self):
        num_ray_laser = self.ui.spinBox_num_ray_laser.value()
        number_laser = self.ui.spinBox_number_laser.value()
        x_divisions = self.ui.spinBox_x_divisions.value()
        self.ui.progressBar.show()
        self.total_lasers = x_divisions + number_laser * num_ray_laser
        self.current_laser = 0

    def update_progressBar(self):
        self.current_laser = self.current_laser + 1
        progress = int(self.current_laser / self.total_lasers * 100)
        self.ui.progressBar.setValue(progress)

    # ============================================================ call
    def call_draw(self):
        self.ui.pushButton_draw.setEnabled(False)

        self.ui.pushButton_reconstruct.setEnabled(True)
        self.ui.comboBox_method.setEnabled(True)

        self.ui.tabWidget.setCurrentIndex(1)  # hiện tab 2

        self.laser_scan()

    def call_reconstruct(self):
        self.ui.pushButton_reconstruct.setEnabled(False)
        self.ui.tabWidget.setCurrentIndex(2)  # hiện tab 2

        x_divisions = self.ui.spinBox_x_divisions.value()
        y_divisions = self.ui.spinBox_y_divisions.value()

        Y_matrix, L_matrix, V_matrix = sf.creat_YLVmatrix(self.laser_rays_parameters, x_divisions, y_divisions)

        num_iterations = None
        X_matrix = None

        start_time = time.time()
        if self.ui.comboBox_method.currentText() == "VM_SART":
            X_matrix, num_iterations = rm.VMR_SART(L_matrix, Y_matrix, V_matrix, w=self.w_constant)

        if self.ui.comboBox_method.currentText() == "SART":
            X_matrix, num_iterations = rm.SART(L_matrix, Y_matrix, V_matrix, w=self.w_constant)

        if self.ui.comboBox_method.currentText() == "Tikhonov":
            X_matrix, num_iterations = rm.Tikhonov(L_matrix, Y_matrix, V_matrix, w=self.w_constant)

        if self.ui.comboBox_method.currentText() == "LSQR":
            X_matrix, num_iterations = rm.LSQR(L_matrix, Y_matrix, V_matrix, w=self.w_constant)

        if self.ui.comboBox_method.currentText() == "NNLS":
            X_matrix, num_iterations = rm.NNLS(L_matrix, Y_matrix, V_matrix, w=self.w_constant)

        stop_time = time.time()
        time_comp = stop_time - start_time

        X_matrix[X_matrix < 0] = 0

        self.draw_cloud_rec(X_matrix, self.axes_rec, self.fig_rec, self.canvas_rec)

        X_origin = np.zeros(x_divisions * y_divisions)
        for cell in self.cell_list:
            x = cell[0]
            y = cell[1]
            X_origin[int((y_divisions - y - 1) * x_divisions + x)] = 30

        self.ui.nrmsd.setText(f"{np.round(rm.nrmsd(X_origin, X_matrix), 3)}")
        self.ui.naaad.setText(f"{np.round(rm.naaad(X_origin, X_matrix), 3)}")
        self.ui.epsilon.setText(f"{np.round(rm.epsilon(X_origin, X_matrix), 3)}")
        self.ui.time_comp.setText(f"{np.round(time_comp, 3)}")

    # ============================================================ callback_update

    def callback_method_changed(self):
        self.ui.pushButton_reconstruct.setEnabled(True)

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))    # reset colorbar
        self.colorbar.update_normal(sm)

        self.draw_rectangle(self.axes_rec, self.fig_rec, self.canvas_rec, index_cell=0)
