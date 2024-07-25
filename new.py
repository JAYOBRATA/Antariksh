import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import rasterio
import pandas as pd

class CraterLocatorApp(QtWidgets.QWidget):
    def _init_(self):
        super()._init_()
        self.initUI()
        
    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Crater Locator')

        self.image1_label = QtWidgets.QLabel(self)
        self.image1_label.setGeometry(50, 50, 300, 300)
        self.image1_label.setFrameShape(QtWidgets.QFrame.Box)
        
        self.image2_label = QtWidgets.QLabel(self)
        self.image2_label.setGeometry(450, 50, 300, 300)
        self.image2_label.setFrameShape(QtWidgets.QFrame.Box)

        self.load_button = QtWidgets.QPushButton('Load Images', self)
        self.load_button.setGeometry(150, 400, 100, 30)
        self.load_button.clicked.connect(self.load_images)

        self.match_button = QtWidgets.QPushButton('Match', self)
        self.match_button.setGeometry(550, 400, 100, 30)
        self.match_button.clicked.connect(self.match_images)

        self.result_label = QtWidgets.QLabel(self)
        self.result_label.setGeometry(150, 450, 500, 30)
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        
        self.crater_catalog = self.load_crater_catalog()
        
    def load_images(self):
        options = QtWidgets.QFileDialog.Options()
        file1, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Chandrayaan-2 TMC Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        file2, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load LRO WAC Mosaic", "", "GeoTIFF Files (.tif *.tiff);;Image Files (.png *.jpg *.bmp)", options=options)
        
        if file1 and file2:
            self.image1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
            self.image2 = rasterio.open(file2)

            self.image2_data = self.image2.read(1)
            
            self.display_image(self.image1, self.image1_label)
            self.display_image(self.image2_data, self.image2_label)
    
    def display_image(self, image, label):
        height, width = image.shape
        bytesPerLine = width
        qimage = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio))
        
    def resample_image(self, image, target_resolution):
        scale_factor = 100 / 5  # From 5m to 100m
        rescaled_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        return rescaled_image

    def detect_and_match_features(self, image1, image2):
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        return keypoints1, keypoints2, matches

    def estimate_transformation(self, keypoints1, keypoints2, matches):
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M

    def warp_image(self, image, M, shape):
        warped_image = cv2.warpPerspective(image, M, (shape[1], shape[0]))
        return warped_image

    def template_matching(self, image, template):
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        return max_loc, max_val

    def match_images(self):
        res_image1 = self.resample_image(self.image1, 100)
        res_image2 = self.resample_image(self.image2_data, 100)
        
        keypoints1, keypoints2, matches = self.detect_and_match_features(res_image1, res_image2)
        M = self.estimate_transformation(keypoints1, keypoints2, matches)
        warped_image = self.warp_image(res_image1, M, res_image2.shape)
        
        max_loc, max_val = self.template_matching(warped_image, res_image2)
        
        result_text = f"Match Location: {max_loc}, Match Value: {max_val:.2f}"
        self.result_label.setText(result_text)
        
        result_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(result_image, max_loc, (max_loc[0] + res_image2.shape[1], max_loc[1] + res_image2.shape[0]), (0, 0, 255), 2)
        self.display_image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), self.image1_label)

        lat, lon = self.get_geo_coordinates(max_loc)
        crater_name = self.get_crater_name(lat, lon)
        if crater_name:
            self.result_label.setText(f"Crater: {crater_name}, Location: ({lat}, {lon}), Match Value: {max_val:.2f}")
        else:
            self.result_label.setText(f"Location: ({lat}, {lon}), Match Value: {max_val:.2f}")

    def get_geo_coordinates(self, pixel_coords):
        col, row = pixel_coords
        lon, lat = self.image2.xy(row, col)
        return lat, lon

    def load_crater_catalog(self):
        # Placeholder for loading crater catalog
        # Here, you can load a CSV or database of craters with their names and coordinates
        # For demonstration, let's create a simple DataFrame
        data = {
            'Name': ['Tycho', 'Copernicus', 'Aristarchus'],
            'Latitude': [-43.3, 9.62, 23.7],
            'Longitude': [348.8, 339.92, 312.7]
        }
        return pd.DataFrame(data)
    
    def get_crater_name(self, lat, lon):
        # Find the nearest crater in the catalog
        distances = np.sqrt((self.crater_catalog['Latitude'] - lat)*2 + (self.crater_catalog['Longitude'] - lon)*2)
        min_idx = distances.idxmin()
        return self.crater_catalog.loc[min_idx, 'Name']

if __name__:
    app = QtWidgets.QApplication(sys.argv)
    ex = CraterLocatorApp()
    ex.show()
    sys.exit(app.exec_())