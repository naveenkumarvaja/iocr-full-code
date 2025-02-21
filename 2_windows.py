import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFrame,
    QPushButton, QCheckBox, QComboBox, QMessageBox, QFileDialog, QMessageBox, QInputDialog, QLineEdit, QGridLayout, QSizePolicy, QListWidget, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem

)

from PyQt5.QtGui import QImage, QPainter
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont,QImage

from PyQt5.QtCore import QRectF, Qt


import joblib
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np

#import tensorflow as tf
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
#from PySide6.QtWidgets import QMessageBox
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage.feature import graycomatrix, graycoprops

from PIL import Image

import matplotlib.pyplot as plt
plt.ioff()


from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QDialog, QVBoxLayout, QLabel
import platform

# Dummy database for login credentials
USER_CREDENTIALS = {
    "naveen": "1234",
    "ramesh": "123",
    "john": "4321"
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oil Palm Bunch Grader")
        self.setWindowIcon(QIcon("logo.png"))
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: white;")
        self.second_window = None
        

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QVBoxLayout()

        
        # HEADER SECTION
        header_layout = QHBoxLayout()

        # Left Logo
        left_logo = QLabel()
        left_logo.setPixmap(QPixmap("logo.png").scaled(150, 150, Qt.KeepAspectRatio))

        # Title
        title = QLabel("Oil Palm Bunch Grader")
        title.setFont(QFont("Arial", 38, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: purple;")

        # Right Logo
        right_logo = QLabel()
        right_logo.setPixmap(QPixmap("icar_logo.png").scaled(150, 150, Qt.KeepAspectRatio))

        # Add widgets in proper order
        header_layout.addWidget(left_logo)
        header_layout.addStretch()  # Adds spacing between left_logo and title
        header_layout.addWidget(title, alignment=Qt.AlignCenter)
        header_layout.addStretch()  # Adds spacing between title and right_logo
        header_layout.addWidget(right_logo)


        # IMAGE DISPLAY
        img_label = QLabel()
        img_label.setPixmap(QPixmap("img4.png").scaled(600, 600, Qt.KeepAspectRatio))
        img_label.setAlignment(Qt.AlignCenter)

        # LOGIN SECTION (Centered)
        login_container = QVBoxLayout()
        login_container.setAlignment(Qt.AlignCenter)
        
        username_label = QLabel("User Name:")
        username_label.setFont(QFont("Arial", 15, QFont.Bold))
        self.username_input = QLineEdit()
        self.username_input.setFixedSize(200, 30)
        self.username_input.setStyleSheet("border: 1px solid black; background-color: white;")

        password_label = QLabel("Password:")
        password_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.password_input = QLineEdit()
        self.password_input.setFixedSize(200, 30)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("border: 1px solid black; background-color: white;")

        login_button = QPushButton("->Sign In")
        login_button.setStyleSheet("background-color: white; padding: 5px; font-size: 14px;")
        login_button.clicked.connect(self.validate_login)

        

        # Create a layout to center the login form
        login_form_layout = QVBoxLayout()
        login_form_layout.setAlignment(Qt.AlignCenter)
        login_form_layout.addWidget(username_label, alignment=Qt.AlignCenter)
        login_form_layout.addWidget(self.username_input, alignment=Qt.AlignCenter)
        login_form_layout.addWidget(password_label, alignment=Qt.AlignCenter)
        login_form_layout.addWidget(self.password_input, alignment=Qt.AlignCenter)
        login_form_layout.addWidget(login_button, alignment=Qt.AlignCenter)
        

        login_container.addLayout(login_form_layout)

        # FOOTER SECTION
        footer_label = QLabel(
            "<span style='color: red;'>Developed by<span style='color: blue;'> S. Shivashankar, K. Suresh and P. Anitha</span><br>"
            "<span style='color: navy;'>ICAR - Indian Institute of Oil Palm Research, Pedavegi, Andhra Pradesh</span><br>"
            "<span style='color: orange;'><i>Financial Support: ICAR & DAFW, Govt of India</i></span><br>"
            "<span style='color: deeppink;'><i>Disclaimer: No liability what so ever is accepted for the use of this application</i></span><br>"
           
        )
        footer_label.setFont(QFont("Arial", 15, QFont.Bold))
        footer_label.setStyleSheet("color: blue;")
        footer_label.setAlignment(Qt.AlignCenter)

        # Add all layouts to the main layout
        main_layout.addLayout(header_layout)
        main_layout.addWidget(img_label)
        main_layout.addLayout(login_container)
        main_layout.addWidget(footer_label)

        central_widget.setLayout(main_layout)

    def validate_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            self.open_second_window()
        else:
            error_label = QLabel("Wrong details, please try again!")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            error_label.setAlignment(Qt.AlignCenter)
            self.sender().parent().layout().addWidget(error_label)

    def open_second_window(self):
        if self.second_window is None:  # Ensure only one instance
            self.second_window = MultiCameraApp()
        self.second_window.show()  # Show the second window
        self.close()  # Close the first window

class  MultiCameraApp(QMainWindow):  
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Second Window")
        self.setGeometry(300, 300, 200, 100)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("This is the second window"))
        self.setLayout(layout)
        
#class MultiCameraApp(QMainWindow):
    #def __init__(self):
        super().__init__()
        self.setWindowTitle("Oil Palm Bunch Grader")
        self.setWindowIcon(QIcon("logo.png"))
        self.setGeometry(100, 100, 1200, 800)
        
        self.cameras = []  # List of active camera objects
        self.camera_count = 3
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_feeds)  # Correct method call
        self.captured_images = []
        self.stitched_image = None
        self.frames = []
        self.image_measurements = {}
        self.cameras_running = False

        self.model = None
        self.scaler = None
        self.classify_folder = "classified_images"
        self.stitched_image = None

        self.current_image_name = ""
        
        self.is_background_removed = False
        self.capture_folder = None
        # Initialize the UI
        self.init_ui()

        self.images = []
        self.current_angle = 0  # Initialize current_angle here
        self.total_images = 0
        self.image_label = QLabel(self)

        self.total_tested = 0
        self.ripe_count = 0
        self.unripe_count = 0
        self.overripe_count =0
        self.underripe_count = 0
        self.temp_measurements = {}
        self.measurements_report = {}
        self.image_count = 1
        self.save_dir = "vartically fused_images"
        self.vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.feed_label = QLabel(self)
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.image_color_data = {}
        self.model_path = None  # To store the model path
        self.measurements = {}
        self.default_model_path = "r_versio161.pkl"

        self.offset = 0  # Offset for manual adjustment
        self.temp_folder = "temp_fusion"  # Temporary folder for saving images
        self.total_tested = 0
        self.new_classification_values = []  # Store new classification values for Bunch Report
        self.multiple_classification_values = []  # Store multiple classification values for Summary Report
        self.uploaded_images = None

        

        
    class ImageEditor(QMainWindow):
        def __init__(self, image1_path, image2_path):
            super().__init__()
            self.setWindowTitle("Image Fusion Editor")
            self.setGeometry(200, 200, 800, 600)

            # Load images
            self.image1 = QPixmap(image1_path)
            self.image2 = QPixmap(image2_path)

            # Create a QGraphicsScene
            self.scene = QGraphicsScene(self)
            self.view = QGraphicsView(self.scene)
            self.setCentralWidget(self.view)

            # Add images as QGraphicsPixmapItem (allows moving & transformations)
            self.image_item1 = QGraphicsPixmapItem(self.image1)
            self.image_item2 = QGraphicsPixmapItem(self.image2)

            self.scene.addItem(self.image_item1)
            self.scene.addItem(self.image_item2)

            # Enable dragging
            self.image_item1.setFlag(QGraphicsPixmapItem.ItemIsMovable)
            self.image_item2.setFlag(QGraphicsPixmapItem.ItemIsMovable)

            # Buttons for actions
            self.save_button = QPushButton("Save Merged Image")
            self.save_button.clicked.connect(self.save_merged_image)
            
            # Layout
            button_layout = QHBoxLayout()
            button_layout.addWidget(self.save_button)

            widget = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(self.view)
            layout.addLayout(button_layout)
            widget.setLayout(layout)

            self.setCentralWidget(widget)

        def save_merged_image(self):
            """Merge two images seamlessly and save the result in a user-selected folder."""

            # Capture the scene as an image
            rect = self.scene.itemsBoundingRect()
            image = QImage(int(rect.width()), int(rect.height()), QImage.Format_ARGB32)
            image.fill(Qt.transparent)

            painter = QPainter(image)
            self.scene.render(painter, QRectF(0, 0, rect.width(), rect.height()))
            painter.end()

            buffer = image.bits()
            buffer.setsize(image.byteCount())
            img = np.frombuffer(buffer, dtype=np.uint8).reshape(image.height(), image.width(), 4)

            # Split images assuming top and bottom halves
            mid_height = img.shape[0] // 2
            frame1 = img[:mid_height]
            frame2 = img[mid_height:]

            # Convert to RGBA if necessary
            if frame1.shape[2] == 3:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2BGRA)
            if frame2.shape[2] == 3:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2BGRA)

            # Define overlap height (Increase for smoother blending)
            overlap_height = 150  # Increased overlap height

            # Ensure new image size
            new_height = frame1.shape[0] + frame2.shape[0] - overlap_height
            new_width = max(frame1.shape[1], frame2.shape[1])
            fused_image = np.zeros((new_height, new_width, 4), dtype=np.uint8)

            # Place the first frame
            fused_image[:frame1.shape[0], :frame1.shape[1]] = frame1

            # Create a cosine-based blending mask for a smoother transition
            gradient_mask = (1 - np.cos(np.linspace(0, np.pi, overlap_height))) / 2
            gradient_mask = gradient_mask.reshape(-1, 1)
            gradient_mask = np.tile(gradient_mask, (1, frame2.shape[1]))

            # Apply a stronger Gaussian blur to soften blending transition
            blurred_top = cv2.GaussianBlur(frame1[-overlap_height:], (11, 11), 0)
            blurred_bottom = cv2.GaussianBlur(frame2[:overlap_height], (11, 11), 0)

            # Match brightness before blending
            frame2[:overlap_height] = cv2.addWeighted(frame2[:overlap_height], 0.5, frame1[-overlap_height:], 0.5, 0)

            # Blend the overlapping region
            for c in range(3):  # Iterate over RGB channels
                fused_image[frame1.shape[0] - overlap_height:frame1.shape[0], :frame2.shape[1], c] = (
                    (1 - gradient_mask) * blurred_top[:, :, c] + gradient_mask * blurred_bottom[:, :, c]
                ).astype(np.uint8)

            # Place the remaining part of the second frame
            fused_image[frame1.shape[0]:, :frame2.shape[1]] = frame2[overlap_height:]

            # Ask user to select save folder
            folder = QFileDialog.getExistingDirectory(None, "Select Save Folder")
            if not folder:
                return  # Exit if no folder is selected

            # Generate a unique image name
            timestamp = np.datetime64('now', 's').astype(str).replace('-', '').replace(':', '').replace('.', '')
            image_name = f"fused_image_{timestamp}.png"
            image_path = os.path.join(folder, image_name)

            # Save the fused image
            cv2.imwrite(image_path, fused_image)

        




        

    def capture_and_open_editor(self, image1_path, image2_path):
        """Opens the image fusion editor after capturing images."""
        if os.path.exists(image1_path) and os.path.exists(image2_path):
            self.editor = self.ImageEditor(image1_path, image2_path)
            self.editor.show()
        else:
            print("Error: One or both images are missing!")


    def update_operation_output(self, image_path):
        from PyQt5.QtGui import QMovie
        """Update the operation output display with the specified image or GIF."""
        if not image_path or not os.path.exists(image_path):
            print(f"Invalid or missing file path: {image_path}")
            self.operation_output_2.setText("No image or GIF to display")
            return

        # Check file type (GIF or image)
        if image_path.lower().endswith((".gif")):
            # Handle GIFs
            movie = QMovie(image_path)
            if not movie.isValid():
                print(f"Failed to load GIF from path: {image_path}")
                self.operation_output_2.setText("Failed to load GIF")
                return

            self.operation_output_2.setMovie(movie)
            self.operation_output_2.setAlignment(Qt.AlignCenter)
            movie.start()
            print(f"GIF loaded and displayed: {image_path}")

        elif image_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            # Handle images
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                print(f"Failed to load image from path: {image_path}")
                self.operation_output_2.setText("Failed to load image")
                return

            # Determine display size
            display_size = self.operation_output_2.size()
            if display_size.width() == 0 or display_size.height() == 0:
                display_size = QSize(400, 400)  # Fallback size

            # Scale image and display
            scaled_pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.operation_output_2.setPixmap(scaled_pixmap)
            self.operation_output_2.setAlignment(Qt.AlignCenter)
        
        else:
            print(f"Unsupported file type: {image_path}")
            self.operation_output_2.setText("Unsupported file type")





    def init_ui(self):
        """Set up the UI with buttons, layouts, and camera feed display."""
        # Create the main vertical layout
        main_layout = QVBoxLayout()

        # Add the logos and title section
        title_layout = QHBoxLayout()

        # Left logo
        left_logo_label = QLabel()
        left_logo_pixmap = QPixmap("logo.png")
        if not left_logo_pixmap.isNull():
            left_logo_pixmap = left_logo_pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            left_logo_label.setPixmap(left_logo_pixmap)
            left_logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        else:
            print("Error: Unable to load left logo.")
        title_layout.addWidget(left_logo_label, alignment=Qt.AlignLeft)

        # Title
        title_label = QLabel("<h1>Oil Palm Bunch Grader</h1>")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            """
            font-size: 25px;
            font-family: Arial;
            font-weight: bold;
            color:rgb(41, 89, 185);
            background-color: #ECF0F1;
            padding: 10px;
            border-radius: 15px;
            """
        )
        title_layout.addWidget(title_label, alignment=Qt.AlignCenter)
    

        # Right logo
        right_logo_label = QLabel()
        right_logo_pixmap = QPixmap("icar_logo.png")
        if not right_logo_pixmap.isNull():
            right_logo_pixmap = right_logo_pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            right_logo_label.setPixmap(right_logo_pixmap)
            right_logo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        else:
            print("Error: Unable to load right logo.")
        title_layout.addWidget(right_logo_label, alignment=Qt.AlignRight)

        main_layout.addLayout(title_layout)

        # Horizontal layout for left and right panels
        h_layout = QHBoxLayout()

        # Set up left panel
        left_panel_layout = self.setup_left_panel()
        h_layout.addLayout(left_panel_layout)

        # Set up right panel
        right_panel_layout = self.setup_right_panel()
        h_layout.addLayout(right_panel_layout)

        # Add a checkbox for background removal
        #self.background_check_box = QCheckBox("Remove Background")
        #main_layout.addWidget(self.background_check_box)

        # Add horizontal layout to the main layout
        main_layout.addLayout(h_layout)

        # Set the main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
       
    

    def setup_left_panel(self):
        """Set up the left panel with buttons and input fields."""
        # Define the layout for the left panel
        self.left_panel_layout = QVBoxLayout()
        # Remove any default spacing and margins
        self.left_panel_layout.setSpacing(28)
        self.left_panel_layout.setContentsMargins(1, 0, 0, 0)
        self.default_model_path = "r_versio161.pkl"

        # Remove any default spacing and margins
        self.left_panel_layout.setSpacing(28)
        self.left_panel_layout.setContentsMargins(1, 0, 0, 0)

        # Title
        title_label = QLabel("<h1 style='text-align:center;'>Save/Load Data</h1>")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            """
            font-size: 15px;
            font-family: Arial;
            font-weight: bold;
            color:rgb(41, 77, 185);
            background-color:rgb(236, 239, 241);
            padding: 10px;
            border-radius: 10px;
            """
        )
        self.left_panel_layout.addWidget(title_label)

        # File Path Display Label
        self.folder_path_label = QLabel("FilePath: None")
        self.folder_path_label.setFixedSize(400, 50)  # Increase height to allow wrapping
        self.folder_path_label.setWordWrap(True)  # Enable word wrapping
        self.folder_path_label.setStyleSheet("""
            color: #2C3E50; 
            font-size: 14px; 
            padding-left: 5px;
            background-color: #ECF0F1;
            border: 1px solid #BDC3C7;
            border-radius: 5px;
        """)
        self.left_panel_layout.addWidget(self.folder_path_label)

        # QPushButton to browse for folder path
        self.browse_button = QPushButton("Browse")
        self.browse_button.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                font-size: 12px;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        self.browse_button.setFixedSize(80, 30)
        self.browse_button.clicked.connect(self.select_folder_path)  # Connect to method
        self.left_panel_layout.addWidget(self.browse_button)

        # QLineEdit for new folder name input
        self.new_folder_input = QLineEdit()
        self.new_folder_input.setPlaceholderText("Enter new folder name (optional)")
        self.new_folder_input.setFixedSize(150, 30)
        self.left_panel_layout.addWidget(self.new_folder_input)

        # QLineEdit for image name input
        self.image_name_input = QLineEdit()
        self.image_name_input.setPlaceholderText("Enter image name prefix")
        self.image_name_input.setFixedSize(150, 30)
        self.left_panel_layout.addWidget(self.image_name_input)

        # Rotation Angle Selector
        self.angle_selector = QComboBox()
        self.angle_selector.addItems(["30", "60", "90"])
        self.angle_selector.setPlaceholderText("Choose a rotation angle:")
        self.angle_selector.setFixedSize(150, 30)
        self.left_panel_layout.addWidget(self.angle_selector)

        # Camera Selector
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(["1 Camera", "2 Cameras"])
        self.camera_selector.setStyleSheet("background-color: #BDC3C7; padding: 10px; border-radius: 5px;")
        self.left_panel_layout.addWidget(self.camera_selector)
        
        # Background removal checkbox
        self.background_check_box = QCheckBox("Remove Background")
        self.background_check_box.setChecked(False)  # Default is unchecked
        self.background_check_box.stateChanged.connect(self.toggle_background_removal)  # Link to toggle method
        self.left_panel_layout.addWidget(self.background_check_box)


        
        # Fusion checkbox
        self.fusion_checkbox = QCheckBox("Use Camera Feeds for Fusion")
        self.fusion_checkbox.setChecked(True)  # Default to using camera feeds
        self.left_panel_layout.addWidget(self.fusion_checkbox)

        # Buttons grid layout
        buttons_layout = QGridLayout()
       
        button_data = [
            ("Start Cameras", self.start_stop_cameras),
            ("Generate 3D Image", self.generate_3d_cylindrical_view),
            ("Capture Images", self.capture_images),
            ("Save Report", self.save_report),
            ("Image Fusion", self.fusion_images),
            ("Load Models", self.load_model),
            ("Analyze Image Color", self.analyze_image_color),
            ("Classification", self.classify_image),
            ("Bunch properties",self.startMeasurement),
        ]

        button_style = """
            QPushButton {
                background-color: orange;
                color: white;
                font-size: 16px;
                padding: 5px;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: rgb(24, 29, 31);
            }
        """

        button_width = 172
        button_height = 40

        # Add buttons to the grid layout
        positions = [
        (0, 0), (0, 1),  # First row
        (1, 0), (1, 1),  # Second row
        (2, 0), (5, 1),  # Third row, single button
        (3, 0),         # Fourth row, single button
        (5, 0),
        (4, 0),         # Fifth row, single button
    ]

        for (text, func), pos in zip(button_data, positions):
            btn = QPushButton(text)
            btn.setStyleSheet(button_style)
            btn.setFixedSize(button_width, button_height)
            btn.clicked.connect(func)
            buttons_layout.addWidget(btn, *pos)
            # Apply custom colors to specific buttons
            if text == "Generate 3D Image":
                btn.setStyleSheet("background-color: yellow; color: black; font-weight: bold;")
            elif text == "Save Report":
                btn.setStyleSheet("background-color: yellow; color: black; font-weight: bold;")
            elif text == "Load Models":
                 btn.setStyleSheet("background-color:#3498DB; color: black; font-weight: bold;")
            else:
                btn.setStyleSheet(button_style)


        # Add the grid layout to the main left panel layout
        self.left_panel_layout.addLayout(buttons_layout)


        # Default model path
        
        self.label_model_path = QLabel(f"Model loaded: {self.default_model_path}")  # Initial text
        self.label_model_path.setFixedSize(400, 50)  # Increase height to allow wrapping
        self.label_model_path.setWordWrap(True)  # Enable word wrapping
        self.label_model_path.setStyleSheet("""
            color: #2C3E50; 
            font-size: 14px; 
            padding-left: 5px;
            background-color: #ECF0F1;
            border: 1px solid #BDC3C7;
            border-radius: 5px;
            
        """)



        
        # Add the label below the button layout
        self.left_panel_layout.addWidget(self.label_model_path)

        # Return the fully set up layout
        return self.left_panel_layout

        
    
    def start_stop_cameras(self):
        """Toggle the cameras on or off based on the current state."""
        if self.cameras_running:
            # Stop the camerasclass MultiCameraApp(QMainWindow):
                self.stop_cameras()  # Implement the actual stop logic here
                self.sender().setText("Start Cameras")  # Change button text back to "Start Cameras"
                self.cameras_running = False
        else:
            # Start the cameras
            self.start_cameras()  # Implement the actual start logic here
            self.sender().setText("Stop Cameras")  # Change button text to "Stop Cameras"
            self.cameras_running = True




    def setup_right_panel(self):
        """Set up the right panel with a split frame for camera feeds and operation outputs."""
        right_panel_layout = QVBoxLayout()

        # Set layout margins and spacing to 0 for the entire right panel layout
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(0)

        # Create a grid layout for the split frame
        split_frame_layout = QGridLayout()
        split_frame_layout.setContentsMargins(0, 0, 0, 0)
        split_frame_layout.setSpacing(0)  # Add some spacing between elements

        # Operational Output 2
        self.operation_output_2 = QLabel("Operation Output")
        self.operation_output_2.setFixedSize(720, 690)  # Fixed size for this widget
        self.operation_output_2.setStyleSheet(
            "background-color: #34495E; "
            "border: 3px solid #1abc9c; "  # Add border to the widget
            "border-radius: 10px;"
        )
        self.operation_output_2.setAlignment(Qt.AlignCenter)
        split_frame_layout.addWidget(self.operation_output_2, 0, 0, 2, 1)  # Spans two rows

        # Camera Feed 1
        self.camera_feed_1 = QLabel("Camera Feed 1")
        self.camera_feed_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_feed_1.setStyleSheet(
            "background-color: #2C3E50; "
            "border: 3px solid #e74c3c; "
            "border-radius: 10px;"
        )
        self.camera_feed_1.setAlignment(Qt.AlignCenter)
        split_frame_layout.addWidget(self.camera_feed_1, 0, 1)  # Row 0, Column 1

        # Camera Feed 2
        self.camera_feed_2 = QLabel("Camera Feed 2")
        self.camera_feed_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_feed_2.setStyleSheet(
            "background-color: #34495E; "
            "border: 3px solid #3498db; "
            "border-radius: 10px;"
        )
        self.camera_feed_2.setAlignment(Qt.AlignCenter)
        split_frame_layout.addWidget(self.camera_feed_2, 1, 1)  # Row 1, Column 1

        # Adjust row stretch to balance sizes
        split_frame_layout.setRowStretch(0, 1)  # Camera Feed 1 gets 1 part
        split_frame_layout.setRowStretch(1, 1)  # Camera Feed 2 gets 1 part
        split_frame_layout.setColumnStretch(0, 2)  # Operation Output gets 2 parts
        split_frame_layout.setColumnStretch(1, 3)  # Camera Feeds share remaining space equally

        # Add the split frame layout to the right panel layout
        right_panel_layout.addLayout(split_frame_layout)

        # Message Box for displaying error or success messages
        self.message_box = QWidget()
        self.message_box.setFixedHeight(80)
        self.message_box.setStyleSheet(
            "background-color: white; "
            "border: 3px solid #1abc9c; "
            "border-radius: 10px;"
        )

        # Message Label
        self.message_label = QLabel("Message goes here")
        self.message_label.setStyleSheet("color: red; font-size: 14px;")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)

        # Add the message label to the message box
        message_layout = QVBoxLayout()
        message_layout.setContentsMargins(10, 10, 10, 10)
        message_layout.addWidget(self.message_label)
        self.message_box.setLayout(message_layout)

        # Add the message box below the operation output
        right_panel_layout.addWidget(self.message_box)
        

        self.project_info_text = QLabel(
            "<span style='color: red;'>Developed by</span><br>"
            "<span style='color: blue;'>S. Shivashankar, K. Suresh and P. Anitha</span><br>"
            "<span style='color: navy;'>ICAR - Indian Institute of Oil Palm Research, Pedavegi, Andhra Pradesh</span><br>"
            "<span style='color: orange; font-style: italic;'>Financial Support: ICAR & DAFW, Govt of India</span><br>"
            "<span style='color: deeppink;'><i>Disclaimer: No liability what so ever is accepted for the use of this application</i></span><br>"
        )

        self.project_info_text.setStyleSheet(
            "font-size: 15px; "  # Slightly larger font for readability
            "background-color: white; "
            "border: 1px solid #1abc9c; "
            "border-radius: 5px; "
            "padding: 15px;"
        )

        self.project_info_text.setAlignment(Qt.AlignCenter)
        self.project_info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.project_info_text.setFixedHeight(120)  # Increase height to ensure all lines fit

        right_panel_layout.addWidget(self.project_info_text)

        return right_panel_layout







        

    def display_message(self, message, is_error=True):
        """Update the message label with a success or error message."""
        # Set the message label's color based on error or success
        color = "red" if is_error else "green"
        
        # Customize the appearance of the message label
        self.message_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
        self.message_label.setText(message)
        
        # Set the alignment to center
        self.message_label.setAlignment(Qt.AlignCenter)
        
        # Enable word wrapping to adjust the label's height automatically
        self.message_label.setWordWrap(True)
        
        # Set a minimum width or allow the label to grow dynamically based on the message
        self.message_label.setMinimumWidth(300)  # Adjust as needed for minimum width
        
        # If needed, adjust the height dynamically by considering the wrapped text
        self.message_label.adjustSize()  # Ensures the size is updated to fit the content
        
        # Assuming you have a main layout
        layout = self.layout()  # Get the main layout of the window
        
        # If you want the message at the top, insert the label at the top of the layout
        if layout:
            layout.insertWidget(0, self.message_label)


    

    def convert_cv_to_qt(frame):
        """Convert a OpenCV image (BGR) to QImage (RGB)."""
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        return qt_image

    def select_folder_path(self):
        """Open a dialog to select a folder and update the QLabel multiple times."""
        try:
            # Open the folder selection dialog
            folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Fused Images")
            
            if folder:
                self.save_folder = folder
                self.display_message(f"Folder path selected: {folder}", is_error=False)
            else:
                self.display_message("No folder selected. Please select a valid folder.", is_error=True)

            if folder:  # If the user selects a valid folder
                # Update the folder path label with the selected folder
                self.folder_path_label.setText(f"FilePath: {folder}")
                self.folder_path_label.setStyleSheet(
                    "color: #2C3E50; font-size: 14px; padding-left: 5px; "
                    "background-color: #ECF0F1; border: 1px solid #BDC3C7; border-radius: 5px;"
                )
                self.folder_path_label.setFixedHeight(self.folder_path_label.sizeHint().height())  # Adjust label height dynamically
                self.capture_folder = folder  # Store the selected folder path for later use
                self.save_folder = folder  # Store the folder as the "selected" folder
                self.message_label.setText(f"Folder path selected: {folder}")
                self.message_label.setStyleSheet("color: green; font-size: 14px;")
                return folder  # Return the selected folder path
            else:
                # Handle case where no folder is selected
                self.message_label.setText("Path not selected. Please choose a valid folder.")
                self.message_label.setStyleSheet("color: red; font-size: 14px;")
                self.folder_path_label.setText("FilePath: None")  # Reset folder path label
                self.folder_path_label.setFixedHeight(self.folder_path_label.sizeHint().height())
                return None  # Return None if no folder was selected
        except Exception as e:
            # Handle unexpected errors gracefully
            print(f"Error in selecting folder: {e}")
            self.message_label.setText("An error occurred while selecting the folder.")
            self.message_label.setStyleSheet("color: red; font-size: 14px;")
            return None  # Return None in case of errorr


    
    

    def toggle_cameras(self):
        """Toggle between starting and stopping the cameras."""
        if hasattr(self, "cameras") and self.cameras_running:  # Cameras are currently running
            self.stop_cameras()
            self.camera_button.setText("Start Cameras")
        else:
            self.start_cameras()
            self.camera_button.setText("Stop Cameras")




    def start_cameras(self):
        """Initialize only cameras connected via USB HUB based on user selection."""
        
        self.cameras = []
        self.frames = []  # Store frames from each camera

        # Detect cameras connected to USB hub
        detected_cameras = self.get_usb_hub_cameras()

        if not detected_cameras:
            self.message_label.setText("No USB Hub-connected cameras detected. Please check your connections.")
            self.message_label.setStyleSheet("color: red; font-size: 14px;")
            print("[ERROR] No USB cameras detected through the hub.")
            return

        print(f"Detected USB Hub Cameras: {detected_cameras}")

        # Get user-selected camera count
        camera_count = self.camera_selector.currentIndex() + 1  # 1 Camera, 2 Cameras, etc.
        selected_cameras = detected_cameras[:camera_count]  # Prioritize USB hub cameras

        if not selected_cameras:
            self.message_label.setText("No USB Hub cameras available for selection.")
            self.message_label.setStyleSheet("color: red; font-size: 14px;")
            print("[ERROR] No USB cameras available for the selected configuration.")
            return

        initialized_cameras = 0

        for cam_index in selected_cameras:
            camera = cv2.VideoCapture(cam_index)
            if camera.isOpened():
                self.cameras.append(camera)
                initialized_cameras += 1
                print(f"USB Hub Camera {cam_index} initialized successfully.")
            else:
                print(f"Failed to initialize USB Hub Camera {cam_index}.")
                camera.release()

        if initialized_cameras > 0:
            self.message_label.setText(f"Successfully initialized {initialized_cameras} USB Hub camera(s).")
            self.message_label.setStyleSheet("color: green; font-size: 14px;")

            if not hasattr(self, "timer"):
                self.timer = QTimer()

            self.timer.timeout.connect(self.update_camera_feeds)
            self.timer.start(30)  # Update every 30ms
            self.cameras_running = True
            print("Frame processing started.")
        else:
            self.message_label.setText("Failed to initialize any USB Hub cameras. Please check your connections.")
            self.message_label.setStyleSheet("color: red; font-size: 14px;")
            print("No USB Hub cameras were successfully initialized.")

    def get_usb_hub_cameras(self):
        """Detect cameras specifically connected to the USB Hub."""
        usb_cameras = []
        system_os = platform.system()

        if system_os == "Linux":
            # List USB devices and filter cameras connected via HUB
            hub_devices = os.popen("lsusb | grep 'Hub'").read().strip()
            camera_list = os.popen("v4l2-ctl --list-devices").read()
            devices = camera_list.split("\n\n")

            for device in devices:
                if "USB" in device or "usb" in device:
                    lines = device.split("\n")
                    for line in lines:
                        if "/dev/video" in line:
                            index = int(re.search(r"/dev/video(\d+)", line).group(1))
                            usb_cameras.append(index)

        elif system_os == "Windows":
            # Detect USB cameras using wmic (checking for USB-Hub connections)
            camera_list = os.popen("wmic path Win32_PnPEntity where \"Name like '%USB%'\" get Name").read()
            lines = camera_list.split("\n")

            for i in range(10):  # Check first 10 video indices
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    usb_cameras.append(i)
                    cap.release()

        return usb_cameras

    def detect_camera_count(self):
        """Detect the number of USB Hub-connected cameras."""
        usb_cameras = self.get_usb_hub_cameras()
        return len(usb_cameras)

    def update_camera_feeds(self):
        """Update camera feeds and display them in the respective QLabel widgets."""
        if not self.cameras:
            print("[WARNING] No cameras initialized. Skipping frame update.")
            return

        self.frames = []  # Reset frames for this update

        for i, camera in enumerate(self.cameras):
            ret, frame = camera.read()

            if not ret:
                print(f"[ERROR] Failed to capture frame from camera {i}.")
                self.frames.append(None)  # Maintain frame count consistency
                self.update_feed_placeholder(i)  # Display placeholder for failed feeds
                continue

            # Convert the frame to RGB format for display
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"[ERROR] Failed to convert frame color for camera {i}: {e}")
                self.frames.append(None)
                self.update_feed_placeholder(i)
                continue

            # Convert the frame to QPixmap
            pixmap = self.convert_frame_to_pixmap(frame_rgb)
            if pixmap:
                self.frames.append(frame)  # Store valid frames
                self.update_feed_pixmap(i, pixmap)  # Update the QLabel
            else:
                print(f"[ERROR] Failed to create pixmap for camera {i}")
                self.frames.append(None)
                self.update_feed_placeholder(i)

    

    def update_feed_pixmap(self, feed_index, pixmap):
        """Update QLabel safely, reinitializing if deleted."""
        import sip
        feed_map = {
            0: "camera_feed_1",  # First Camera Output
            1: "camera_feed_2",  # Second Camera Output
        }

        if feed_index in feed_map:
            label_attr = feed_map[feed_index]
            label = getattr(self, label_attr, None)

            if label is None or sip.isdeleted(label):  # If QLabel is deleted, recreate it
                print(f"[INFO] Reinitializing QLabel {label_attr} (was deleted).")
                setattr(self, label_attr, QLabel(self))
                label = getattr(self, label_attr)
                label.setScaledContents(True)

            label.setPixmap(pixmap)  # Now update the QLabel safely
        else:
            print(f"[WARNING] Invalid feed index: {feed_index}.")


    def update_feed_placeholder(self, feed_index):
        """Display a placeholder image or text for the specified feed."""
        placeholder_pixmap = QPixmap(640, 480)  # Adjust size as needed
        placeholder_pixmap.fill(Qt.gray)  # Fill with a neutral color

        feed_map = {
            0: self.camera_feed_1,  # First Camera Placeholder
            1: self.camera_feed_2,  # Second Camera Placeholder
        }
        
        if feed_index in feed_map:
            feed_map[feed_index].setPixmap(placeholder_pixmap)
        else:
            print(f"[WARNING] Invalid feed index: {feed_index}.")



    def stop_cameras(self):
        """Release all cameras and stop frame processing."""
        try:
            # Release all cameras
            if hasattr(self, "cameras") and self.cameras:
                for camera in self.cameras:
                    if camera.isOpened():
                        camera.release()
                self.cameras = []  # Clear the camera list after releasing

            # Stop the timer if it exists and is active
            if hasattr(self, "timer") and self.timer.isActive():
                self.timer.stop()

            # Clear the camera feed display
            if hasattr(self, "operation_output_2"):
                self.operation_output_2.clear()

            # Set camera state to stopped
            self.cameras_running = False

            # Display success message
            self.display_message("Cameras stopped successfully.", is_error=False)  # Success message (is_error=False)

            print("Cameras stopped successfully.")
            
        except Exception as e:
            # Handle any errors that occur during the stopping process
            self.display_message(f"Error stopping cameras: {str(e)}", is_error=True)
            print(f"Error stopping cameras: {str(e)}")



    def update_display_with_frames(self):
        """Update the GUI with frames from all cameras."""
        for i, frame in enumerate(self.frames):
            if frame is not None:
                # Convert frame to QImage
                height, width, _ = frame.shape
                image = QImage(frame.data, width, height, 3 * width, QImage.Format_BGR888)
                pixmap = QPixmap(image)

                # Update QLabel with the frame
                if i == 0:
                    self.camera_feed_1.setPixmap(pixmap)
                elif i == 1:
                    self.camera_feed_2.setPixmap(pixmap)

    
    def convert_frame_to_pixmap(self, frame):
        """Convert OpenCV frame (NumPy array) to QPixmap."""
        try:
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(q_image)
        except Exception as e:
            print(f"[ERROR] Failed to convert frame to QPixmap: {e}")
            return None




    def combine_images(self, images):
        """Combine multiple images into a grid based on the number of cameras."""
        if len(images) == 1:
            return images[0]
        elif len(images) == 2:
            return np.hstack(images)
        elif len(images) == 3:
            return np.hstack(images + [np.zeros_like(images[0])])
        elif len(images) == 4:
            top_row = np.hstack([images[0], images[1]])
            bottom_row = np.hstack([images[2], images[3]])
            return np.vstack([top_row, bottom_row])
    


    def convert_to_qt_image(self, image):
        """Convert an OpenCV image to a QImage."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    from PyQt5.QtWidgets import QLabel

    def calibrate_scale(self, contour, reference_width_cm):
        """
        Calibrate scale using the reference object and calculate cm per pixel.
        """
        # Get the width of the detected reference object in pixels
        _, _, w, _ = cv2.boundingRect(contour)

        if w == 0:
            print("[ERROR] Reference object width is zero.")
            return None  # Calibration failed if width is zero
        
        cm_per_pixel = reference_width_cm / w  # Calculate the scale factor
        return cm_per_pixel
        
        


    def capture_images(self):
        """
        Capture and save images from all active cameras with measurements and annotations.
        """
        try:
            if not self.capture_folder:
                self.display_message("Error: Please select a folder first!", is_error=True)
                return

            folder = self.capture_folder.strip()
            new_folder_name = self.new_folder_input.text().strip()
            if new_folder_name:
                folder = os.path.join(folder, new_folder_name)
                os.makedirs(folder, exist_ok=True)

            print(f"Saving images to: {folder}")  # Debugging folder path
            self.stitching_folder = folder

            base_name = self.image_name_input.text().strip()
            if not base_name:
                self.display_message("Error: Please enter an image name prefix!", is_error=True)
                return

            os.makedirs(folder, exist_ok=True)
            self.captured_images = []
            remove_background = self.background_check_box.isChecked()
            image_counter = 1

            for i, frame in enumerate(self.frames):
                if frame is None:
                    self.display_message(f"Capture Failed: Frame {i + 1} is invalid. Skipping.", is_error=True)
                    continue

                print(f"Frame {i + 1} captured.")  # Debugging frame capture

                if remove_background:
                    processed_frame = self.remove_background(frame)
                    if processed_frame is None or processed_frame.size == 0:
                        self.display_message(f"Error: Background removal failed for Frame {i + 1}. Skipping.", is_error=True)
                        continue
                else:
                    processed_frame = frame


                # Update current_image_name before saving the image
                self.current_image_name = f"{base_name}_{image_counter}.png"

                file_path = os.path.join(folder, self.current_image_name)

                success = cv2.imwrite(file_path, processed_frame)
                print(f"Saving image {self.current_image_name}: {success}")  # Debugging save success

                if success:
                    self.display_message(f"Captured image saved at: {file_path}", is_error=False)
                    self.captured_images.append(file_path)

                    # Display the first captured image in the UI
                    if image_counter == 1:
                        pixmap = QPixmap(file_path)
                        self.operation_output_2.setPixmap(pixmap)
                        self.operation_output_2.setAlignment(Qt.AlignCenter)
                        self.operation_output_2.setScaledContents(True)
                    image_counter += 1
                else:
                    self.display_message(f"Error: Failed to save image {self.current_image_name}.", is_error=True)

            if self.captured_images:
                self.display_message(f"Images saved successfully in:\n{folder}\nFiles named: {base_name}_*.png", is_error=False)
            else:
                self.display_message("Error: No images were captured successfully.", is_error=True)

        except Exception as e:
            self.display_message(f"Error: An unexpected error occurred: {str(e)}", is_error=True)
            print(f"Error: {str(e)}")

   
    def remove_background(self, image):
        """Remove the background and keep only the object with no extra pixels."""
        try:
            # Ensure the image is in proper color space (BGR)
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("Image must be a color image (3 channels).")

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Initialize mask and models for GrabCut
            mask = np.zeros(gray_image.shape, np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Define an initial rectangle for GrabCut
            height, width = gray_image.shape
            margin = 10  # Adjustable margin for the rectangle
            rect = (margin, margin, width - 2 * margin, height - 2 * margin)

            # Apply GrabCut algorithm
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

            # Refine the mask to extract the foreground
            refined_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Find contours and create a bounding box around the object
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                refined_mask = refined_mask[y:y+h, x:x+w]
                image = image[y:y+h, x:x+w]

            # Convert the image to RGBA format
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

            # Apply mask directly to alpha channel
            result[:, :, 3] = refined_mask * 255

            return result
        except Exception as e:
            print(f"Error in background removal: {str(e)}")
            return image  # Return the original image if an error occurs
    




    def toggle_background_removal(self, checked):
        """Toggle background removal based on the checkbox state."""
        self.is_background_removed = checked
        self.display_message(f"Background removal {'enabled' if self.is_background_removed else 'disabled'}.", is_error=False)
        print(f"Background removal {'enabled' if self.is_background_removed else 'disabled'}.")



    def remove_background_no_pixel(self, image):
        """Remove the background and keep only the object with no extra pixels."""
        try:
            # Ensure the image is in proper color space (BGR)
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("Image must be a color image (3 channels).")

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Initialize mask and models for GrabCut
            mask = np.zeros(gray_image.shape, np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Define an initial rectangle for GrabCut
            height, width = gray_image.shape
            margin = 10  # Adjustable margin for the rectangle
            rect = (margin, margin, width - 2 * margin, height - 2 * margin)

            # Apply GrabCut algorithm
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

            # Refine the mask to extract the foreground
            refined_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Find contours and create a bounding box around the object
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                refined_mask = refined_mask[y:y+h, x:x+w]
                image = image[y:y+h, x:x+w]

            # Convert the image to RGBA format
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

            # Apply mask directly to alpha channel
            result[:, :, 3] = refined_mask * 255

            return result
        except Exception as e:
            print(f"Error in background removal: {str(e)}")
            return image  # Return the original image if an error occurs
    
    

    


    def fusion_images(self):
        """Fuse images based on the checkbox selection: either from system or camera feeds."""
        use_camera_feeds = self.fusion_checkbox.isChecked()  # Check if the checkbox is checked

        if use_camera_feeds:
            # Fusion using camera feeds
            self.fuse_from_camera_feeds()  # Capture frames first
        else:
            # Fusion using system images from a folder
            self.fuse_from_system_images()
            

    def fuse_from_camera_feeds(self):
        """Capture two images, save them, and open an editing window."""
        if not hasattr(self, 'frames') or len(self.frames) < 2:
            self.display_message("Error: Not enough frames captured.", is_error=True)
            return

        frame1, frame2 = self.frames[:2]

        if frame1 is None or frame2 is None:
            self.display_message("Error: Frames are empty.", is_error=True)
            return

        # Ensure temp folder exists
        os.makedirs(self.temp_folder, exist_ok=True)

        temp_path1 = os.path.join(self.temp_folder, "temp_frame1.jpg")
        temp_path2 = os.path.join(self.temp_folder, "temp_frame2.jpg")

        # Save images
        success1 = cv2.imwrite(temp_path1, frame1)
        success2 = cv2.imwrite(temp_path2, frame2)

        if not success1 or not success2:
            self.display_message("Error: Failed to save images.", is_error=True)
            return

        self.display_message("Frames saved temporarily. Open Adjustments.", is_error=False)

        # Open editing window in a separate thread
        self.capture_and_open_editor(temp_path1, temp_path2)

        
    
    def fuse_from_system_images(self, overlap=200):
        """Fuse objects from images by concatenating their left and right edges."""
        # Step 1: Ask the user to select the folder with images
        folder_path = QFileDialog.getExistingDirectory(None, "Select Image Folder")
        if not folder_path:
            self.display_message("No folder selected. Fusion aborted.", is_error=True)
            return

        # Step 2: Check if a folder has been selected previously for saving fused images
        if not hasattr(self, 'save_folder') or not self.save_folder:
            # If no folder is selected for saving, ask the user to select one
            self.display_message("No folder selected for saving fused images. Please select a folder first.", is_error=True)
            # Open the folder selection dialog to choose the save folder
            self.select_folder_path()  # This function will handle the folder selection
            if not hasattr(self, 'save_folder') or not self.save_folder:
                return  # Abort if no save folder is selected

        # If a save folder has been set, proceed with fusion
        save_dir = self.save_folder

        # Step 3: Get list of valid image files in the folder
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(image_extensions)]

        if not image_paths:
            self.display_message("No valid images found in the folder.", is_error=True)
            return

        # Step 4: Load images with transparency support
        images = [cv2.imread(img_path, cv2.IMREAD_UNCHANGED) for img_path in image_paths]

        failed_images = [image_paths[i] for i, img in enumerate(images) if img is None]
        if failed_images:
            self.display_message(f"Failed to load images: {', '.join(failed_images)}", is_error=True)
            return

        # Step 5: Ensure all images have 4 channels (RGBA)
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) if img.shape[2] == 3 else img for img in images]

        # Step 6: Resize images to a consistent size for better alignment
        target_size = (800, 600)  # You can modify this size if you wish
        images = [cv2.resize(img, target_size) for img in images]

        # Step 7: Handle folder creation if necessary
        folder = self.save_folder.strip()  # Get the base folder path
        new_folder_name = self.new_folder_input.text().strip()  # Get the folder name input from user
        
        if new_folder_name:
            # Combine the base folder with the new folder name
            folder = os.path.join(folder, new_folder_name)
            os.makedirs(folder, exist_ok=True)  # Create the new folder if it doesn't exist

        # Step 8: Fuse images
        fused_image = self.fuse_objects(images, overlap, folder)

        if fused_image is not None:
            self.display_message("Object fusion completed successfully.", is_error=False)
            #cv2.imshow("Fused Image", fused_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            self.display_message("Fusion failed due to object detection or image processing issues.", is_error=True)

    def detect_objects(self, image):
        """Detect objects in an image using contours."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def fuse_objects(self, images, overlap=200, save_dir="fused_images"):
        """Fuse objects in images by blending their overlapping edges and saving intermediate results."""
        if len(images) == 0:
            print("No images provided for fusion.")
            return None

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Initialize fused image as the first image
        fused_image = images[0]

        # Automatically save the first fused image
        self.save_fused_image(fused_image, save_dir)  # Remove `save_dir` from method

        for idx, img in enumerate(images[1:], start=1):
            # Ensure both images have an alpha channel
            if fused_image.shape[2] == 3:
                fused_image = cv2.cvtColor(fused_image, cv2.COLOR_BGR2BGRA)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            # Define the width of the overlap region
            overlap_width = overlap

            # Compute the dimensions of the new fused image
            new_width = fused_image.shape[1] + img.shape[1] - overlap_width
            new_height = max(fused_image.shape[0], img.shape[0])
            new_fused_image = np.zeros((new_height, new_width, 4), dtype=np.uint8)

            # Place the fused image on the canvas
            new_fused_image[:fused_image.shape[0], :fused_image.shape[1]] = fused_image

            # Blend the overlapping region
            overlap_region_fused = fused_image[:, -overlap_width:]
            overlap_region_img = img[:, :overlap_width]

            alpha = np.linspace(0, 1, overlap_width).reshape(1, -1)  # Gradient for blending
            alpha = np.tile(alpha, (overlap_region_fused.shape[0], 1))  # Match height

            blended_overlap = (1 - alpha)[:, :, None] * overlap_region_fused + alpha[:, :, None] * overlap_region_img
            new_fused_image[:overlap_region_fused.shape[0], fused_image.shape[1] - overlap_width:fused_image.shape[1]] = blended_overlap.astype(np.uint8)

            # Place the non-overlapping part of the current image
            new_fused_image[:img.shape[0], fused_image.shape[1]:] = img[:, overlap_width:]

            # Update the fused image for the next iteration
            fused_image = new_fused_image

            # Save the intermediate fused image
            self.save_fused_image(fused_image, save_dir)

            print(f"Intermediate image {idx} saved to {save_dir}")

        # Return the final fused image
        return fused_image

    def save_fused_image(self, fused_image, save_dir):
        from datetime import datetime
        """Save the fused image to the selected folder."""
        try:
            # Check if a folder path is selected
            if not save_dir:
                raise ValueError("No folder selected for saving the image. Please select a folder first.")

            # Ensure the selected folder exists
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
             
            base_name = self.image_name_input.text().strip()
            if not base_name:
                base_name = "fusion_img"
               

            # Generate the save path for the fused image
           
            save_path = os.path.join(save_dir, f"{base_name}.png")

            # Save the fused image to the selected folder
            if cv2.imwrite(save_path, fused_image):
                self.update_operation_output(save_path)  # Update UI or log
                self.display_message(f"Image saved successfully at {save_path}", is_error=False)
            else:
                raise IOError("Failed to save image. Check file permissions or disk space.")
        except Exception as e:
            # Display an error message if something goes wrong
            self.display_message(f"Error saving image: {str(e)}", is_error=True)


    def load_model(self):
        """Prompt the user to load a model file and display its path."""
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Pickle Files (*.pkl);;All Files (*)")

        if model_path:
            self.model_path = model_path  # Save the model path
            try:
                # Load the model and scaler from the file
                with open(model_path, 'rb') as file:
                    model_data = joblib.load(file)  # Load the dictionary containing the model and scaler
                    self.model = model_data.get('model')  # Extract the model
                    self.scaler = model_data.get('scaler')  # Extract the scaler
                    
                    if self.model and self.scaler:
                        self.display_message("Model and scaler loaded successfully.", is_error=False)
                        # Update label with model path
                        self.label_model_path.setText(f"Model loaded from: {self.model_path}")
                    else:
                        self.display_message("Invalid model file. Please upload a correct model and scaler.", is_error=True)
                        self.label_model_path.setText("Model file is invalid.")
            except Exception as e:
                self.display_message(f"Error loading model: {str(e)}", is_error=True)
                self.label_model_path.setText(f"Error: {str(e)}")
        else:
            self.display_message("Model file not selected.", is_error=True)
            self.label_model_path.setText("No model selected.")


    def load_new_images(self):
        """Prompt the user to load multiple images for classification."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)  # Allow selecting multiple files
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        file_dialog.setViewMode(QFileDialog.List)
        
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.uploaded_images = file_paths  # Store selected image paths in a list
                print(f"Loaded images: {self.uploaded_images}")
            else:
                self.display_message("No images selected.", is_error=True)
        else:
            self.display_message("File dialog canceled. No images loaded.", is_error=True)


    def classify_image(self):
        """Classify multiple images and save results."""
        # Use the default model if no model is loaded
        if not self.model:
            if hasattr(self, 'default_model_path') and self.default_model_path:
                self.load_model_from_file(self.default_model_path)  # Load default model
            else:
                self.display_message("Model not loaded. Please load a model first.", is_error=True)
                return

        if not self.model:
            self.display_message("No valid model available for classification.", is_error=True)
            return

        if not hasattr(self, 'save_folder') or not self.save_folder:
            self.display_message("No folder selected for saving classified images. Please select a folder first.", is_error=True)
            self.select_folder_path()
            if not hasattr(self, 'save_folder') or not self.save_folder:
                return

        if not hasattr(self, 'uploaded_images') or not self.uploaded_images:
            self.load_new_images()

        if not self.uploaded_images:
            self.display_message("No valid images uploaded for classification.", is_error=True)
            return

        # Initialize the new classification list for the current classification
        
        self.new_classification_values = []

        try:
            # Ensure classification_counts dictionary exists for Bunch Report
            if not hasattr(self, 'classification_counts'):
                self.classification_counts = {}

            

            # Load the current multiple classification values if they are not initialized
            if not hasattr(self, 'multiple_classification_values'):
                self.multiple_classification_values = []

            for image_path in self.uploaded_images:
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Error: Image {image_path} not loaded properly.")
                        continue

                    # Resize image for feature extraction
                    image_resized = cv2.resize(image, (128, 128))

                    # Extract features
                    color_features = self.extract_color_features(image_resized)
                    texture_features = self.extract_texture_features(image_resized)
                    shape_features = self.extract_shape_features(image_resized)
                    deep_features = self.extract_deep_features(image_resized)

                    # Combine features
                    features = np.concatenate([deep_features, color_features, texture_features, shape_features])

                    # Scale features if model requires it
                    features_scaled = self.scaler.transform([features])
                    prediction = self.model.predict(features_scaled)[0]

                    # Update classification counters dynamically
                    self.total_tested += 1

                    # Update Bunch Report Counter (Accumulates values per batch)
                    if prediction in self.classification_counts:
                        self.classification_counts[prediction] += 1
                    else:
                        self.classification_counts[prediction] = 1

                    # Update invudea_summury (Always stores only the latest classification result)
                    self.invudea_summury = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "prediction": prediction}

                    # Handle folder creation
                    folder = os.path.join(self.save_folder.strip(), self.new_folder_input.text().strip())
                    os.makedirs(folder, exist_ok=True)

                    # Save and display the classified image
                    classified_image_path = self.save_classified_image(image, prediction, folder)
                    self.update_operation_output(classified_image_path)

                    # Add to new classification values for Bunch Report
                    self.new_classification_values.append({"image_name": os.path.basename(image_path), "prediction": prediction})

                    # Add to multiple classification values for Summary Report (accumulating all classifications)
                    self.multiple_classification_values.append({"image_name": os.path.basename(image_path), "prediction": prediction})

                    self.display_message(f"Image {os.path.basename(image_path)} classified successfully. Prediction: {prediction}.", is_error=False)

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    self.display_message(f"Error classifying image {os.path.basename(image_path)}. Please try again.", is_error=True)

            # Reset uploaded images after processing
            self.uploaded_images = None

        except Exception as e:
            print(f"Error during the classification process: {e}")
            self.display_message("An error occurred while processing the images. Please try again.", is_error=True)


    def load_model_from_file(self, model_path):
        """Load the model and scaler from the specified file."""
        try:
            with open(model_path, 'rb') as file:
                model_data = joblib.load(file)  # Load the dictionary containing the model and scaler
                self.model = model_data.get('model')  # Extract the model
                self.scaler = model_data.get('scaler')  # Extract the scaler
                
                if self.model and self.scaler:
                    self.display_message(f"Default model loaded from: {model_path}", is_error=False)
                    self.label_model_path.setText(f"Model loaded from: {model_path}")
                else:
                    self.display_message("Invalid model file. Please upload a correct model and scaler.", is_error=True)
                    self.label_model_path.setText("Model file is invalid.")
        except Exception as e:
            self.display_message(f"Error loading model: {str(e)}", is_error=True)
            self.label_model_path.setText(f"Error: {str(e)}")

    def save_classified_image(self, image, prediction, folder):
        """Save the classified image with a name indicating the prediction."""
        import os
        import cv2

        # Create the filename based on the prediction
        base_name = self.image_name_input.text().strip()
        if not base_name:
            base_name = "classifi_img"
           
        image_name = f"{prediction}_{base_name}.jpg"
        image_path = os.path.join(folder, image_name)

        

        # Save the image
        cv2.imwrite(image_path, image)

        return image_path





           
    def extract_deep_features(self, image):
        """
        Extract deep features using VGG16 pre-trained model.
        """
        image = cv2.resize(image, (128, 128))  # Resize image to match VGG16 input
        image = preprocess_input(np.expand_dims(image, axis=0))
        features = self.vgg_model.predict(image).flatten()
        return features

    def extract_color_features(self, image):
        """Extract color histogram features in RGB and HSV spaces."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_rgb = [cv2.calcHist([image], [i], None, [64], [0, 256]) for i in range(3)]
        hist_hsv = [cv2.calcHist([hsv_image], [i], None, [64], [0, 256]) for i in range(3)]
        hist_features = np.concatenate([hist.flatten() for hist in hist_rgb + hist_hsv])
        hist_features /= np.linalg.norm(hist_features)  # Normalize histogram
        return hist_features

    @staticmethod
    def extract_texture_features(image):
        """
        Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray_image, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=256, symmetric=True, normed=True)
        contrast = np.mean([graycoprops(glcm, 'contrast')[0, i] for i in range(4)])
        correlation = np.mean([graycoprops(glcm, 'correlation')[0, i] for i in range(4)])
        energy = np.mean([graycoprops(glcm, 'energy')[0, i] for i in range(4)])
        homogeneity = np.mean([graycoprops(glcm, 'homogeneity')[0, i] for i in range(4)])
        return [contrast, correlation, energy, homogeneity]

    def extract_shape_features(self, image):
        """
        Extract shape features using contours.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            return [area, perimeter, circularity]
        return [0, 0, 0]




    


    


    def display_frame(self, img):
        """Display the processed frame on the QLabel."""
        label_width = self.camera_feed_1.width()
        label_height = self.camera_feed_1.height()

        # Resize the frame for display
        resized_frame = cv2.resize(img, (label_width, label_height))

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Update the QLabel with the processed frame
        pixmap = QPixmap.fromImage(qt_image)
        self.camera_feed_1.setPixmap(pixmap)
        self.camera_feed_1.setAlignment(Qt.AlignCenter)
        self.camera_feed_1.setScaledContents(True)


    def load_images_from_folder(self, folder):
        """Load images from the specified folder and ensure they have the same size and channels."""
        images = []
        target_size = None
        for filename in sorted(os.listdir(folder)):
            img_path = os.path.join(folder, filename)
            if img_path.endswith(".jpg") or img_path.endswith(".png"):
                img = cv2.imread(img_path)
                if img is not None:
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    if target_size is None:
                        target_size = (img.shape[1], img.shape[0])  # (width, height)
                    img = cv2.resize(img, target_size)
                    images.append(img)
        
        if len(images) < 2:
            self.display_message("Not enough valid images to simulate 3D view.", is_error=True)
            return []
        return images


    def save_3d_as_gif(self, images, save_path, duration_per_frame=300):
        """Save images as a slow-motion GIF."""
        
        # Ensure images are provided
        if not images:
            self.display_message("No images provided for GIF creation.", is_error=True)
            return

        # Ensure save path ends with .gif
        if not save_path.endswith(".gif"):
            save_path += ".gif"
        
        # Ensure the save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            # Convert OpenCV images to PIL format
            pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
            
            # Save GIF
            pil_images[0].save(
                save_path, save_all=True, append_images=pil_images[1:], duration=duration_per_frame, loop=0
            )
            
            # Update output and log
            self.update_operation_output(save_path)
            print(f"3D view data saved as slow-motion GIF to {save_path}")
    
        except Exception as e:
            self.display_message(f"Error saving GIF: {e}", is_error=True)


    def save_3d_as_video(self, images, save_path, fps=30):
        """Save images as a video."""
        if not save_path.endswith(".mp4"):
            save_path += ".mp4"

        height, width, _ = images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        for img in images:
            video_writer.write(img)

        video_writer.release()
        print(f"3D view data saved as video to {save_path}")


    def simulate_3d_view_with_mouse(self, images):
        """Simulate 3D view with mouse movement."""
        total_images = len(images)
        window_name = "3D View Simulation"

        def get_blend_factor(angle, total_images):
            normalized_angle = angle % 360
            image_index = int(normalized_angle // (360 / total_images))
            blend_factor = (normalized_angle % (360 / total_images)) / (360 / total_images)
            return image_index, blend_factor

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_angle
            if event == cv2.EVENT_MOUSEMOVE:
                width = param.shape[1] if param is not None else 1
                current_angle = (x / max(1, width)) * 360

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback, param=images[0])
        current_angle = 0

        try:
            while True:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                index, blend_factor = get_blend_factor(current_angle, total_images)
                next_index = (index + 1) % total_images
                
                # Ensure images have the same size
                try:
                    blended_image = cv2.addWeighted(images[index], 1 - blend_factor, images[next_index], blend_factor, 0)
                    cv2.imshow(window_name, blended_image)
                except cv2.error as e:
                    self.display_message(f"Error during 3D simulation: {e}", is_error=True)
                    break

                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        finally:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(window_name)


    def generate_3d_cylindrical_view(self):
        """Generate and save a 3D cylindrical view."""
        if not hasattr(self, 'save_folder') or not self.save_folder:
            self.display_message("No folder selected for saving. Please select a folder.", is_error=True)
            self.select_folder_path()
            if not hasattr(self, 'save_folder') or not self.save_folder:
                return

        folder = QFileDialog.getExistingDirectory(None, "Select Folder Containing Images")
        if not folder:
            self.display_message("No folder selected. Exiting.", is_error=True)
            return

        images = self.load_images_from_folder(folder)
        if not images:
            return

        base_folder = self.save_folder.strip()
        new_folder_name = self.new_folder_input.text().strip() if self.new_folder_input else ""
        folder_to_save = os.path.join(base_folder, new_folder_name) if new_folder_name else base_folder
        os.makedirs(folder_to_save, exist_ok=True)  # Ensure the folder is created

        # Ask user for save format
        save_format, ok = QInputDialog.getItem(
            None, "Select Save Format", "Choose the format to save 3D data:", ["GIF", "Video"], 0, False
        )
        if not ok:
            self.display_message("Save format selection canceled. Exiting.", is_error=True)
            return

        # Determine file extension
        file_extension = "gif" if save_format == "GIF" else "mp4"

        # Define full file path
        save_filename = f"3d_view.{file_extension}"  # You can change the default filename
        save_path = os.path.join(folder_to_save, save_filename)  # Ensure file is saved inside the folder

        self.update_operation_output(save_path)  # Use full file path now
        print(f"3D view data will be saved to {save_path}")
        self.display_message(f"3D view data will be saved to {save_path}",is_error=False)

        if not save_path:
            self.display_message("No save location selected. Exiting.", is_error=True)
            return

        if save_format == "GIF":
            self.save_3d_as_gif(images, save_path)
        elif save_format == "Video":
            self.save_3d_as_video(images, save_path)

        self.simulate_3d_view_with_mouse(images)



    def analyze_image_color(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", 
                                                "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        
        if not file_path:
            print("No file selected.")
            return
        
        self.image_color_data ={}
        
        image = cv2.imread(file_path)

        if image is None:
            print("Error: Image could not be loaded.")
            return

        # Convert image to RGB format (OpenCV loads images in BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Compute the mean values for each color channel using NumPy
        avg_r, avg_g, avg_b = np.mean(image[:, :, 0]), np.mean(image[:, :, 1]), np.mean(image[:, :, 2])

        # Ensure values are within 0-255 range
        avg_r, avg_g, avg_b = round(avg_r, 2), round(avg_g, 2), round(avg_b, 2)

        print(f"Average Red (R): {avg_r}")
        print(f"Average Green (G): {avg_g}")
        print(f"Average Blue (B): {avg_b}")

        # Store the average color values
        self.image_color_data[file_path] = {
            "Average Red (R)": avg_r,
            "Average Green (G)": avg_g,
            "Average Blue (B)": avg_b
        }

        # Plot a bar chart of average color values
        colors = ['Red', 'Green', 'Blue']
        avg_values = [avg_r, avg_g, avg_b]

        plt.bar(colors, avg_values, color=['red', 'green', 'blue'])
        plt.title("Average Color Values")
        plt.xlabel("Color Channel")
        plt.ylabel("Average Value (0-255)")
        plt.ylim(0, 255)  # Ensure the y-axis stays within the valid color range
        plt.show()



    


    

    """def startMeasurement(self):

        if not hasattr(self, 'save_folder') or not self.save_folder:
                self.display_message("No folder selected for saving classified images. Please select a folder first.", is_error=True)
                self.select_folder_path()
                if not hasattr(self, 'save_folder') or not self.save_folder:
                    return
                
        options = QFileDialog.Options()
        fileNames, _ = QFileDialog.getOpenFileNames(self, "Open Image Files", "", 
                                                    "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", 
                                                    options=options)

        if fileNames:
            self.images = []  # Store multiple images
            self.image_names = []  # Store image names

            for fileName in fileNames:
                image_name = os.path.basename(fileName)
                image = cv2.imread(fileName)
                if image is not None:
                    self.images.append(image)
                    self.image_names.append(image_name)
            
        if self.images:
                self.current_image_index = 0  # Track which image is being processed
                #self.displayImage(self.images[self.current_image_index])
                self.selectROIAndProcess()  # Automatically select ROI and process the image
        else:
                self.display_message("No valid images loaded.", is_error=True)"""
    
    def startMeasurement(self):

        if not hasattr(self, 'save_folder') or not self.save_folder:
            self.display_message("No folder selected for saving classified images. Please select a folder first.", is_error=True)
            self.select_folder_path()
            if not hasattr(self, 'save_folder') or not self.save_folder:
                return

        options = QFileDialog.Options()
        fileNames, _ = QFileDialog.getOpenFileNames(self, "Open Image Files", "", 
                                                    "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", 
                                                    options=options)

        if fileNames:
            self.images = []  # Store multiple images
            self.image_names = []  # Store image names

            for fileName in fileNames:
                image_name = os.path.basename(fileName)
                image = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available
                
                if image is not None:
                    if image.shape[-1] == 4:  # If image has alpha channel
                        bgr = image[:, :, :3]  # Extract BGR channels
                        alpha = image[:, :, 3]  # Extract alpha channel
                        black_background = np.zeros_like(bgr)  # Create a black background
                        alpha_factor = alpha[:, :, None] / 255.0
                        image = (bgr * alpha_factor + black_background * (1 - alpha_factor)).astype(np.uint8)
                    
                    # Add huge black background
                    bg_height = max(image.shape[0] * 2, 540)
                    bg_width = max(image.shape[1] * 2, 950)
                    huge_background = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
                    
                    # Center the image on the huge background
                    y_offset = (bg_height - image.shape[0]) // 2
                    x_offset = (bg_width - image.shape[1]) // 2
                    huge_background[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
                    image = huge_background
                    
                    self.images.append(image)
                    self.image_names.append(image_name)
        
        if self.images:
            self.current_image_index = 0  # Track which image is being processed
            self.selectROIAndProcess()  # Automatically select ROI and process the image
        else:
            self.display_message("No valid images loaded.", is_error=True)

    def selectROIAndProcess(self):
        if not hasattr(self, 'images') or not self.images:
            self.display_message("No images uploaded. Please upload images before selecting ROI.", is_error=True)
            return

        self.measurements = {}

        for i, image in enumerate(self.images):
            image_name = self.image_names[i]

            # Display image and let user select ROI
            roi = cv2.selectROI(f"Select ROI - {image_name}", image, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(f"Select ROI - {image_name}")  # Close window after selection

            # **Cancel Handling**: If user cancels (ROI is 0,0,0,0), skip processing
            if roi == (0, 0, 0, 0):
                self.display_message(f"ROI selection canceled for {image_name}. Skipping this image.", is_error=True)
                continue  # Move to the next image
            
            # Check if ROI is valid (width & height > 0)
            x, y, w, h = roi
            if w > 0 and h > 0:
                cropped_image = image[y:y + h, x:x + w]
                # Process cropped image
                self.processImage(cropped_image, image_name)
            else:
                self.display_message(f"Invalid ROI for {image_name}. Skipping this image.", is_error=True)

        self.display_message("ROI selection and processing complete.", is_error=False)






    def processImage(self, cropped_image, image_name):
        try:
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            self.display_message(f"Error converting image to grayscale: {str(e)}", is_error=True)
            return

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        processed_img = cropped_image.copy()

        if not contours:
            self.display_message(f"No contours detected in {image_name}. Try adjusting the threshold.", is_error=True)
            return

        try:
            largest_contour = max(contours, key=cv2.contourArea)
            real_width_cm, real_height_cm = 31, 24  # Known object dimensions
            ppcm = self.calculate_ppcm(largest_contour, real_width_cm, real_height_cm)

            output_folder = "Bunch Properties"
            os.makedirs(output_folder, exist_ok=True)

            existing_files = [f for f in os.listdir(output_folder) if f.startswith("bunch_") and f.endswith(".jpg")]
            next_index = len(existing_files) + 1

            self.measurements[image_name] = {}  # Store measurements per image

            for contour in contours:
                area = round(cv2.contourArea(contour) / (ppcm**2), 2)
                perimeter = round(cv2.arcLength(contour, True) / ppcm, 2)

                (center_x, center_y), (width, height), rotation_angle = cv2.minAreaRect(contour)
                width_cm, height_cm = sorted([round(width / ppcm, 2), round(height / ppcm, 2)])

                if width_cm < 16 and height_cm < 15:
                    continue  # Skip this object

                aspect_ratio = round(width_cm / height_cm, 2)
                box_area = round(width_cm * height_cm, 2)
                extent = round(area / box_area if box_area > 0 else 0, 2)

                hull = cv2.convexHull(contour)
                convex_area = round(cv2.contourArea(hull) / (ppcm**2), 2)
                convex_perimeter = round(cv2.arcLength(hull, True) / ppcm, 2)

                solidity = round(area / convex_area if convex_area > 0 else 0, 2)
                convexity = round(convex_perimeter / perimeter if perimeter > 0 else 0, 2)

                moments = cv2.moments(contour)
                eccentricity = round(
                    np.sqrt(1 - (moments["mu20"] / (moments["mu20"] + moments["mu02"]))),
                    2
                ) if moments["mu20"] + moments["mu02"] != 0 else 0

                circularity = round((4 * np.pi * area) / (perimeter**2), 2) if perimeter > 0 else 0

                # Store measurements for the current image
                self.measurements[image_name] = {
                    "Width": width_cm,
                    "Height": height_cm,
                    "Aspect Ratio": aspect_ratio,
                    "Bounding Box Area": box_area,
                    "Perimeter": perimeter,
                    "Extent": extent,
                    "Solidity": solidity,
                    "Convexity": convexity,
                    "Eccentricity": eccentricity,
                    "Circularity": circularity,
                }

                print(f"""
                    Width: {width_cm}
                    Height: {height_cm}
                    Aspect Ratio: {aspect_ratio}
                    Bounding Box Area: {box_area}
                    Perimeter: {perimeter}
                    Extent: {extent}
                    Solidity: {solidity}
                    Convexity: {convexity}
                    Eccentricity: {eccentricity}
                    Circularity: {circularity}
                    """)

                
                    

                cv2.drawContours(processed_img, [contour], -1, (0, 255, 0), 2)

                # Save processed image
                output_path = os.path.join(output_folder, f"bunch_{next_index}.jpg")
                cv2.imwrite(output_path, processed_img)
                next_index += 1

                print(f"Saved: {output_path}")
                self.update_operation_output( output_path)
                print(f"Measurements for {image_name}: {self.measurements[image_name]}")
                

        except Exception as e:
            self.display_message(f"Processing error: {str(e)}", is_error=True)
            return

        #self.displayImage(processed_img)


        # Save report with image name
        #self.save_report(self.image_name,measurements)

        cv2.drawContours(processed_img, [contour], -1, (0, 255, 0), 2)

        #self.displayImage(processed_img)

    def calculate_ppcm(self, reference_contour, real_width_cm, real_height_cm):
        x, y, w, h = cv2.boundingRect(reference_contour)
        pixel_width, pixel_height = max(w, h), min(w, h)
        ppcm_w = pixel_width / real_width_cm
        ppcm_h = pixel_height / real_height_cm
        return (ppcm_w + ppcm_h) / 2

    def displayImage(self, img):
        """Display only the ROI selection functionality, hiding the rest."""
        # Convert image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape

        # Resize image if too large (max width: 800px, max height: 600px)
        max_width, max_height = 800, 600
        if w > max_width or h > max_height:
            scaling_factor = min(max_width / w, max_height / h)
            img = cv2.resize(img, (int(w * scaling_factor), int(h * scaling_factor)))
            h, w, _ = img.shape  # Update dimensions

        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Create QLabel for displaying ROI selection only
        if not hasattr(self, 'image_label'):
            self.image_label = QLabel(self)
            self.image_label.setScaledContents(True)
            self.image_label.setAlignment(Qt.AlignCenter)  # Center alignment

        self.image_label.setPixmap(pixmap)
        self.image_label.resize(w, h)  # Adjust size dynamically
        self.image_label.show()

   
    

    def save_report(self):
        from datetime import datetime
        from PyQt5.QtWidgets import QFileDialog, QInputDialog
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import Table, TableStyle
        from reportlab.pdfgen import canvas

        """Save a report with measured object dimensions, color predictions, and classification results to a PDF file."""

        # Get user inputs for batch details
        batch_number, ok = QInputDialog.getText(self, "Batch Number", "Enter Batch Number:")
        if not ok or not batch_number.strip():
            return

        batch_size, ok = QInputDialog.getInt(self, "Batch Size", "Enter Batch Size:")
        if not ok or batch_size <= 0:
            return

        report_type = self.get_report_type()
        if not report_type:
            return

        # Validate classification data
        if not getattr(self, 'classification_counts', {}):
            self.display_message("Classification not done yet. Please classify before saving the report.", is_error=True)
            return

        # Get save location
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "PDF Files (*.pdf)")
        if not file_path:
            self.display_message("No save location selected. Exiting.", is_error=True)
            return

        try:
            c = canvas.Canvas(file_path, pagesize=letter)
            c.setFont("Helvetica", 10)

            # Add Logos at the top (beside the heading)
            logo_left_path = "logo.png"  # Update with your logo path
            logo_right_path = "icar_logo-bg.jpg"  # Update with your logo path

            # Place the left logo
            c.drawImage(logo_left_path, 30, 740, width=50, height=50)  # Adjust position and size
            # Place the right logo
            c.drawImage(logo_right_path, 510, 740, width=50, height=50)  # Adjust position and size

            # Report Header
            c.setFont("Helvetica-Bold", 14)
            c.drawString(90, 740, "Oil Palm Fresh Fruit Bunch Analyzer (OPFFBA) Test Report")
            c.setFont("Helvetica-Bold", 12)
            c.drawString(30, 720, f"Batch Report - {batch_number}")
            c.setFont("Helvetica", 10)
            c.drawString(30, 705, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            c.drawString(30, 690, f"Batch Size: {batch_size}")

            y_position = 670
            c.setFont("Helvetica-Bold", 12)

            # Set the report section title based on report type
            if report_type == "Bunch Report":
                c.drawString(30, y_position, "Measurement and Classification Information:")
            else:
                c.drawString(30, y_position, "Bunch Classification Information:")

            y_position -= 30  # Space after the title

            # Measurement Details (Only for Bunch Report)
            if report_type == "Bunch Report" and hasattr(self, 'measurements'):
                for image_name, data in self.measurements.items():
                    # Table with image name as header
                    table_data = [[f"Image Name: {image_name}", ""]]  # Image name as a header row
                    table_data.append(["Measurement", "Value (cm)"])  # Column headers

                    for key, value in data.items():
                        table_data.append([key, f"{value:.2f}" if isinstance(value, (int, float)) else str(value)])

                    col_widths = [250, 120]  # Adjusted column widths
                    table = Table(table_data, colWidths=col_widths)
                    table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgrey),  # Image name row
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                        ("BACKGROUND", (0, 1), (-1, 1), colors.grey),  # Column headers
                        ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
                        ("ALIGN", (0, 1), (-1, 1), "CENTER"),
                        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),

                        ("FONTNAME", (0, 2), (-1, -1), "Helvetica"),  # Table data
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("PADDING", (0, 0), (-1, -1), 5),
                    ]))

                    # Estimate the table height based on rows
                    table_height = 15 * (len(table_data) + 1)  # Estimate table height
                    y_position -= table_height  # Adjust y_position down

                    # Draw the table on the canvas
                    table.wrapOn(c, 30, y_position)
                    table.drawOn(c, 30, y_position)

                    # Update y_position after printing the table
                    y_position -= 40  # Adjust this based on the table height

                    # Check if a new page is needed
                    if y_position < 100:
                        c.showPage()
                        c.setFont("Helvetica", 10)
                        y_position = 750  # Reset y_position for new page

                # If no measurements available, print message
                if not self.measurements:
                    c.setFont("Helvetica-Oblique", 10)
                    c.drawString(30, y_position, "No measurement data available.")
                    y_position -= 20

                # Color Data Section
                if hasattr(self, 'image_color_data') and self.image_color_data:
                    for image_path, color_data in self.image_color_data.items():
                        c.drawString(30, y_position, f"Image: {image_path}")
                        y_position -= 15
                        for color, value in color_data.items():
                            c.drawString(50, y_position, f"{color}: {value:.2f}" if isinstance(value, float) else str(value))
                            y_position -= 15
                        y_position -= 10
                else:
                    c.setFont("Helvetica-Oblique", 10)
                    c.drawString(30, y_position, "Color values not available.")
                    y_position -= 20

            # Classification Summary
            c.setFont("Helvetica-Bold", 12)
            c.drawString(30, y_position, "Classification Summary:")
            y_position -= 20

            # Classification values (latest classifications for Bunch Report)
            if report_type == "Bunch Report" and hasattr(self, 'new_classification_values'):
                for item in self.new_classification_values:
                    c.drawString(30, y_position, f"Image Name: {item['image_name']} - Prediction: {item['prediction']}")
                    y_position -= 20
                    if y_position < 100:
                        c.showPage()
                        c.setFont("Helvetica", 10)
                        y_position = 750

            # Summary Report - Multiple Classifications
            if report_type == "Summary Report" and hasattr(self, 'multiple_classification_values'):
                for item in self.multiple_classification_values:
                    c.drawString(30, y_position, f"Image Name: {item['image_name']} - Prediction: {item['prediction']}")
                    y_position -= 20
                    if y_position < 100:
                        c.showPage()
                        c.setFont("Helvetica", 10)
                        y_position = 750

                # Display classification summary
                c.drawString(30, y_position, f"Total FFB Analysed: {self.total_tested}")
                y_position -= 20
                c.drawString(30, y_position, "Total Classification Summary:")
                y_position -= 20
                for label, count in self.classification_counts.items():
                    c.drawString(30, y_position, f"{label}: {count}")
                    y_position -= 20

            # Disclaimer
            c.setFont("Helvetica", 9)
            c.drawString(30, y_position - 40, "Disclaimer: This is a System Generated Report, Proper Calibration Required for Precision.")

            c.save()
            self.display_message(f"Report saved successfully to {file_path}.", is_error=False)

        except Exception as e:
            self.display_message(f"Error saving report: {str(e)}", is_error=True)


    def get_report_type(self):
        """Custom dialog for selecting report type using radio buttons."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Report Type")

        layout = QVBoxLayout()

        summary_report = QRadioButton("Summary Report")
        bunch_report = QRadioButton("Bunch Report")
        summary_report.setChecked(True)  # Default selection

        layout.addWidget(summary_report)
        layout.addWidget(bunch_report)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        layout.addWidget(button_box)
        dialog.setLayout(layout)

        if dialog.exec_():  # If user clicks OK
            return "Summary Report" if summary_report.isChecked() else "Bunch Report"
        return None  # If user clicks Cancel


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()  # Ensure the MainWindow runs first
    window.show()
    sys.exit(app.exec_())


