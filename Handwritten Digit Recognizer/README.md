🔢 Handwritten Digit Recognizer

A Computer Vision & Machine Learning project that detects and recognizes handwritten digits from images using HOG + SVM.

✨ Features
🔍 Handwritten digit detection
🎨 Blue-ink image preprocessing
✂️ Automatic digit segmentation
📐 28×28 digit normalization
📊 HOG feature extraction
🔄 Data augmentation
🤖 SVM classification
🖼️ Prediction output generation

🧠 Workflow
Input Image
    ↓
Image Preprocessing
    ↓
Digit Detection
    ↓
28×28 Normalization
    ↓
HOG Features
    ↓
Data Augmentation
    ↓
SVM Classifier
    ↓
Digit Prediction

🛠️ Technologies

Python
OpenCV
NumPy
Pillow
Scikit-image
Scikit-learn
HOG
SVM

📂 Project Structure
Handwritten-Digit-Recognizer/
│
├── main.py
├── model.py
├── config.py
├── image_processing.py
├── feature_extraction.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── custom_train_digits.jpg
│   └── test_image.png
│
└── output/
    ├── prediction.png
    ├── final_digits.png
    └── training_box_overlay.png
    
⚙️ Installation

git clone https://github.com/vickykuthe/Codec-Technologies-Project.git
cd Codec-Technologies-Project/Handwritten-Digit-Recognizer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

▶️ Run
Make sure these files are inside the data folder:
custom_train_digits.jpg
test_image.png
Then run:
python main.py
The generated results will be saved in:

output/
🖼️ Output

![Prediction](output/prediction.png)
<img width="1542" height="748" alt="image" src="https://github.com/user-attachments/assets/e79919f2-6f70-46c8-bec1-63ef002041a9" />


![Final Result](output/final_digits.png)
<img width="1542" height="748" alt="image" src="https://github.com/user-attachments/assets/1bb61e4b-77fa-4bf5-9d00-25675b187ad5" />


🚀 Future Improvements
CNN-based recognition
MNIST dataset training
Real-time webcam recognition
Web application
Prediction confidence scores

👨‍💻 Author
Vicky Kuthe
B.Tech – Information Technology

📜 Copyright

© 2026 Vicky Kuthe. All Rights Reserved.

This project is developed for educational purposes. Unauthorized reproduction, redistribution, or claiming the project as your own is not permitted.
