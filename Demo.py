import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from PIL import Image, ImageTk
import time
from tkinter import ttk
from skimage.feature import hog

model = load('D:\DoAn2\Test_Gender\GUI\model.sav')

def checking(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("D:\DoAn2\Test_Gender\haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        result_label.config(text="Không tìm thấy khuôn mặt trong ảnh!")  # Thông báo nếu không tìm thấy khuôn mặt
        return
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face.flatten() / 255.0
        face = np.expand_dims(face, axis=0)

    # Trích xuất đặc trưng HOG từ khuôn mặt 
    #    face = cv2.GaussianBlur(face, (5, 5), 0)
    #    face = face / 255.0
    #    features = hog(face, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, feature_vector=True)
    #    classes = model.predict(features.reshape(1, -1))
    #    probability=model.predict_proba(features.reshape(1, -1))
    
        # Dự đoán giới tính và xác suất
        classes = model.predict(face)
        if classes[0] < 0.5:
            gender = f"Male" 
        else:
            gender = f"Female"
        result_label.config(text=f"Giới tính: {gender}") 

        #  Vẽ hình chữ nhật xung quanh khuôn mặt và hiển thị dự đoán 
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    image = cv2.resize(image, (400, 400))  
     # Hiển thị ảnh trong giao diện tkinter
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = Image.fromarray(image) 
    image = ImageTk.PhotoImage(image) 
    label.config(image=image)  
    label.image = image  

def webcam():
    face_cascade = cv2.CascadeClassifier("D:\DoAn2\Test_Gender\haarcascade_frontalface_default.xml")
    Categories = ['female', 'male']
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face.flatten() / 255.0
            face = np.expand_dims(face, axis=0)

        # Trích xuất đặc trưng HOG
        #   face = cv2.GaussianBlur(face, (5, 5), 0)
        #   face = face / 255.0
        #   features = hog(face, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, feature_vector=True)
        #   classes = model.predict(features.reshape(1, -1))
        #   probability=model.predict_proba(features.reshape(1, -1))
            classes = model.predict(face)
            probability=model.predict_proba(face)
            if classes[0] < 0.5:
                gender = f"Male ({probability[0][1]*100:.2f}%)" 
            else:
                gender = f"Female ({probability[0][0]*100:.2f}%)"
            # Draw rectangle around face and display gender prediction
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Thoát nếu nhấn 'q'
            break
    cap.release() # Thả webcam và đóng cửa sổ
    cv2.destroyAllWindows()

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        entry_path.delete(0, tk.END)  # Xóa nội dung cũ trong entry
        entry_path.insert(0, file_path)  # Hiển thị đường dẫn mới trong entry
        checking(file_path)

def clear_image():
    label.config(image='')  # Xóa ảnh hiện tại trong Label
    result_label.config(text="")  # Xóa kết quả phân loại
    entry_path.delete(0, tk.END)  # Xóa đường dẫn ảnh

# Tạo cửa sổ
window = tk.Tk()
window.title("Gender Classifier")
window.geometry("800x600")

# Tạo tiêu đề
window_title = tk.Label(window, text="CHƯƠNG TRÌNH PHÂN LOẠI GIỚI TÍNH NAM NỮ",font=("Arial", 18, "bold"))
window_title.pack(pady=(20, 5))  # Giảm khoảng cách bên dưới và tăng khoảng cách bên trên

# Tạo khung chứa các thành phần trong giao diện 
frame = tk.Frame(window)
frame.pack(pady=10)

# Thanh chứa đường dẫn ảnh
entry_path = tk.Entry(frame, width=50)
entry_path.grid(row=0, column=1, padx=(0, 5))

# Tạo nút "Open Image"
button_open_image = tk.Button(frame, text="Open Image", command=open_image, font=("Arial", 10), width=10, bg='blue', fg='white')
button_open_image.grid(row=0, column=0, padx=(0, 5))

# Tạo nút "Open Webcam"
button_open_webcam = tk.Button(frame, text="Open Webcam", command=webcam, font=("Arial", 10), width=10, bg='green', fg='white')
button_open_webcam.grid(row=0, column=2, padx=(0, 5))

# Tạo nút "Clear Image"
button_clear_image = tk.Button(frame, text="Clear Image", command=clear_image, font=("Arial", 10), width=10, bg='red', fg='white')
button_clear_image.grid(row=0, column=3, padx=(0, 5))

# Tạo nút "Thoát"
button_exit = tk.Button(frame, text="Exit", command=window.quit, font=("Arial", 10), width=10, bg='gray', fg='white')
button_exit.grid(row=0, column=4, padx=(0, 5))

# Tạo khung chứa kết quả phân loại
result_frame = tk.Frame(window)
result_frame.pack(pady=(10, 0))

# Nhãn để hiển thị kết quả phân loại
result_label = tk.Label(result_frame, text="", font=("Arial", 12))
result_label.pack()

# Tạo khung chứa nhãn ảnh
image_frame = tk.Frame(window, bd=2, relief="groove", width=400, height=400)
image_frame.pack(pady=(10, 0))
image_frame.pack_propagate(False)  # Ngăn chặn frame thay đổi kích thước theo nội dung

# Nhãn để hiển thị ảnh
label = tk.Label(image_frame)
label.pack(expand=True)

window.mainloop()
