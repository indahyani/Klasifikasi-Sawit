import streamlit as st
from PIL import Image
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler 


# read and modificated dataset RGB
RGB = pd.read_csv('RGB.csv')
x_RGB = RGB.drop(["Unnamed: 0"], axis=1)
y_RGB = RGB["Class"]
x_RGB = x_RGB.drop(["Class"], axis=1)
dx_RGB = RGB.drop(["Unnamed: 0"], axis=1)

# read and modificated dataset YCBCR
YCBCR = pd.read_csv('YCRCB.csv')
x_YCBCR = YCBCR.drop(["Unnamed: 0"], axis=1)
y_YCBCR = YCBCR["Class"]
x_YCBCR= x_YCBCR.drop(["Class"], axis=1)
dx_YCBCR = YCBCR.drop(["Unnamed: 0"], axis=1)

# read and modificated dataset HSV
HSV = pd.read_csv('HSV.csv')
x_HSV = HSV.drop(["Unnamed: 0"], axis=1)
y_HSV = HSV["Class"]
x_HSV = x_HSV.drop(["Class"], axis=1)
dx_HSV = HSV.drop(["Unnamed: 0"], axis=1)

# read and modificated dataset YUV
YUV = pd.read_csv('YUV.csv')
x_YUV = YUV.drop(["Unnamed: 0"], axis=1)
y_YUV = YUV["Class"]
x_YUV = x_YUV.drop(["Class"], axis=1)
dx_YUV = YUV.drop(["Unnamed: 0"], axis=1)

# read and modificated dataset LAB
LAB = pd.read_csv('LAB.csv')
x_LAB = LAB.drop(["Unnamed: 0"], axis=1)
y_LAB = LAB["Class"]
x_LAB = x_LAB.drop(["Class"], axis=1)
dx_LAB = LAB.drop(["Unnamed: 0"], axis=1)


# read and modificated dataset XYZ
XYZ = pd.read_csv('XYZ.csv')
x_XYZ = XYZ.drop(["Unnamed: 0"], axis=1)
y_XYZ = XYZ["Class"]
x_XYZ = x_XYZ.drop(["Class"], axis=1)
dx_XYZ = XYZ.drop(["Unnamed: 0"], axis=1)

st.title("PERBANDINGAN MODEL WARNA DALAM MENGKLASIFIKASI TINGKAT KEMATANGAN SAWIT MENGGUNAKAN ALGORITMA K-NEAREST NEIGHBORS")
uploadFile = st.file_uploader(label="Open Image", type=['png'])
# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image
# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(uploadFile)
    st.image(img)
    # st.write("Image Uploaded Successfully")
else:
    st.write("Make Sure your image format is PNG")

st.sidebar.title("Informasi Umum")
def informasi_umum():
    st.sidebar.markdown("Dataset RGB")
    st.sidebar.dataframe(dx_RGB)
    st.sidebar.markdown("Dataset YCBCR")
    st.sidebar.dataframe(dx_YCBCR)
    st.sidebar.markdown("Dataset HSV")
    st.sidebar.dataframe(dx_HSV)
    st.sidebar.markdown("Dataset YUV")
    st.sidebar.dataframe(dx_YUV)
    st.sidebar.markdown("Dataset LAB")
    st.sidebar.dataframe(dx_LAB)
    st.sidebar.markdown("Dataset XYZ")
    st.sidebar.dataframe(dx_XYZ)
    # st.sidebar(plot3D)

df = informasi_umum()
d_RGB = st.sidebar.button("3D PLOT RGB")
if d_RGB:
        Red = dx_RGB["R"]
        Green = dx_RGB["G"]
        Blue = dx_RGB["B"]
        Class = dx_RGB["Class"]
        dplot = px.scatter_3d(RGB,  x=Red, y=Green, z=Blue,
                color=Class)
        st.write(dplot)
# button crop
# crop_mulai = st.button("Crop")

# if crop_mulai:
#         # Select ROI
#         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         r = cv2.selectROI("select the area", img)
#         # Crop image
#         cropped_image = img[int(r[1]):int(r[1]+r[3]), 
#                             int(r[0]):int(r[0]+r[2])]
        
#         # Display cropped image
#         cv2.imshow("Cropped image", cropped_image)
#         cropped_image  = cv2.resize(cropped_image, dsize=(90, 70))
#         st.image(cropped_image)


Proses = st.button("Proses")


if Proses:
    #get mean color models from image
    # img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    avg_color_per_row_RGB = np.average(img, axis=0)
    avg_color_RGB = np.average(avg_color_per_row_RGB, axis=0)
    avg_color_RGB = np.array([avg_color_RGB])
    # st.code(avg_color)
    # x = x['Class'] = x.Class.replace({'Matang':0, 'Mentah':1, 'Mengkal':2})
    # knn using color model RGB
    x_train_RGB, x_test_RGB, y_train_RGB, y_test_RGB = train_test_split(x_RGB, y_RGB, shuffle=True, test_size=0.20, random_state=2)
    scaler = StandardScaler()  
    scaler.fit(x_train_RGB)
    x_train_RGB = scaler.transform(x_train_RGB)  
    x_test_RGB = scaler.transform(x_test_RGB)
    knn_RGB = KNeighborsClassifier(n_neighbors=5)
    knn_RGB.fit(x_train_RGB, y_train_RGB)
    hasil_RGB = scaler.transform(avg_color_RGB)
    prediksi_RGB = knn_RGB.predict(hasil_RGB)
    

    #  knn using color model YCBCR
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_YCBCR = cv2.cvtColor(img_RGB,cv2.COLOR_RGB2YCR_CB)
    avg_color_per_row_YCBCR = np.average(img_YCBCR, axis=0)
    avg_color_YCBCR = np.average(avg_color_per_row_YCBCR, axis=0)
    avg_color_YCBCR = np.array([avg_color_YCBCR])
    x_train_YCBCR, x_test_YCBCR, y_train_YCBCR, y_test_YCBCR = train_test_split(x_YCBCR, y_YCBCR, shuffle=True, test_size=0.20, random_state=2)
    scaler = StandardScaler()  
    scaler.fit(x_train_YCBCR)
    x_train_YCBCR = scaler.transform(x_train_YCBCR)  
    x_test_YCBCR = scaler.transform(x_test_YCBCR)
    knn_YCBCR = KNeighborsClassifier(n_neighbors=9)
    knn_YCBCR.fit(x_train_YCBCR, y_train_YCBCR)
    hasil_YCBCR = scaler.transform(avg_color_YCBCR)
    prediksi_YCBCR = knn_YCBCR.predict(hasil_YCBCR)
   

    # knn using color model HSV
    img_HSV = cv2.cvtColor(img_RGB,cv2.COLOR_RGB2HSV)
    avg_color_per_row_HSV = np.average(img_HSV, axis=0)
    avg_color_HSV = np.average(avg_color_per_row_HSV, axis=0)
    avg_color_HSV = np.array([avg_color_HSV])
    x_train_HSV, x_test_HSV, y_train_HSV, y_test_HSV = train_test_split(x_HSV, y_HSV, shuffle=True, test_size=0.20, random_state=2)
    scaler = StandardScaler()  
    scaler.fit(x_train_HSV)
    x_train_HSV = scaler.transform(x_train_HSV)  
    x_test_HSV = scaler.transform(x_test_HSV)
    knn_HSV = KNeighborsClassifier(n_neighbors=1)
    knn_HSV.fit(x_train_HSV, y_train_HSV)
    hasil_HSV = scaler.transform(avg_color_HSV)
    prediksi_HSV = knn_HSV.predict(hasil_HSV)
    
   # knn using color model YUV
    img_YUV = cv2.cvtColor(img_RGB,cv2.COLOR_RGB2YUV)
    avg_color_per_row_YUV = np.average(img_YUV, axis=0)
    avg_color_YUV = np.average(avg_color_per_row_YUV, axis=0)
    avg_color_YUV = np.array([avg_color_YUV])
    x_train_YUV, x_test_YUV, y_train_YUV, y_test_YUV = train_test_split(x_YUV, y_YUV, shuffle=True, test_size=0.20, random_state=2)
    scaler = StandardScaler()  
    scaler.fit(x_train_YUV)
    x_train_YUV = scaler.transform(x_train_YUV)  
    x_test_YUV = scaler.transform(x_test_YUV)
    knn_YUV = KNeighborsClassifier(n_neighbors=9)
    knn_YUV.fit(x_train_YUV, y_train_YUV)
    hasil_YUV = scaler.transform(avg_color_YUV)
    prediksi_YUV = knn_YUV.predict(hasil_YUV)
    
    # knn using color model LAB
    img_LAB = cv2.cvtColor(img_RGB,cv2.COLOR_RGB2LAB)
    avg_color_per_row_LAB = np.average(img_LAB, axis=0)
    avg_color_LAB = np.average(avg_color_per_row_LAB, axis=0)
    avg_color_LAB = np.array([avg_color_LAB])
    x_train_LAB, x_test_LAB, y_train_LAB, y_test_LAB = train_test_split(x_LAB, y_LAB, shuffle=True, test_size=0.20, random_state=2)
    scaler = StandardScaler()  
    scaler.fit(x_train_LAB)
    x_train_LAB = scaler.transform(x_train_LAB)  
    x_test_LAB = scaler.transform(x_test_LAB)
    knn_LAB = KNeighborsClassifier(n_neighbors=5)
    knn_LAB.fit(x_train_LAB, y_train_LAB)
    hasil_LAB = scaler.transform(avg_color_LAB)
    prediksi_LAB = knn_LAB.predict(hasil_LAB)
   
     # knn using color model XYZ
    img_XYZ = cv2.cvtColor(img_RGB,cv2.COLOR_RGB2XYZ)
    avg_color_per_row_XYZ = np.average(img_XYZ, axis=0)
    avg_color_XYZ= np.average(avg_color_per_row_XYZ, axis=0)
    avg_color_XYZ = np.array([avg_color_XYZ])
    x_train_XYZ, x_test_XYZ, y_train_XYZ, y_test_XYZ = train_test_split(x_XYZ, y_XYZ, shuffle=True, test_size=0.20, random_state=2)
    scaler = StandardScaler()  
    scaler.fit(x_train_XYZ)
    x_train_XYZ = scaler.transform(x_train_XYZ)  
    x_test_XYZ = scaler.transform(x_test_XYZ)
    knn_XYZ = KNeighborsClassifier(n_neighbors=8)
    knn_XYZ.fit(x_train_XYZ, y_train_XYZ)
    hasil_XYZ = scaler.transform(avg_color_XYZ)
    prediksi_XYZ = knn_XYZ.predict(hasil_XYZ)
    
    col1, col2 = st.columns(2)
    with col1: 
        st.subheader("Hasil Klasifikasi RGB")
        st.code(prediksi_RGB)
        st.subheader("Hasil Klasifikasi YCBCR")
        st.code(prediksi_YCBCR)
        st.subheader("Hasil Klasifikasi HSV")
        st.code(prediksi_HSV)
    with col2:
        st.subheader("Hasil Klasifikasi YUV")
        st.code(prediksi_YUV)
        st.subheader("Hasil Klasifikasi LAB")
        st.code(prediksi_LAB)
        st.subheader("Hasil Klasifikasi XYZ")
        st.code(prediksi_XYZ)

    

        # if proses_mulai:
        #         # ambil nilai gambar 
        #     avg_color_per_row = np.average(cropped_image, axis=0)
        #     avg_color = np.average(avg_color_per_row, axis=0)
        #     avg_color = np.array([avg_color])
        #     # x = x['Class'] = x.Class.replace({'Matang':0, 'Mentah':1, 'Mengkal':2})
        #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        #     scaler = StandardScaler()  
        #     scaler.fit(x_train)
        #     x_train = scaler.transform(x_train)  
        #     x_test = scaler.transform(x_test)
        #     knn = KNeighborsClassifier(n_neighbors=3)
        #     knn.fit(x_train, y_train)
        #     prediksi = knn.predict(avg_color)
        #     st.subheader("Hasil Klasifikasi")
        #     st.code(prediksi)




    
#     st.code("")
#     st.subheader("Akurasi")
#     st.code("")

#     st.code("")




   
