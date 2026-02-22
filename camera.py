import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import pandas as pd

# ---------------- LOAD FACE CASCADE ----------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ---------------- BUILD MODEL ----------------
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.load_weights('model.h5')

# ---------------- EMOTION LABELS ----------------
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# ---------------- SONG FILES ----------------
music_dist = {
    0: "songs/angry.csv",
    1: "songs/disgusted.csv",
    2: "songs/fearful.csv",
    3: "songs/happy.csv",
    4: "songs/neutral.csv",
    5: "songs/sad.csv",
    6: "songs/surprised.csv"
}

show_text = [4]  # Default Neutral

# ---------------- EMOTION PREDICTION ----------------
def predict_emotion_from_image(img):

    global show_text

    image = cv2.resize(img, (600, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion_label = "Neutral"

    for (x, y, w, h) in face_rects:
        roi_gray_frame = gray[y:y + h, x:x + w]

        cropped_img = cv2.resize(roi_gray_frame, (48, 48))
        cropped_img = np.reshape(cropped_img, (1, 48, 48, 1))

        prediction = emotion_model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(prediction))

        show_text[0] = maxindex
        emotion_label = emotion_dict[maxindex]

    return emotion_label


# ---------------- MUSIC RECOMMENDATION ----------------
def music_rec():
    df = pd.read_csv(music_dist[show_text[0]])
    df = df[['Name', 'Album', 'Artist']]
    df = df.head(15)
    return df
	
	