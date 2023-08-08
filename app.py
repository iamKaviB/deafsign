from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import tensorflow as tf
import pickle
import nltk
import os
import cv2
from collections import Counter


IMAGE_SIZE = [224, 224]
stemmer = nltk.stem.PorterStemmer()


app = Flask(__name__)
CORS(app)

classes1 = pickle.load(open("./models/en/imageNames.pkl", 'rb'))
words1 = pickle.load(open("./models/en/words.pkl", 'rb'))
model1 = tf.keras.models.load_model("./models/en/model_english_sign.h5")

classes2 = pickle.load(open("./models/sl/imageNames.pkl", 'rb'))
words2 = pickle.load(open("./models/sl/words.pkl", 'rb'))
model2 = tf.keras.models.load_model("./models/sl/model_sinhala_sign.h5")

model3 = tf.keras.models.load_model('./models/video/video.h5')
classes3 = [
    "bad","beautiful",
    "can","cancel","child","cold","comeback","cough",
    "date","don't know","drink",
    "easy","eat","eight","eighty",
    "fifty","fine","five","forty","four","free",
    "give","good","good night",
    "have a good day","help","how","how much","headache","hungry",
    "like","listen","love",
    "me","monitoring","mother",
    "nine","ninety",
    "one","one hundred",
    "seven","seventy","six","sixty","small","soon","speak",
    "ten","thank you","they","three","time","today","tomorrow","try",
    "welcome","what time is it","who","why","write",
    "your",
    "zero"
]


def clean_up_sentence(sentence):
    # tokenize
    sentence_words = nltk.word_tokenize(sentence.lower())
    return sentence_words


def predict_en_sign(sentence):
    # tokenize
    sentence_words = clean_up_sentence(sentence)

    return_list = []
    for s in sentence_words:
        # matrix of N words
        bag = [0] * len(words1)

        for i, w in enumerate(words1):
            if w == stemmer.stem(s):
                bag[i] = 1

        res = model1.predict(np.array([bag]))[0]
        ERROR_THRESHOLD = 0.08
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        print(results)

        if results:
            predicted_class_index = results[0][0]
            return_list.append(classes1[predicted_class_index])
        else:
            return_list.append("")

    return return_list


def predict_sl_sign(sentence):
    ERROR_THRESHOD = 0.01
    SECOND_WORD = ""
    SECOND_WORD_THRESHOD = 0.2
    CONSIDER_SECOND_WORD = 0
    # tokenize
    sentence_words = clean_up_sentence(sentence)
    predictList = []
    return_list = []
    for s in sentence_words:
        # matrix of N words
        bag = [0] * len(words2)

        for i, w in enumerate(words2):

            if w == s:
                bag[i] = 1

            if CONSIDER_SECOND_WORD == 1 and SECOND_WORD == w:
                bag[i] = 1
                CONSIDER_SECOND_WORD = 0

        res = model2.predict(np.array([bag]))[0]

        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOD]

        # sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        if results:
            print(results[0][0])
            return_list.append(classes2[results[0][0]])
        else:
            print("huhu")
            return_list.append("s")

        # return_list.append(classes2[results[0][0]]['groups'])

    return return_list


def preprocess_frame(frame):
    # Resize the frame to the desired image size
    resized_frame = cv2.resize(frame, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    preprocessed_frame = np.expand_dims(resized_frame, axis=0)
    return preprocessed_frame

def predict_frame(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    prediction = model3.predict(preprocessed_frame)


    return prediction


def split_video(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    count = 0
    all_pred = []
    while count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frame_output_path = f"output/frame_{count}.jpg"
            cv2.imwrite(frame_output_path, frame)

            img_array = cv2.imread(frame_output_path)
            resized_arr = cv2.resize(img_array, (224, 224))
            x = np.array(resized_arr) / 255
            x = x.reshape(-1, 224, 224,3)
            predic = model3.predict(x)
            index_of_max_value = np.argmax(predic[0])
            all_pred.append(index_of_max_value)

            print("predicting",index_of_max_value)

        count += 1

    cap.release()
    counter = Counter(all_pred)
    print(all_pred)
    most_common_value, _ = counter.most_common(1)[0]
    print(most_common_value)
    return most_common_value
def convert_videos_to_frames(video_path, frames_per_video=30):
    return split_video(video_path)




# english sentence
@app.route('/predict/en', methods=['POST'])
def predict_en():
    sentence = request.json['sentence']
    result = predict_en_sign(sentence)

    res = {
        "result": result
    }

    return jsonify(res)


# sinhala sentence
@app.route('/predict/sl', methods=['POST'])
def predict_sl():
    sentence = request.json['sentence']
    result = predict_sl_sign(sentence)

    res = {
        "result": result
    }

    return jsonify(res)

#video detection
@app.route('/predict/video',methods=['POST'])
def predict_video():
    # Check if files were sent in the request
    if 'files' not in request.files:
        return "No files were uploaded.", 400

    # Get the list of uploaded files
    uploaded_files = request.files.getlist('files')

    # Save the uploaded files to a temporary location
    temp_folder = 'temp_videos'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    video_paths = []
    for file in uploaded_files:
        video_path = os.path.join(temp_folder, "file")
        file.save(video_path)
        video_paths.append(video_path)

        # img_array = cv2.imread(video_path)
        # resized_arr = cv2.resize(img_array, (224, 224))
        # x = np.array(resized_arr) / 255
        #
        # x = x.reshape(-1, 224, 224,3)
        # predic = model3.predict(x)
        # index_of_max_value = np.argmax(predic[0])
        #
        # print(index_of_max_value)



    # output_folder = 'output_frames'
    result = convert_videos_to_frames(video_path)
    print(result)

    res = {
        "result": classes3[int(result)]
    }

    return jsonify(res)



if __name__ == '__main__':
    app.run()
