from random import randrange as rand
import csv
import os
import sys
import cv2
import pickle
import datetime

print(f"Arguments count: {len(sys.argv)}")
for i, arg in enumerate(sys.argv):
    print(arg)


def check_accuracy(possible_ids):
    if len(possible_ids) < 15:
        possible_ids.append(cms_id)
    elif len(possible_ids) > 15:
        possible_ids.clear()
    else:
        first_id_count = 0
        second_id_count = 0
        first_id = possible_ids[0]
        second_id = ""
        for item in possible_ids:
            if second_id == "" and item != first_id:
                second_id = item

            if item == first_id:
                first_id_count += 1
            if item == second_id:
                second_id_count += 1
        if first_id_count > second_id_count:
            set_.add(first_id)

        else:
            set_.add(second_id)
        cv2.putText(frame, "ATTENDANCE MARKED", (x - 70, y - 50), font, 1, color, stroke, cv2.LINE_AA)


def mark_attendance(face_id):
    # create a folder with all the attendance files if it doesnt exist
    dir = os.path.dirname(os.path.abspath(__file__)) + r"\Attendance Files"
    if not os.path.exists(dir):
        os.mkdir(dir)
    # then inside that folder create a subject folder, attendance of that subject will be stored in there
    if not os.path.exists(os.path.join(dir, subject)):
        os.mkdir(os.path.join(dir, subject))

    date_ = datetime.date.today()
    time_ = datetime.datetime.now().strftime("%H:%M:%S")
    # creating a csv file in the subject folder
    filename = os.path.join(os.path.join(dir, subject), subject + " " + str(date_) + ".csv")
    headings = ("CMS ID ", "Name", "Time", "Attendance")
    data_list = [face_id, name, str(time_), "Present"]
    with open(filename, 'a', newline="") as csvfile:
        move = csv.writer(csvfile)
        if num_students == 0:
            move.writerow(headings)
        move.writerow(data_list)
        print(str(data_list[1]) + "'s" + " attendace marked.")


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("names.pickle", 'rb') as file:
    names = pickle.load(file)

with open("cms_ids.pickle", 'rb') as file:
    CMS_IDS = pickle.load(file)
capture = cv2.VideoCapture(0)

subject = sys.argv[1].upper()

num_students = 0
set_ = set()
accuracy = []

while True:
    is_true, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        person_face = gray[y:y + h, x:x + w]
        person_id, inaccuracy = recognizer.predict(person_face)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)
        stroke = 1
        color_rect = (rand(0, 256), rand(0, 256), rand(0, 256))
        # only print names if prediction less inaccurate for more accurate results
        if inaccuracy < 80:
            name = names[person_id]
            cms_id = CMS_IDS[person_id]
            stroke = 1
            cv2.putText(frame, name, (x, y - 5), font, 1, color, stroke, cv2.LINE_AA)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color_rect, 2)
            check_accuracy(accuracy)
            cv2.putText(frame, "ATTENDANCE MARKED", (x - 70, y - 50), font, 1, color, stroke, cv2.LINE_AA)
        else:
            text = "PERSON NOT REGISTERED"
            cv2.putText(frame, text, (x - 100, y), font, 1, color, stroke, cv2.LINE_AA)
            color = (rand(0, 256), rand(0, 256), rand(0, 256))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('Attendance', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# storing cms ids in a set and then calling the mark attendance function so attendance is marked only once
for cms_id in set_:
    name = names[CMS_IDS.index(cms_id)]
    mark_attendance(cms_id)
    num_students += 1

capture.release()
cv2.destroyAllWindows()