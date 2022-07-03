import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime


# create employees list
rute = "Employees\\Employees"
pictures = []
emp_names = []
emp_list = os.listdir(rute)

for n in emp_list:
    actual_image = cv2.imread(f"{rute}\\{n}")
    pictures.append(actual_image)
    emp_names.append(os.path.splitext(n)[0])


# code images
def code(images):

    # new list
    coded_list = []

    # turn every image to RGB
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # code
        coded = fr.face_encodings(image)[0]

        # append to list
        coded_list.append(coded)

    # return coded list
    return coded_list


# register admissions
def admissions(employee):
    f = open("admissions.csv", "r+")
    data_list = f.readlines()
    adm_names = []
    for line in data_list:
        ingress = line.split(",")
        adm_names.append(ingress[0])

    if employee not in adm_names:
        now = datetime.now()
        string_now = now.strftime("%H:%M:%S")
        f.writelines(f"\n{employee}, {string_now}")


emp_clist = code(pictures)

# capture webcam image
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# read webcam image
success, image = capture.read()

if not success:
    print("Couldn't take capture")
else:
    # face recognition
    capture_face = fr.face_locations(image)

    # code captured face
    c_capture_face = fr.face_encodings(image, capture_face)

    # search coincidences
    for cod_face, ub_face in zip(c_capture_face, capture_face):
        coincidences = fr.compare_faces(emp_clist, cod_face)
        distances = fr.face_distance(emp_clist, cod_face)

        print(distances)

        coincidence_index = numpy.argmin(distances)

        # show coincidences
        if distances[coincidence_index] > 0.6:
            print("Couldn't find any coincidence in our employee database")

        else:

            # find employee name
            name = emp_names[coincidence_index]

            y1, x2, y2, x1 = ub_face
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            admissions(name)

            # show captured image
            cv2.imshow("Webcam image", image)

            # Keep window open
            cv2.waitKey(0)
