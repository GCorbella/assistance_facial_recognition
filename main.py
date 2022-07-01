import cv2
import face_recognition as fr

# load images
control_photo = fr.load_image_file("yo_tunel.jpg")
test_photo = fr.load_image_file("yo_vino.jpg")

# change images to RGB
control_photo = cv2.cvtColor(control_photo, cv2.COLOR_BGR2RGB)
test_photo = cv2.cvtColor(test_photo, cv2.COLOR_BGR2RGB)

# locate control face
face_A_place = fr.face_locations(control_photo)[0]
face_A_coded = fr.face_encodings(control_photo)[0]

# locate test face
face_B_place = fr.face_locations(test_photo)[0]
face_B_coded = fr.face_encodings(test_photo)[0]

# show rectangle
cv2.rectangle(control_photo,
              (face_A_place[3], face_A_place[0]),
              (face_A_place[1], face_A_place[2]),
              (0, 255, 0)
              )
cv2.rectangle(test_photo,
              (face_B_place[3], face_B_place[0]),
              (face_B_place[1], face_B_place[2]),
              (0, 255, 0)
              )

# show images
cv2.imshow("yo_tunel.jpg", control_photo)
cv2.imshow("yo_vino.jpg", test_photo)

# keep program open
cv2.waitKey(0)