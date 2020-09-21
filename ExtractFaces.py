import os
import cv2
import sys
#image_path = "data-faces/test/Introvert/"
image_path='data-faces/Extrovert/'

def crop_face(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=7,minSize=(30, 30))

    print("[INFO] Found {0} Faces.".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)

        #status = cv2.imwrite('faces_detected.jpg', image)
        #print("[INFO] Image faces_detected.jpg written to filesystem: ", status)


def save_faces(cascade, imgname):
    img = cv2.imread(os.path.join(image_path, imgname))
    for i, face in enumerate(cascade.detectMultiScale(img, scaleFactor=1.3,minNeighbors=5,minSize=(30, 30))):
        x, y, w, h = face
        sub_face = img[y:y + h, x:x + w]
        #cv2.imshow('image',sub_face)
        #cv2.waitKey(0)
        sub_face = cv2.resize(sub_face, (256,256), interpolation = cv2.INTER_AREA)
        #cv2.imwrite(os.path.join("faces", "{}_{}.jpg".format(imgname, i)), sub_face)
        cv2.imwrite(image_path+str(w) + str(h) + str(i)+'_faces.jpg',sub_face)
#if __name__ == '__main__':
face_cascade = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(face_cascade)
# Iterate through files
for f in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]:
    #crop_face(f)#
    save_faces(cascade, f)
    #print(f)
