import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path ='dataSet1'

def getImagesAndLabels(path):
    #Lấy đường dẫn của các file trong folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces =[]
    IDs =[]
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        #Chia tên file để lấy ID của ảnh
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print (ID)
        IDs.append(ID)
        cv2.imshow("traning",faceNp)
        cv2.waitKey(10)
    return IDs, faces

Ids, faces = getImagesAndLabels(path)

#training
recognizer.train(faces,np.array(Ids))
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
