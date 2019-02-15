import cv2
import sqlite3
cam = cv2.VideoCapture(0);
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

#chèn dữ liệu vào sqlite
def insertOrUpdate(Id,Name):
    conn = sqlite3.connect("FaceBase.db");
    cmd = "SELECT * FROM People WHERE ID="+str(Id);
    cursor = conn.execute(cmd);
    isRecordExist =0;
    for row in cursor:
        isRecordExist=1
    if (isRecordExist ==1):
        cmd="UPDATE People SET Name="+str(Name)+"WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO People(ID,Name) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd);
    conn.commit();
    conn.close();

id = input('Nhap vao ID: ');
name = input('Nhap vao ten cua ban: ');
insertOrUpdate(id,name);
sampleNum =0
while(True):
    ret, img = cam.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h),(255,0,0), 2)

        sampleNum+=1
        cv2.imwrite("dataSet1/User."+id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])

        cv2.imshow('frame',img);
    if (cv2.waitKey(100) & 0xFF == ord('q')):
        break
    elif (sampleNum>20):
        break
cam.release()
cv2.destroyAllWindows()
    
