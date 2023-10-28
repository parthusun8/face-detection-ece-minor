import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import numpy as np
from deepface import DeepFace
import gspread

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(imgModeList))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")

sa = gspread.service_account_from_dict({
    "type": "service_account",
    "project_id": "srmasv-grader",
    "private_key_id": "ad28c202755298e854d1a6ecfd6f1806ba60653c",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCrcWpdhASvnH4s\nBoEHNN+Dg2uaEZT2u2fkInZ6/SaA08rQskxl+0c30lmWw9ZkGbF9gOXBFTpuLu7T\n00TKEQ14G4dXz+d+4Eig2Krtgj8ZvUF5SbQUYaOnhSeje/fxp/V/DPa7/GgcjOkk\nhg+kxKlZe1p29Hvza0Vkuhu6rGtRmcn6SzpF5AQf5+/UFl0PyOeviUNsQ31RG9Ab\nVIRuPjCB8/CfJcGaB+v4mpTzb49yvX+or2cz/D+oJq9En/OaST3BRSrfpuqFGiv8\nAsaw1lEikov1bf9G534chslbFw1suPmm/TfOcrra6hrta7n+Vth2/eogxI7FuJwy\nmBM7oFVVAgMBAAECggEABy29EURdY925ehjISf0gnM5McPY/1rhQfdYkQDsduR2P\n1lRDlwOXXv2BAICbwgjstyobX2MxqqUZcqeGhQWDcF41Y8ESBZcuQ1np3P8Wqvpo\nkEO8oBvZ2+0F6gu0p/BApdIXpMB0EDn66yttN+4qKdPcnQguBn/H4eivD453rJd3\nzgqk8g10MgjqaxyYpC0DhDHAMwsYyLoIx6b/js2VRnyHGFs2DT89d6QXQgkbWa7Q\nnmZLhFxesvfFjUUSm+cKgcstAEZif1/pK00/vougvny2pQPVFJpXcJjwgCPLGLGJ\nRaIkFBiSpkKwjUrBZgIGwWF/9p8iswfGVDFCxsfS0wKBgQDnHtgDGhUwJbaV+yek\nBHzT0f7UQ6pOJ/SocGfUovAOkxdgb6MxM1L5n+nUzU3jJc7twy/rSaZGx3vfrAWt\naFZFs1bfM7M8qDjQaScAmE2DRKTLFCNufpgiwGwj4GvVeCuphgHQq0HCIqZHRuNx\nHzuKlZx4bf6gim40adpOYnBQnwKBgQC95f8WdH1BRCUWfYYPb22gyjCDg8atuTUx\nvbJAtKtmAwfDjUiIDkXdyZB/ZQ1I19rHtD/iaOCIE1NlwY9yzhXCMhAttwxgER1y\neMa/mbBoyxad//RPqEnBbGTtHhlS7rmBBwZYNGeNv6ZXeCOfsuUQNainUACQUioU\nE3kIlnYRiwKBgQCcxDIXIIkAqIQJhVslCJo2/ziPd/o7myR7YAs9kuurpx/zHKYf\nyEVe5pYv7rYG/e03Hu8Q9FrhpYujcsZDEgN6saOaMDJCUYR/8OMwpx3kjRoOWXGT\nEDFXWRfA2gequyE6kpgGlzM6YFBTBoVdhKbZYJPKqClPcbZAvRADnQe71wKBgEFv\nmqnrNCokSD9qu6Jf/D/WzEbVRsYgNvNP8beYSiOZ0bgR3Dd965dUWKV9dclvECWW\nuBjVmOFq/2bl/v9JgnfrdmW6WNpVq3cBpULqu74wlTeWtmIolFnBdzm6EHHibyvF\n9uq4DCLtMm2bqXLjW0ltpBMbX0Zb+cH3P9K0vjSnAoGBAKWq8JSTPTuvx6YQaa1j\nUoBToQrzx2usIZL9mvMbjMVL4E+Li+SoHaCFXEFCDllQGaNVFn1bFadHcBqut+64\ntzz+PhrTwDItwZKf5dSiz8JjE5HTLTX8cISfKVVGONimLRkk2ujRbVTk8jmoVrWa\nfwa7IL2PQOs6Gow3yvRq9Lpl\n-----END PRIVATE KEY-----\n",
    "client_email": "serviceaccount@srmasv-grader.iam.gserviceaccount.com",
    "client_id": "109484706480290888675",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/serviceaccount%40srmasv-grader.iam.gserviceaccount.com"
  })

sh = sa.open("EMPLOYEE EMOTION RECORDS")

wks = sh.worksheet("Sheet1")

records = wks.get_all_records()
studentInfos = {}
for index, i in enumerate(records):
    print(i)
    studentInfos[i['Reg No']] = i
    studentInfos[i['Reg No']]['index'] = str(index + 2)


modeType = 0
counter = 0
id = -1
imgStudent = []

studentInfo = {
    "total_attendance": "total",
    "major": "major",
    "standing": "standing",
    "year": "year",
    "starting_year": "starting_year",
    "name": "name"
}
while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print("matches", matches)
            # print("faceDis", faceDis)

            matchIndex = np.argmin(faceDis)
            # print("Match Index", matchIndex)

            if matches[matchIndex]:
                print("Known Face Detected")
                print(studentIds[matchIndex])
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]
                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1
            else:
                modeType = 1

        if counter != 0:
            if counter == 1:
                pass
                # Get the Data
                # studentInfo = db.reference(f'Students/{id}').get()
                # print(studentInfo)
                # Get the Image from the storage
                # blob = bucket.get_blob(f'Images/{id}.png')
                # array = np.frombuffer(blob.download_as_string(), np.uint8)
                # imgStudent = cv2.imdecode(imgStudent, cv2.COLOR_BGRA2BGR)
                # Update data of attendance
                # datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                #                                    "%Y-%m-%d %H:%M:%S")
                # secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                # print(secondsElapsed)
                # if secondsElapsed > 30:
                #     # ref = db.reference(f'Students/{id}')
                #     studentInfo['total_attendance'] += 1
                #     # ref.child('total_attendance').set(studentInfo['total_attendance'])
                #     # ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                # else:
                #     modeType = 3
                #     counter = 0
                #     imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:

                # if 10 < counter < 20:
                #     modeType = 2

                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                print(result[0]['dominant_emotion'])
                studentInfos[id]["DAY 4"] = result[0]['dominant_emotion']
                print(f"I{studentInfos[id]['index']}")
                wks.update("I" + studentInfos[id]['index'], studentInfos[id]["DAY 4"])

                if True:
                    # cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                    #             cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfos[id]['Major']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfos[id]['DAY 4']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfos[id]['Passing_Year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfos[id]['Year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w, h), _ = cv2.getTextSize(studentInfos[id]['Name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackground, str(studentInfos[id]['Name']), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    imgStudent = cv2.imread("Images/" + studentIds[matchIndex] + ".png")
                    imgStudent = cv2.resize(imgStudent, (216, 216))

                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

                counter += 1

                # if counter >= 20:
                #     counter = 0
                #     modeType = 0
                #     studentInfo = []
                #     imgStudent = []
                #     imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        pass
        # modeType = 0
        # counter = 0
    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
