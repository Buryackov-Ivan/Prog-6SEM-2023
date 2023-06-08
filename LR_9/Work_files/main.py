import cv2


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (255, 125, 10), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes


def findFaces(pic_or_cam=None):
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    genderList = ['Male ', 'Female']
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)

    if pic_or_cam:
        frame = cv2.imread(pic_or_cam)
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]):
                        min(faceBox[3], frame.shape[0]-1), max(0, faceBox[0]):min(faceBox[2], frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (
                faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        while cv2.waitKey(1) < 0:
            cv2.imshow("Laboratory work #9", resultImg)

    else:
        video = cv2.VideoCapture(0)

        while cv2.waitKey(1) < 0:
            hasFrame, frame = video.read()

            if not hasFrame:
                cv2.waitKey
                break

            resultImg, faceBoxes = highlightFace(faceNet, frame)

            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1]):
                             min(faceBox[3], frame.shape[0]-1), max(0, faceBox[0]):min(faceBox[2], frame.shape[1]-1)]
                blob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                print(f'Gender: {gender}')

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                print(f'Age: {age[1:-1]} years')

                cv2.putText(resultImg, f'{gender}, {age}', (
                    faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Laboratory work #9", resultImg)


if __name__ == '__main__':
    findFaces()  # Распознавание с камеры
    #findFaces('./photodune-2359510.jpg')  # Распознавание с фото