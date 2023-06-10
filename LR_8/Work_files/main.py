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

    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    if pic_or_cam:
        frame = cv2.imread(pic_or_cam)
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        while cv2.waitKey(1) < 0:
            cv2.imshow("Laboratory work #8", resultImg)

    else:
        video = cv2.VideoCapture(0)

        while cv2.waitKey(1) < 0:
            hasFrame, frame = video.read()

            if not hasFrame:
                cv2.waitKey
                break

            resultImg, faceBoxes = highlightFace(faceNet, frame)

            cv2.imshow("Laboratory work #8", resultImg)


if __name__ == '__main__':
    print("Откуда брать иззображение?\n С камеры - cam\n С файла - file")
    data = input("Введите источник: ")
    if data == 'cam':
        findFaces()
    elif data == 'file':
        file = input("Введите полное название файла: ")
        print("Название: ", file)
        findFaces(file)
    else:
        print("Неверный формат!")
