import cv2
import dlib
import numpy as np
import serial
from imutils import face_utils
from time import time

yaw_threshold = 15
# ser = serial.Serial('/dev/ttyUSB0')
ser = None
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cam = None
frame_width, frame_height = 352, 288
run_app = True
WINDOW_TITLE = 'Head Tracking'


def draw_object(frame, points):
    for x, y in points:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


def minmax_object(points):
    min = np.amin(points, axis=0)
    max = np.amax(points, axis=0)

    return min, max


def object_center(points):
    return np.mean(points, axis=0)


def send_serial(signal):
    if ser is not None and ser.isOpen():
        ser.write(signal)


def free_and_close():
    global run_app, cam

    run_app = False

    if cam is not None and cam.isOpened():
        cam.release()

    if ser is not None and ser.isOpen():
        send_serial('S')
        ser.close()

    cv2.destroyAllWindows()


def close_button_listener(event, x, y, flags, param):
    global frame_height

    if event == cv2.EVENT_LBUTTONUP and y > frame_height:
        free_and_close()


def main():
    global frame_width, frame_height, run_app

    cam = cv2.VideoCapture(0)

    cam.set(3, frame_width)
    cam.set(4, frame_height)

    close_button = np.zeros((80, frame_width, 3), dtype=np.uint8)
    cv2.putText(close_button, 'Close', (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow(WINDOW_TITLE)
    cv2.setMouseCallback(WINDOW_TITLE, close_button_listener)

    directions = {'L': 'LEFT',
                  'R': 'RIGHT',
                  'G': 'FORWARD',
                  'S': 'STOP'}

    while run_app:
        t = time()
        ret, frame = cam.read()

        if not ret:
            print('Could not grab frame')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        if len(rects) == 0:
            direction = 'S'
        else:
            shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            nose = shape[27:36]
            jaw = shape[0:17]

            draw_object(frame, nose)
            draw_object(frame, jaw)

            nose_center = object_center(nose)
            face_center = object_center(jaw)
            face_minmax = minmax_object(jaw)
            face_width = face_minmax[1][0] - face_minmax[0][0]

            horizontal_threshold = face_width / yaw_threshold

            if nose_center[0] > (face_center[0] + horizontal_threshold):
                direction = 'L'
            elif nose_center[0] < (face_center[0] - horizontal_threshold):
                direction = 'R'
            else:
                direction = 'G'

        send_serial(direction)

        frame = np.vstack((frame, close_button))

        cv2.putText(frame, directions[direction], (frame_width - 220, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255),
                    1, cv2.LINE_AA)
        cv2.imshow(WINDOW_TITLE, frame)

        fps = 1 / (time() - t + .00000001)

        print('FPS: ', fps)

        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    main()