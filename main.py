import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PoseModule import PoseDetector
import numpy as np
from PIL import ImageGrab
import src.utils as ut
import configuration.config as cfg
import src.window_GUI as win


def main():

    camera = cv2.VideoCapture(0)
    image_elem, record_button, window = win.create_GUI()

    number_file = 0
    pTime = 0
    detector_face = FaceMeshDetector(maxFaces=1)
    detector_pose = PoseDetector()
    face_detection = False
    eyes_detection = True
    draw_eyes = False
    recording = False
    game_frame = False
    show_face_detection = True
    moving = False
    car_mode_on = False

    while camera.isOpened():


        event, values = window.read(timeout=0)

        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)

        try:
            if face_detection:
                frame_copy = frame.copy()
                frame, faces = detector_face.findFaceMesh(frame)

                if draw_eyes:
                    ut.draw_eyes(frame, faces)

                if eyes_detection:
                    image_eyes = ut.eyes_detection(frame_copy, faces)

                    output = cv2.resize(image_eyes, [120, 40])
                    output = ut.increase_brightness(output, 10)
                    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

                    cv2.imshow('just eyes', gray)
                    cv2.moveWindow('just eyes', 540, 10)

                    movement = ut.predict(gray)

                    car_mode_on, moving = ut.do_action(movement, car_mode_on, moving)


            if game_frame:
                img_game = ImageGrab.grab(bbox=(0, 300, 800, 800))
                img_game_np = np.array(img_game)
                frame_game = ut.detect_pose(img_game_np, detector_pose)
                cv2.imshow("frame_game", frame_game)

        except Exception as err:
            print(err)


        window, cfg.path, recording, face_detection, game_frame, show_face_detection = win.check_events(event, window,
                                                                                                        cfg.path,
                                                                                                        recording,
                                                                                                        face_detection,
                                                                                                        game_frame,
                                                                                                        show_face_detection,
                                                                                                        record_button)

        pTime = ut.window_display_info(frame, cfg.path, pTime, car_mode_on, face_detection)

        if show_face_detection:
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            image_elem.update(data=imgbytes)

        if recording:
            frame, faces = detector_face.findFaceMesh(frame)
            frame_copy = frame.copy()
            image_eyes = ut.eyes_detection(frame_copy, faces)
            number_file = win.record(image_eyes, cfg.path, number_file)


        # Reading the key
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

        # k to close the PySimpleGUI window for better performance
        if k == ord("k"):
            show_face_detection = not show_face_detection
            if show_face_detection == True:
                image_elem, record_button, window = win.create_GUI()
            else:
                window.close()


        if k == ord("l"):
            face_detection = not face_detection
            if not face_detection:
                cv2.destroyWindow('just eyes')
            print("face_detection ", face_detection)

        if k == ord("e"):
            draw_eyes = not draw_eyes

        if k == ord("b"):
            if cfg.x == -150:
                cfg.x = -25
                cfg.time_ = 0.2
            else:
                cfg.x = -150
                cfg.time_ = 0.1


main()
