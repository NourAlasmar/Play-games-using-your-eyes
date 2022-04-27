import time
import cv2
import src.keyboard as keyboard
import mouse as ms
import configuration.config as cfg
import statistics
import numpy as np


buffer_predictions = []
def predict(frame):

    global buffer_predictions
    frame = cv2.resize(frame, (120, 40))
    frame = frame / 255
    frame = np.expand_dims(frame, axis=0)
    frame = np.expand_dims(frame, axis=-1)
    frame.shape
    prediction = cfg.new_model.predict(frame)

    if prediction.max() > cfg.threshold:
        key = np.argmax(prediction)
        if (len(buffer_predictions)) < cfg.buffer_size:
            buffer_predictions.append(key)
        else:
            buffer_predictions.append(key)
            buffer_predictions.pop(0)

        if key == 0 or key == 3:
            mode = statistics.mode(buffer_predictions)
            movement = cfg.classes[mode]
        else:
            movement = cfg.classes[key]

        return movement
    else:
        #return f"Predcition under the permited {cfg.threshold} threshold accuracy"
        return None

def do_action(movement,car_mode_on, moving ):

    if movement != "eyes_centered":
        ########################
        # Eyes movements
        ########################
        if movement == "eyes_left":
            keyboard.direct_key("a")
        elif movement == "eyes_right":
            keyboard.direct_key("d")
        elif movement == "eyes_up":
            moving = keyboard.direct_key_move("w", 0.4, moving)
        elif movement == "eyes_closed":
            keyboard.direct_key_sleep("f", 0.2)
        elif movement == "blink_left":
            keyboard.click_left_mouse()
        elif movement == "blink_right":
            keyboard.direct_key_sleep("NUMPADENTER", 0.08)
            car_mode_on = not car_mode_on
            keyboard.direct_key_sleep("h", 0.08)
           # print("car_mode : ", car_mode_on)
            time.sleep(0.2)
        ########################
        # Head movements
        ########################
        elif movement == "head_left":
             ms.move(-cfg.x_, 0, absolute=False, duration=cfg.time_)
        elif movement == "head_right":
            ms.move(cfg.x_, 0, absolute=False, duration=cfg.time_)
        elif movement == "head_up":
            if car_mode_on:
             ms.move(0, -cfg.x_, absolute=False, duration=cfg.time_)
        elif movement == "head_down":
            ms.move(0, cfg.x_, absolute=False, duration=cfg.time_)
        if movement:
            print("movement", movement)
    else:
        keyboard.direct_key_released("a")
        keyboard.direct_key_released("d")
        keyboard.direct_key_released("s")
        keyboard.direct_key_released("f")
        ms.unhook_all()

    return car_mode_on, moving

def detect_pose(frame, detector_pose):

    frame = detector_pose.findPose(frame)
    lmList, bboxInfo = detector_pose.findPosition(frame, bboxWithHands=False)
    if bboxInfo:
        center = bboxInfo["center"]
        cv2.circle(frame, center, 5, (255, 0, 255), cv2.FILLED)
    return frame

def window_display_info(frame, path, pTime, car_mode_on, detect_face):

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (180, 30, 30), 2)
    cv2.putText(frame, path, (15, 470), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
   # if detect_face:
    #    cv2.putText(frame, "Car Mode: " + str(car_mode_on), (500, 40), cv2.FONT_HERSHEY_PLAIN, 1, (180, 30, 30), 1)

    return pTime

def draw_eyes(img, faces):

    # left eye landmarks
    leye = [130, 30, 29, 28, 27, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25, 33, 246, 161, 160, 159, 158, 157, 173, 133,
            155, 154, 153, 145, 144, 163, 7, 33]
    # right eye landmarks
    reye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 463, 414, 286, 258, 259,
            257, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341]

    try:
        for landmark in leye:
            cv2.circle(img, ((faces[0][landmark])[0], (faces[0][landmark])[1]), 2, (255, 50, 0), cv2.FILLED)
        for landmark in reye:
            cv2.circle(img, ((faces[0][landmark])[0], (faces[0][landmark])[1]), 2, (255, 50, 0), cv2.FILLED)
    except Exception as err:
        print(err)

def eyes_detection(frame, faces):


    try:

        x_min, y_min = faces[0][130]
        x_max, y_max = faces[0][359]

        image_eyes = frame[y_min - 20:y_max + 20, x_min - 10:x_max + 10]
        return image_eyes
    except IndexError as err:
        print(err)

def increase_brightness(img, value=30):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img