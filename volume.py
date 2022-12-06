import cv2
import mediapipe as mp
import math
from pynput.keyboard import Key, Controller

keyboard = Controller()

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if ret:
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    index_f_x = (int)(landmarks.landmark[8].x*640)
                    index_f_y = (int)(landmarks.landmark[8].y*480)

                    middle_f_x = (int)(landmarks.landmark[12].x*640)
                    middle_f_y = (int)(landmarks.landmark[12].y*480)

                    cv2.circle(frame, (index_f_x, index_f_y), 4, (0,0,255), 5)
                    cv2.circle(frame, (middle_f_x, middle_f_y), 4, (0,0,255), 5)

                    cv2.line(frame, (index_f_x, index_f_y), (middle_f_x, middle_f_y), (255, 0, 0), 4)

                    distance = math.dist((index_f_x, index_f_y), (middle_f_x, middle_f_y))

                    if distance > 35:
                        # increase volume 
                        keyboard.press(Key.media_volume_up)
                        keyboard.release(Key.media_volume_up)
                    else:
                        # decrease volume
                        keyboard.press(Key.media_volume_down)
                        keyboard.release(Key.media_volume_down)


            
            cv2.imshow("window", cv2.flip(frame, 1))
        

        key = cv2.waitKey(10)

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()