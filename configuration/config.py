from tensorflow.keras.models import load_model

classes = {  # 11 classes/movements that the neural network will predict given an eyes pic
    0: 'blink_left',
    1: 'blink_right',
    2: 'eyes_centered',
    3: 'eyes_closed',
    4: 'eyes_left',
    5: 'eyes_right',
    6: 'eyes_up',
    7: 'head_down',
    8: 'head_left',
    9: 'head_right',
    10: 'head_up',
}
threshold = 0.80

x_ = 25
time_ = 0.1

n = [12, 10, 8]


buffer_size = 2
new_model = load_model('model/80epochs_11classes_10.h5')


path="C:/Users/Nour/Desktop/nourrr"