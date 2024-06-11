import torch
import torchvision
import cv2
import time
import threading
from datetime import datetime, timedelta
from roboflow import Roboflow

if torch.cuda.is_available():
    print('you are using gpu to process the video camera')
else:
    print('no gpu is found in this python environment. using cpu to process')

class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()
        self.cond = threading.Condition()
        self.is_running = False
        self.frame = None
        self.pellets_num = 0
        self.callback = None
        super().__init__(name=name)
        self.start()

    def start(self):
        self.is_running = True
        super().start()

    def stop(self, timeout=None):
        self.is_running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.is_running:
            (rv, img) = self.capture.read()
            assert rv
            counter += 1
            with self.cond:
                self.frame = img if rv else None
                self.pellets_num = counter
                self.cond.notify_all()
            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.pellets_num + 1
                if seqnumber < 1:
                    seqnumber = 1
                rv = self.cond.wait_for(lambda: self.pellets_num >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.pellets_num, self.frame)
            return (self.pellets_num, self.frame)


# Replace the class_labels definition
class_labels = {1: 'Pellets'}


def main():
    # Connect to Roboflow and load the model
    rf = Roboflow(api_key="t0yZkYja0BgquHGKivMK")
    project = rf.workspace("I@FAD_NYP").project("YOUR_PROJECT_NAME")
    version = project.version("")
    model = version.model

    cap = cv2.VideoCapture('rtsp://admin:Citi123!@192.168.1.64:554/Streaming/Channels/101')
    cap.set(cv2.CAP_PROP_FPS, 30)

    fresh = FreshestFrame(cap)

    object_count = {1: 0}

    feeding = False
    feeding_timer = None

    first_feeding_time = 15
    feeding_minutes_min = 29

    second_feeding_time = 15
    second_feeding_time_min = 31

    showing_timer = None
    desired_time = None

    formatted_desired_time = None
    current_datetime = datetime.now()

    while True:
        temp_object_count = {1: 0}

        current_time = datetime.now().time()
        if (current_time.hour == first_feeding_time or current_time.hour == second_feeding_time) and (current_time.minute == feeding_minutes_min or current_time.minute == second_feeding_time_min) and current_time.second == 0:
            feeding = True
            feeding_timer = None
            showing_timer = None

        cnt, frame = fresh.read(seqnumber=object_count[1] + 1)
        if frame is None:
            break

        # Prepare image for Roboflow
        img_encoded = cv2.imencode('.jpg', frame)[1].tostring()

        # Perform inference with Roboflow model
        predictions = model.predict(img_encoded).json()['predictions']

        for prediction in predictions:
            label = prediction['class']
            if label in class_labels.values():
                box = prediction['bbox']
                score = prediction['confidence']

                if score > 0.975:
                    cv2.rectangle(frame, (int(box['x']), int(box['y'])), (int(box['x'] + box['width']), int(box['y'] + box['height'])), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label}: {score:.2f}', (int(box['x']), int(box['y']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    temp_object_count[1] += 1

                    if label == 'Pellets' and feeding_timer is None and feeding:
                        feeding_timer = time.time()

        for label, count in temp_object_count.items():
            object_count[label] = count

        if feeding_timer is not None and feeding:
            elapsed_time = (time.time() - feeding_timer)

            print(f'elapsed time: {elapsed_time:.3f}')

            if elapsed_time > 60 and sum(object_count.values()) > 3:
                feeding = False
                feeding_timer = None
                showing_timer = time.time()

            elif object_count[1] == 0:
                feeding_timer = None

        for label, count in object_count.items():
            text = f'{class_labels[label]} Count: {count}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_position = (frame.shape[1] - text_size[0] - 10, 30 * label)
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        text_position_feed = (frame.shape[1] - text_size[0] - 10, 30 * (max(object_count.keys()) + 1))
        round_position = (frame.shape[1] - 200 - 50, 30 * (1 + 1))

        if feeding:
            cv2.putText(frame, "Feeding...", text_position_feed, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            if showing_timer is not None:
                i = time.time() - showing_timer
                if i > 3:
                    showing_timer = None
                    print('running')
                else:
                    cv2.putText(frame, "Stop Feeding", text_position_feed, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if current_time.hour <= first_feeding_time and current_time.minute <= feeding_minutes_min:
                    desired_time = current_datetime.replace(hour=first_feeding_time, minute=feeding_minutes_min, second=0, microsecond=0)
                    formatted_desired_time = desired_time.strftime("%I:%M %p")
                    cv2.putText(frame, "next round: " + formatted_desired_time, round_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif current_time.hour <= second_feeding_time and current_time.minute <= second_feeding_time_min:
                    desired_time = current_datetime.replace(hour=second_feeding_time, minute=second_feeding_time_min, second=0, microsecond=0)
                    formatted_desired_time = desired_time.strftime("%I:%M %p")
                    cv2.putText(frame, "next round: " + formatted_desired_time, round_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    next_day = current_datetime + timedelta(days=1)
                    desired_time = next_day.replace(hour=first_feeding_time, minute=feeding_minutes_min, second=0, microsecond=0)
                    formatted_desired_time = desired_time.strftime("%I:%M %p")
                    cv2.putText(frame, "next round: " + formatted_desired_time, round_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.namedWindow('Pellets Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Pellets Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fresh.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
