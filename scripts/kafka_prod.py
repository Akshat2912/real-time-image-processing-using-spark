import cv2
import json
import base64
from confluent_kafka import Producer

# Kafka Configuration
KAFKA_TOPIC = "webcam-stream"
KAFKA_BROKER = "localhost:9092"

producer = Producer({'bootstrap.servers': KAFKA_BROKER})

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_data = encode_frame(frame)
    producer.produce(KAFKA_TOPIC, key="frame", value=json.dumps({"image": frame_data}))
    producer.flush()

cap.release()
