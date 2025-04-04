from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StringType, StructType, StructField
import cv2
from ultralytics import YOLO
import numpy as np
import base64  
import happybase  # HBase client
from hdfs import InsecureClient
import time
import os

# Initialize Spark session
spark = SparkSession.builder.appName("KafkaImageConsumer") \
    .config("spark.sql.streaming.stateStore.stateFormatVersion", "2") \
    .getOrCreate()

# Reduce logging
spark.sparkContext.setLogLevel("ERROR")

# Initialize YOLO model
model = YOLO("/home/akshat/YOLO models/arrowcone2.engine", task='detect')

# HDFS & HBase Configuration
HDFS_URL = "http://localhost:9870"
hdfs_client = InsecureClient(HDFS_URL, user="akshat")
HDFS_IMAGE_DIR = "/user/akshat/images/"
HBASE_HOST = "localhost"
TABLE_NAME = "object_detections"
connection = happybase.Connection(HBASE_HOST)
table = connection.table(TABLE_NAME)

# Define schema for JSON data
schema = StructType([
    StructField("image", StringType(), True)
])

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "webcam-stream") \
    .option("startingOffsets", "latest") \
    .load()

df = df.selectExpr("CAST(value AS STRING)").withColumn("data", from_json(col("value"), schema)).select(col("data.image").alias("image"))

# **Use foreachBatch instead of foreach**
def process_batch(batch_df, batch_id):
    """ Process each micro-batch in a loop """
    rows = batch_df.collect()
    for row in rows:
        if row.image is None:
            continue

        try:
            # Decode Base64
            img_data = base64.b64decode(row.image)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # YOLO Detection
            pred = model.predict(img, device='cuda', half=True, verbose=True)[0]
            
            for value in pred.boxes:
                box = list(value.xyxy[0].cpu())
                prob = float(value.conf.cpu())
                clss = float(list(value.cls.cpu())[0])

                # Draw bounding box
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 0, 0), 1)
                cv2.putText(img, f'Prob: {prob:.3f} class {clss:.2f}', (int(box[0]) - 20, int(box[1]) - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 0, 0), 2)

                x = int((box[0] + box[2]) / 2)
                y = int((box[1] + box[3]) / 2)

                # Save Image to HDFS
                hdfs_path = f"{HDFS_IMAGE_DIR}frame_{timestamp.replace(':', '_')}.jpg"
                local_path = f"/tmp/frame_{timestamp.replace(':', '_')}.jpg"
                cv2.imwrite(local_path, img)

                # Upload to HDFS
                with open(local_path, 'rb') as f:
                    hdfs_client.write(hdfs_path, f, overwrite=True)
                
                os.remove(local_path)

                # Store metadata in HBase
                row_key = f"{timestamp}_{clss}"
                table.put(row_key.encode(), {
                    b"info:timestamp": timestamp.encode(),
                    b"info:class": str(clss).encode(),
                    b"info:x": str(x).encode(),
                    b"info:y": str(y).encode(),
                    b"info:image_path": hdfs_path.encode()
                })

                print(f"Stored in HBase: class={clss}, x={x}, y={y}, path={hdfs_path}")

            # Display image (optional)
            cv2.imshow("Received Image", img)
            cv2.waitKey(1)

        except Exception as e:
            print(f"Error processing image: {e}")

# Apply function to micro-batches
query = df.writeStream \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()
cv2.destroyAllWindows()
