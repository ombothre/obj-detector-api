import cv2 as cv

import torch
from ultralytics import YOLO

import psycopg2 as psql

import json
from pydantic import BaseModel
from typing import List
import tempfile

from fastapi import FastAPI, File, UploadFile, Form


class ObjectDetector:

    def __init__(self, vid):

        self.labels = None
        self.vid = vid
        self.video_frames = []
        self.video_timestamps = []

        self.video_to_frames()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print("Using Device: ", self.device)

        self.model = YOLO("yolov8m.pt")
        self.model.fuse()

        self.CLASS_NAMES_DICT = self.model.model.names

    def video_to_frames(self):

        cap = cv.VideoCapture(self.vid)

        isTrue = 1
        frame_count = 0
        frame_rate = cap.get(cv.CAP_PROP_FPS)
        delay = 2
        delay_frames = int(frame_rate * delay)

        while isTrue:

            isTrue, frame = cap.read()

            if frame_count % delay_frames == 0:
                self.video_frames.append(frame)
                timestamp = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
                self.video_timestamps.append(timestamp)

            frame_count += 1

        cap.release()

    def predict(self):

        results = []
        outputs = self.model(self.video_frames, verbose=False)

        # print(outputs[2].boxes.conf.numpy())

        for i in range(len(self.video_timestamps)):

            output = outputs[i]
            time = int(self.video_timestamps[i])

            hours = time // 3600
            minutes = (time % 3600) // 60
            seconds = time % 60

            timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            confs = output.boxes.conf.numpy()
            ids = output.boxes.cls.numpy()

            objects = {}

            for j in range(len(ids)):

                if confs[j] > 0.6:

                    obj = self.CLASS_NAMES_DICT[ids[j]]

                    if obj in objects:
                        objects[obj] += 1

                    else:
                        objects[obj] = 1

            objects = json.dumps(objects)

            result = {
                "timestamp": timestamp,
                "objects": objects
            }

            results.append(result)

        return results


class Obj(BaseModel):
    object: str
    freq: int


# Fast API
app = FastAPI()


@app.get("/")
def home():
    return {"message": "Hello"}


@app.post("/process-video/")
async def operate(name: str = Form(...), video: UploadFile = File(...)):

    # {"name": "", "video": ""}

    vid = await video.read()

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(vid)
        vid_file_path = temp_file.name

    v_id = 0

    conn = psql.connect(
        dbname="xyz",
        user="postgres",
        password="root",
        host="localhost"
    )

    cur = conn.cursor()

    try:
        # Create videos table if not exists
        cur.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    Video_ID SERIAL PRIMARY KEY,
                    Name VARCHAR(100) UNIQUE
                )
            ''')

        # Create detections table if not exists
        cur.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    Detection_ID SERIAL PRIMARY KEY,
                    Video_ID INT REFERENCES videos(Video_ID),
                    Timestamp TIME WITHOUT TIME ZONE,
                    Objects JSONB
                )
            ''')

        # Commit the transaction to apply changes
        conn.commit()

    except psql.Error as e:
        conn.rollback()  # Rollback changes if an error occurs
        print("Error:", e)

    try:
        cur.execute("INSERT INTO videos (Name) VALUES (%s)", (name,))
        conn.commit()

        cur.execute("SELECT Video_ID FROM videos WHERE Name = %s", (name,))
        v_id = cur.fetchone()
        v_id = int(v_id[0])

    except psql.Error as e:
        conn.rollback()
        print(e)

    # processing
    detector = ObjectDetector(vid_file_path)  # insert image path/link
    results = detector.predict()

    try:
        for result in results:
            cur.execute("INSERT INTO public.detections (Video_ID,Timestamp,Objects) VALUES (%s, %s, %s::JSONB)",
                        (v_id, result['timestamp'], result['objects']))
        conn.commit()

    except psql.Error as e:
        conn.rollback()
        print(e)

    cur.close()
    conn.close()

    return {"name": name, "filename": video.filename}


@app.post("/predict/{name}/")
def process(name: str, objects: List[Obj]):

    # [
    #     {"object": "person", "freq": 4},
    #     {"object": "sports ball", "freq": 2}
    # ]

    conn = psql.connect(            # Enter your database details
        dbname="xyz",
        user="postgres",
        password="root",
        host="localhost"
    )

    cur = conn.cursor()

    objs = []
    mins = []

    for obj in objects:
        objs.append(obj.object)
        mins.append(obj.freq)

    cur.execute("SELECT Video_ID FROM videos WHERE Name = %s", (name,))
    v_id = cur.fetchone()
    v_id = int(v_id[0])

    # Constructing the query with parameterized inputs
    subconditions = []

    params = [name, v_id]  # Parameters for v_name and v_id
    for key, min_value in zip(objs, mins):
        condition = f"jsonb_exists(d.Objects, %s) AND (d.Objects->>%s)::int >= %s"
        subconditions.append(condition)
        params.extend([key, key, min_value])  # Extend parameters for key, key, and min_value

    conditions = " AND ".join(subconditions)
    query = f"""
                SELECT d.Timestamp
                FROM detections d
                JOIN videos v ON d.Video_ID = v.Video_ID
                WHERE v.Name = %s
                AND d.Video_ID = %s
                AND {conditions}
            """

    cur.execute(query, params)  # Pass all parameters as a single lis
    results = cur.fetchall()

    results = [i[0].strftime('%H:%M:%S') for i in results]

    cur.close()
    conn.close()

    return {"results": results}
