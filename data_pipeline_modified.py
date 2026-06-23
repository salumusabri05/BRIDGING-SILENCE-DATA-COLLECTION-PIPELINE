"""
BRIDGING SILENCE — DATA COLLECTION PIPELINE
Input:
data/
 ├── WORD_A/
 │    ├── sample_001.mp4
 │    └── ...
 └── WORD_B/

Outputs:
processed_data/
 ├── landmarks/
 │    └── WORD/*.npy
 └── preview/
      └── WORD/*.mp4

Output format:
(MAX_SEQ_LENGTH, 141)

141 =
5 pose landmarks (nose, shoulders, elbows)
+
21 left hand
+
21 right hand
(x,y,z)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.framework.formats import landmark_pb2

DATA_DIR="data"
OUT_DIR="processed_data"

# Updated to 120 frames
MAX_SEQ_LENGTH=120
FEATURES=141
FPS=30
WIDTH=640
HEIGHT=480

mp_holistic=mp.solutions.holistic
mp_draw=mp.solutions.drawing_utils

POSE=[0,11,12,13,14]

def normalize_frame(vec):
    vec=vec.copy()

    pose=vec[:15].reshape(5,3)
    lh=vec[15:78].reshape(21,3)
    rh=vec[78:].reshape(21,3)

    if np.any(pose):
        ls=pose[1]
        rs=pose[2]

        center=(ls+rs)/2

        dist=np.linalg.norm(ls-rs)
        if dist>1e-6:
            pose=(pose-center)/dist

            if np.any(lh):
                lh=(lh-center)/dist

            if np.any(rh):
                rh=(rh-center)/dist

    return np.concatenate([
        pose.flatten(),
        lh.flatten(),
        rh.flatten()
    ])


def extract(results):

    pose=[]

    if results.pose_landmarks:
        for idx in POSE:
            lm=results.pose_landmarks.landmark[idx]
            pose.extend([lm.x,lm.y,lm.z])
    else:
        pose=[0]*15

    lh=[]

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh.extend([lm.x,lm.y,lm.z])
    else:
        lh=[0]*63

    rh=[]

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh.extend([lm.x,lm.y,lm.z])
    else:
        rh=[0]*63

    return normalize_frame(
        np.array(
            pose+lh+rh,
            dtype=np.float32
        )
    )


def pad(seq):

    if len(seq)<MAX_SEQ_LENGTH:
        seq=np.vstack([
            seq,
            np.zeros(
                (
                    MAX_SEQ_LENGTH-len(seq),
                    FEATURES
                )
            )
        ])

    return seq[:MAX_SEQ_LENGTH]


def reconstruct(data,outfile):

    writer=cv2.VideoWriter(
        str(outfile),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH,HEIGHT)
    )

    for i,frame in enumerate(data):

        # Updated to white background
        canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)

        if np.all(frame==0):
            cv2.putText(
                canvas,
                "PAD",
                (280,220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,0), # Updated to black text
                2
            )
            writer.write(canvas)
            continue

        pose=frame[:15].reshape(5,3)
        lh=frame[15:78].reshape(21,3)
        rh=frame[78:].reshape(21,3)

        def draw(points,color):

            for p in points:
                if p[0] == 0 and p[1] == 0 and p[2] == 0:
                    continue
                x=int((p[0]*0.25+0.5)*WIDTH)
                y=int((p[1]*0.25+0.5)*HEIGHT)

                cv2.circle(
                    canvas,
                    (x,y),
                    3,
                    color,
                    -1
                )

        draw(pose,(0,0,255))
        draw(lh,(0,255,0))
        draw(rh,(255,255,0))

        cv2.putText(
            canvas,
            f"Frame {i}",
            (10,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            .7,
            (0,0,0), # Updated to black text
            2
        )

        writer.write(canvas)

    writer.release()


def run():

    land_root=Path(OUT_DIR)/"landmarks"
    prev_root=Path(OUT_DIR)/"preview"

    land_root.mkdir(
        parents=True,
        exist_ok=True
    )

    prev_root.mkdir(
        parents=True,
        exist_ok=True
    )

    with mp_holistic.Holistic(
        min_detection_confidence=.5,
        min_tracking_confidence=.5
    ) as holo:

        for word in sorted(os.listdir(DATA_DIR)):

            folder=Path(DATA_DIR)/word

            if not folder.is_dir():
                continue

            print(f"\nProcessing {word}")

            (land_root/word).mkdir(
                exist_ok=True
            )

            (prev_root/word).mkdir(
                exist_ok=True
            )

            for video in folder.iterdir():

                if video.suffix.lower() not in [
                    ".mp4",
                    ".avi",
                    ".mov"
                ]:
                    continue

                cap=cv2.VideoCapture(
                    str(video)
                )

                seq=[]

                while True:

                    ok,frame=cap.read()

                    if not ok:
                        break

                    rgb=cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    res=holo.process(rgb)

                    seq.append(
                        extract(res)
                    )

                cap.release()

                seq=np.array(seq)

                original=len(seq)

                seq=pad(seq)

                npy=(
                    land_root
                    /word
                    /(video.stem+".npy")
                )

                np.save(
                    npy,
                    seq
                )

                reconstruct(
                    seq,
                    prev_root
                    /word
                    /(video.stem+".mp4")
                )

                print(
                    f"✓ {video.name}"
                    f" | frames={original}"
                    f" -> {seq.shape}"
                )

    print("\nDONE")


if __name__=="__main__":
    run()
