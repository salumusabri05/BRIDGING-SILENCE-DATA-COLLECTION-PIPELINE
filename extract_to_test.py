"""
extract_to_test.py
──────────────────
Scans any folder of videos (flat or class-subfolders) and extracts
MediaPipe Holistic landmarks into .npy files saved under  test/

Usage:
    python extract_to_test.py <path_to_video_folder>

Examples:
    python extract_to_test.py data/JINA
    python extract_to_test.py C:/my_videos
    python extract_to_test.py data            ← scans all word sub-folders

Output structure:
    test/
     └── <word_or_flat>/
           └── <video_stem>.npy   shape: (120, 141)
"""

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# ── Config (matches data.ipynb exactly) ─────────────────────────────────────
MAX_SEQ_LENGTH = 120
FEATURES       = 141
POSE_IDX       = [0, 11, 12, 13, 14]   # nose, l-shoulder, r-shoulder, l-elbow, r-elbow
VIDEO_EXT      = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
OUTPUT_ROOT    = Path("test")

mp_holistic = mp.solutions.holistic


# ── Helper functions (identical to data.ipynb) ───────────────────────────────
def normalize_frame(vec):
    vec  = vec.copy()
    pose = vec[:15].reshape(5, 3)
    lh   = vec[15:78].reshape(21, 3)
    rh   = vec[78:].reshape(21, 3)

    if np.any(pose):
        ls     = pose[1]
        rs     = pose[2]
        center = (ls + rs) / 2
        dist   = np.linalg.norm(ls - rs)
        if dist > 1e-6:
            pose = (pose - center) / dist
            if np.any(lh):
                lh = (lh - center) / dist
            if np.any(rh):
                rh = (rh - center) / dist

    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])


def extract(results):
    pose = []
    if results.pose_landmarks:
        for idx in POSE_IDX:
            lm = results.pose_landmarks.landmark[idx]
            pose.extend([lm.x, lm.y, lm.z])
    else:
        pose = [0] * 15

    lh = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh.extend([lm.x, lm.y, lm.z])
    else:
        lh = [0] * 63

    rh = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh.extend([lm.x, lm.y, lm.z])
    else:
        rh = [0] * 63

    return normalize_frame(np.array(pose + lh + rh, dtype=np.float32))


def pad(seq):
    if len(seq) < MAX_SEQ_LENGTH:
        seq = np.vstack([seq, np.zeros((MAX_SEQ_LENGTH - len(seq), FEATURES))])
    return seq[:MAX_SEQ_LENGTH]


# ── Core processing ──────────────────────────────────────────────────────────
def process_video(video_path: Path, out_dir: Path, holistic):
    cap = cv2.VideoCapture(str(video_path))
    seq = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        seq.append(extract(res))

    cap.release()

    if not seq:
        print(f"  ⚠  No frames read from {video_path.name} — skipped")
        return

    seq      = pad(np.array(seq))
    out_path = out_dir / (video_path.stem + ".npy")
    np.save(out_path, seq)
    print(f"  ✓  {video_path.name}  →  {out_path}  shape={seq.shape}")


def collect_videos(root: Path):
    """
    Returns a list of (video_path, label) tuples.
    - If root contains sub-folders  → each sub-folder name is the label.
    - If root contains videos directly → label is the root folder name.
    """
    pairs = []
    subdirs = [d for d in root.iterdir() if d.is_dir()]

    if subdirs:
        for sub in sorted(subdirs):
            for v in sorted(sub.iterdir()):
                if v.suffix.lower() in VIDEO_EXT:
                    pairs.append((v, sub.name))
    else:
        for v in sorted(root.iterdir()):
            if v.suffix.lower() in VIDEO_EXT:
                pairs.append((v, root.name))

    return pairs


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Bridging Silence — Landmark Extractor → test/")
    print("=" * 55)
    print()

    # Ask the user to type the path
    raw = input("Enter path to video folder: ").strip().strip('"').strip("'")

    if not raw:
        print("ERROR: No path entered. Exiting.")
        sys.exit(1)

    input_path = Path(raw)
    if not input_path.exists():
        print(f"ERROR: Path does not exist:\n  {input_path}")
        sys.exit(1)

    videos = collect_videos(input_path)
    if not videos:
        print(f"ERROR: No videos (.mp4 .avi .mov .mkv .webm) found in:\n  {input_path}")
        sys.exit(1)

    print(f"\nFound {len(videos)} video(s).")
    print(f"Output  →  {OUTPUT_ROOT.resolve()}\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holo:
        for video_path, label in videos:
            out_dir = OUTPUT_ROOT / label
            out_dir.mkdir(parents=True, exist_ok=True)
            process_video(video_path, out_dir, holo)

    print(f"\nDONE — results saved to '{OUTPUT_ROOT.resolve()}'")


if __name__ == "__main__":
    main()
