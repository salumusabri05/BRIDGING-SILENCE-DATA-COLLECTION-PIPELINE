# Bridging Silence — TSL Data Collection & Preprocessing Pipeline

This pipeline converts raw Tanzanian Sign Language (TSL) videos into a training-ready landmark dataset for word recognition.

The system processes videos, extracts body and hand landmarks using MediaPipe Holistic, normalizes the coordinates, pads sequences into a fixed length, and generates reconstructed landmark videos for visual inspection.

---

## Pipeline Overview

```text
Raw Videos
↓
MediaPipe Holistic
↓
Landmark Extraction
↓
Coordinate Normalization
↓
Sequence Padding
↓
Training Dataset (.npy)
↓
Landmark Reconstruction Videos
```

---

## Input Structure

Place videos inside a `data/` directory using class-based folders.

```text
data/
├── AHADI/
│   ├── sample_001.mp4
│   ├── sample_002.mp4
│
├── AJIRA/
│   ├── sample_001.mp4
│
├── AMINI/
│   ├── sample_001.mp4
```

Each folder name becomes the class label.

---

## Extracted Features

Each frame produces **141 features**.

```text
Pose (5 landmarks)
+
Left Hand (21 landmarks)
+
Right Hand (21 landmarks)
```

### Pose Landmarks

| Landmark | MediaPipe Index |
|----------|----------------|
| Nose | 0 |
| Left Shoulder | 11 |
| Right Shoulder | 12 |
| Left Elbow | 13 |
| Right Elbow | 14 |

Each landmark stores:

```text
x
y
z
```

Total:

```text
(5 × 3) + (21 × 3) + (21 × 3)
=
141 features
```

---

## Sequence Processing

Videos naturally contain different numbers of frames.

Examples:

```text
AHADI → 120 frames
AMINI → 140 frames
KISIMA → 161 frames
```

To create uniform training input:

- Short videos are padded with zeros
- Long videos are truncated

Final output shape:

```python
(120, 141)
```

Where:

```text
120 → Frames
141 → Features per frame
```

---

## Landmark Normalization

Coordinates are normalized using shoulder position.

Steps:

1. Compute body center using left and right shoulders
2. Translate all landmarks relative to body center
3. Scale coordinates using shoulder distance

This improves robustness to:

- Camera distance
- Signer position
- Body movement

---

## Output Structure

After processing:

```text
processed_data/
├── landmarks/
│   ├── AHADI/
│   │   ├── sample_001.npy
│   │   ├── sample_002.npy
│
├── preview/
│   ├── AHADI/
│   │   ├── sample_001.mp4
│   │   ├── sample_002.mp4
```

---

## Output Files

### Landmark Dataset

Training-ready NumPy arrays.

Example:

```python
sample_001.npy
```

Shape:

```python
(120, 141)
```

---

### Preview Videos

Reconstructed landmark videos.

Purpose:

- Inspect extraction quality
- Detect missing hands
- Detect unstable landmarks
- Validate normalization

---

## Training Compatibility

Designed for:

```text
Conv1D
↓
BiLSTM
↓
Dense
↓
Word Classification
```

Example model input:

```python
input_shape=(120,141)
```

---

## Recording Recommendations

### Device

Recommended:

```text
Google Pixel
1080p
30 FPS
Landscape
```

---

### Framing

Keep visible:

```text
✓ Head
✓ Shoulders
✓ Elbows
✓ Hands
```

---

### Distance

```text
1.7–2.0 meters
```

---

### Lighting

Use:

```text
Front lighting
Plain background
```

Avoid:

```text
Backlight
Dark environments
```

---

## Quality Checklist

Before training:

- [ ] Every class contains videos
- [ ] Landmarks reconstructed correctly
- [ ] No missing hands
- [ ] No cropped elbows
- [ ] Consistent recording setup
- [ ] Output shape is `(120,141)`

---

## Goal

Produce clean, consistent, training-ready TSL landmark sequences for robust word recognition.