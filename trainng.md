# Bridging Silence — TSL Word Recognition Training Pipeline (Without Kalman Filtering)

## Objective

Train a **word-level Tanzanian Sign Language (TSL) recognition model** using body and hand landmarks extracted from videos.

This first version intentionally avoids temporal filtering (Kalman) to validate the complete pipeline before adding complexity.

Pipeline:

```text
Video
↓
MediaPipe Landmark Extraction
↓
Sequence Processing
↓
Padding / Sampling
↓
Dataset Build
↓
CNN + BiLSTM Training
↓
Evaluation
```

---

# 1. Data Collection Standard

This stage determines model quality more than architecture.

## Recording Device

Recommended:

```text
Google Pixel
```

Settings:

```text
Resolution: 1920×1080
FPS: 30
Orientation: Landscape
HDR: OFF
Stabilization: OFF
Zoom: 1×
```

---

## Recording Environment

Maintain consistency.

### Camera Position

```text
Distance: 1.8–2.0 meters
Height: Chest level
```

Subject visible:

```text
✓ Nose
✓ Shoulders
✓ Elbows
✓ Both hands
```

---

## Lighting

Use:

```text
Front lighting
Consistent brightness
Plain background
```

Avoid:

```text
Backlight
Changing rooms
Dark shadows
```

---

## Signing Style

Record naturally.

Each clip:

```text
Start pose
↓
Perform sign
↓
Hold 0.3–0.5 sec
↓
Stop
```

Target duration:

```text
2–3 seconds
```

Expected:

```text
60–90 frames
```

---

# 2. Dataset Structure

Raw dataset:

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
```

Processed output:

```text
processed_landmarks/
├── AHADI/
│   ├── sample_001.npy
│
├── AJIRA/
```

---

# 3. Landmark Extraction

Extract:

## Pose (5 landmarks)

```text
0  → Nose
11 → Left Shoulder
12 → Right Shoulder
13 → Left Elbow
14 → Right Elbow
```

Features:

```text
5 × 3 = 15
```

---

## Left Hand

```text
21 landmarks × 3
```

Features:

```text
63
```

---

## Right Hand

```text
21 landmarks × 3
```

Features:

```text
63
```

---

Total:

```text
141 features
```

Output shape:

```text
(frames,141)
```

Example:

```text
(72,141)
```

---

# 4. Sequence Processing

No Kalman filtering.

No smoothing.

Preserve original movement.

---

## Missing Landmark Handling

Do NOT insert zeros immediately.

Preferred:

```text
missing frame
↓
reuse previous frame
```

Example:

```text
frame 42 missing
↓

use frame 41
```

Avoid:

```text
[0,0,0]
```

because it creates artificial motion.

---

# 5. Frame Standardization

Model requires fixed length.

Use:

```python
MAX_SEQ_LENGTH = 90
```

Reason:

```text
Most signs ≈ 2 seconds
30 FPS
≈ 60–90 frames
```

Rules:

---

### If shorter

Example:

```text
72 → 90
```

Pad:

```text
0.0
```

---

### If longer

Example:

```text
110 → 90
```

Use temporal sampling.

Never cut.

Example:

```python
indices = np.linspace(
0,
len(seq)-1,
90
).astype(int)

seq = seq[indices]
```

Final shape:

```text
(90,141)
```

---

# 6. Dataset Split

Recommended:

```text
70% Train
15% Validation
15% Test
```

Split by:

```text
video
```

Not by frame.

---

# 7. Model Architecture (Baseline)

Goal:

Learn:

```text
hand shape
+
movement
+
timing
```

Architecture:

```text
Input
↓
Conv1D
↓
BatchNorm
↓
Conv1D
↓
BiLSTM
↓
BiLSTM
↓
Dense
↓
Softmax
```

Input:

```text
(90,141)
```

Output:

```text
Number of words
```

---

# 8. Training Configuration

Optimizer:

```text
Adam
```

Learning Rate:

```text
0.001
```

Loss:

```text
SparseCategoricalCrossentropy
```

Batch:

```text
8
```

Epochs:

```text
100
```

Callbacks:

```text
EarlyStopping
ReduceLROnPlateau
```

---

# 9. Evaluation

Measure:

```text
Accuracy
Confusion Matrix
Classification Report
```

Inspect:

```text
Which words collapse
Which words confuse
```

---

# 10. Success Criteria

Stage 1:

```text
2 words
Train >90%
Val >80%
```

Stage 2:

```text
10 words
Train >80%
Val >70%
```

Stage 3:

```text
50+ words
Train >75%
Val >65%
```

---

# 11. Future Improvements (Not Yet)

After baseline works:

```text
Kalman Filtering
Temporal Augmentation
Transformers
CTC
Attention
Multi-person robustness
```

Do not add them before baseline succeeds.

---

# Final Rule

Do not optimize architecture before verifying:

```text
Video
→ Landmarks
→ Dataset
→ Training
→ Prediction
```

A clean pipeline beats a complex model.
