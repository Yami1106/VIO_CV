# S-MSCKF & Learned Stereo Visual-Inertial Odometry

## Phase 1: S-MSCKF
Python implementation of S-MSCKF for visual-inertial odometry, evaluated on the EuRoC MAV dataset.

### Requirements
- Python 3.6+
- numpy
- scipy
- opencv-python (`cv2`)
- pandas
- matplotlib
- ffmpeg (for video generation)
- [pangolin](https://github.com/uoip/pangolin) (optional, for real-time 3D trajectory visualization)

Install Python dependencies:
```bash
pip install numpy scipy opencv-python pandas matplotlib
```

### Running the VIO Pipeline

#### With visualization (requires pangolin + PyOpenGL)
```bash
python vio.py --view --path /path/to/MH_01_easy
```

#### Without visualization
```bash
python vio.py --path /path/to/MH_01_easy
```

#### Saving the log for evaluation
Pipe the terminal output to a log file while still viewing it on screen:
```bash
python vio.py --view --path /path/to/MH_01_easy 2>&1 | tee phase1_run.log
```

### Evaluating Results
Once you have a log file from the VIO run, use `phase1_outputs.py` to compute metrics and generate plots:
```bash
python phase1_outputs.py --log phase1_run.log --dataset /path/to/MH_01_easy --outdir phase1_outputs
```

#### Arguments
| Argument    | Description                              | Default           |
|-------------|------------------------------------------|-------------------|
| `--log`     | Path to the saved VIO log file           |                   |
| `--dataset` | Path to the EuRoC MH_01_easy directory   |                   |
| `--outdir`  | Directory to save evaluation outputs     | `phase1_outputs`  |

#### Outputs
| File                             | Description                                              |
|----------------------------------|----------------------------------------------------------|
| `estimate_raw_tum.txt`           | Raw estimated trajectory in TUM format                   |
| `groundtruth_matched_tum.txt`    | Time-matched ground truth in TUM format                  |
| `estimate_aligned_tum.txt`       | SE(3)-aligned estimated trajectory in TUM format         |
| `phase1_trajectory_overlay.png`  | XY and XZ trajectory comparison plot (estimate vs GT)    |
| `Output.mp4`                     | Animated trajectory visualization video                  |
| `phase1_metrics.txt`             | Computed error metrics                                   |

#### Metrics
The script computes and prints the following metrics after SE(3) Umeyama alignment:
- **RMSE ATE** — Root Mean Square Error of Absolute Trajectory Error
- **MAE** — Mean Absolute Error
- **Median ATE** — Median Absolute Trajectory Error
- **Max ATE** — Maximum Absolute Trajectory Error
- **Final Drift** — Euclidean distance between final estimated and ground truth positions

---

## Phase 2: Learned Stereo Visual-Inertial Odometry

A deep learning approach to VIO using stereo image pairs rendered from Blender scenes. A multi-scale CNN extracts visual features from 4-channel stereo inputs (left_t, right_t, left_t+1, right_t+1), a bidirectional LSTM with attention encodes IMU data, and gated fusion combines both modalities. Pose graph optimization corrects accumulated drift as a post-processing step.

### Requirements
- Python 3.10+
- PyTorch (with CUDA)
- Blender 4.x+
- scipy, matplotlib, numpy

```bash
pip install torch torchvision scipy matplotlib numpy
```

### Step 1: Generate Trajectories
```bash
cd Phase2/data_generation_stereo
python generate_all_scenes_stereo.py
```
Generates 10 camera trajectories (10,000 poses each at 100Hz) across circle, oval, diamond, figure8, star, clover, mouse, halfmoon, sid, and picasso shapes.

### Step 2: Render Stereo Images in Blender

Renders left and right camera images for each pose using Cycles with GPU. Baseline of 0.25m at 6m camera height gives ~13px disparity at 320×320.

data.blend (scenes 1–5):
```bash
blender -b /path/to/data.blend -P render_stereo.py -- --scene_name scene_001_circle --stereo --baseline 0.25 --image_stride 10
blender -b /path/to/data.blend -P render_stereo.py -- --scene_name scene_002_oval --stereo --baseline 0.25 --image_stride 10
blender -b /path/to/data.blend -P render_stereo.py -- --scene_name scene_003_diamond --stereo --baseline 0.25 --image_stride 10
blender -b /path/to/data.blend -P render_stereo.py -- --scene_name scene_004_figure8 --stereo --baseline 0.25 --image_stride 10
blender -b /path/to/data.blend -P render_stereo.py -- --scene_name scene_005_star --stereo --baseline 0.25 --image_stride 10
```

texture2.blend (scenes 6–10):
```bash
blender -b /path/to/texture2.blend -P render_stereo.py -- --scene_name scene_006_clover --stereo --baseline 0.25 --image_stride 10
blender -b /path/to/texture2.blend -P render_stereo.py -- --scene_name scene_007_mouse --stereo --baseline 0.25 --image_stride 10
blender -b /path/to/texture2.blend -P render_stereo.py -- --scene_name scene_008_halfmoon --stereo --baseline 0.25 --image_stride 10
blender -b /path/to/texture2.blend -P render_stereo.py -- --scene_name scene_009_sid --stereo --baseline 0.25 --image_stride 10
blender -b /path/to/texture2.blend -P render_stereo.py -- --scene_name scene_010_picasso --stereo --baseline 0.25 --image_stride 10
```

### Step 3: Process Data
```bash
python process_all_scenes_stereo.py
```
Computes IMU measurements from poses and builds training sample pairs for all scenes.

### Step 4: Precompute Gyro Rotations
```bash
python precompute_rotation_stereo.py
```
Integrates gyroscope readings into relative quaternions and appends them to each scene's samples.

### Step 5: Split Dataset
```bash
python split_dataset_stereo.py
```

| Split | Scenes | Environment |
|-------|--------|-------------|
| Train (7) | circle, oval, diamond, figure8, clover, mouse, halfmoon | data.blend + texture2.blend |
| Val (1) | star | data.blend |
| Test (2) | sid, picasso | texture2.blend |

### Step 6: Train
```bash
cd Phase2/training_stereo

# Vision-only: predicts full 7D pose (3D translation + 4D quaternion)
python train_stereo.py --mode stereo_vision

# Visual-inertial: predicts 3D translation (rotation from gyro integration)
python train_stereo.py --mode stereo_visual_inertial
```
Models saved as `stereo_vision_best.pth` and `stereo_visual_inertial_best.pth`.

### Step 7: Evaluate (Dead Reckoning)
```bash
python test_stereo.py --all --splits train val test
```
Produces per-scene trajectory plots and ATE metrics using raw dead-reckoned predictions.

### Step 8: Evaluate (With Pose Graph Optimization)
```bash
python test_stereo_optimized.py --all --splits test
```
Applies post-processing optimization with relative, loop closure, and smoothness constraints. Outputs comparison plots showing dead-reckoned vs optimized vs ground truth trajectories.
