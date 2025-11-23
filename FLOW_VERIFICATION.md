# Pipeline Flow Verification ✅

## Complete Data Processing & Training Flow

---

## 📋 FLOW SUMMARY

```
1. Setup & Configuration
   ↓
2. Data Loading (load_files)
   ↓
3. Dataset Creation (SELDDataset)
   ├── Load Audio → Mel Spectrograms
   ├── Load Metadata → Labels
   ├── Concatenate All Data
   └── Create Windowed Segments
   ↓
4. Model Architecture
   ├── CSPDarkNet53 Backbone
   └── SMRSELDWithCSPDarkNet
   ↓
5. Loss Function (SMRSELDLoss)
   ├── Class MSE Loss
   ├── AIUR Loss
   └── Converging Localization Loss
   ↓
6. Training Loop (train_model)
   ├── Create DataLoaders
   ├── Initialize Model & Optimizer
   ├── Training Epochs
   └── Save Best Model
   ↓
7. Testing Loop (test_model)
   ├── Load Best Model
   ├── Run Inference
   └── Create Visualizations
```

---

## 🔍 DETAILED FLOW VERIFICATION

### **PHASE 1: Setup & Configuration** ✅

**File Lines:** 1-200

**Components:**
- Logging setup (`setup_logging`) - Line 38
- Device detection (`get_device`) - Line 83
- Configuration class (`Config`) - Line 105
- Global variables initialized

**Verification:**
- ✅ Logger configured before any operations
- ✅ CUDA device detected and CuDNN enabled
- ✅ All paths and hyperparameters defined

---

### **PHASE 2: Data Loading** ✅

**File Lines:** 205-272

**Function:** `load_files(config)`

**What happens:**
1. Scans directories for audio files (`.wav`)
2. Matches metadata files (`.csv`) by basename
3. Combines SONY and TAU datasets
4. Returns 4 lists: train_audio, train_meta, test_audio, test_meta

**Verification:**
- ✅ Checks if USE_FULL_DATASET flag
- ✅ Validates metadata files exist for each audio file
- ✅ Returns matched pairs: audio ↔ metadata

**Output:**
```python
train_audio_files: ['path/to/audio1.wav', 'path/to/audio2.wav', ...]
train_meta_files:  ['path/to/audio1.csv', 'path/to/audio2.csv', ...]
test_audio_files:  ['path/to/test1.wav', ...]
test_meta_files:   ['path/to/test1.csv', ...]
```

---

### **PHASE 3: Audio Processing** ✅

**File Lines:** 274-316

#### 3A. Load Audio (`load_audio`) - Line 274
**Input:** Audio file path  
**Process:**
- Uses `torchaudio.load()` to load multi-channel audio
- Verifies 4 channels (FOA format)

**Output:** `waveform` tensor (4, num_samples), `sample_rate`

#### 3B. Convert to Mel Spectrogram (`audio_to_mel_spectrogram`) - Line 284
**Input:** Waveform (4, samples)  
**Process:**
1. Creates MelSpectrogram transform (n_fft, hop_length, n_mels)
2. Processes each of 4 channels separately
3. Concatenates channels
4. Converts to dB scale

**Output:** `mel_spec` tensor (4, 64, time_frames)

**Verification:**
- ✅ Preserves all 4 FOA channels
- ✅ Uses config parameters (n_fft=960, hop_length=480, n_mels=64)
- ✅ Output shape: [Channels=4, MelBins=64, TimeFrames]

---

### **PHASE 4: Label Generation** ✅

**File Lines:** 318-380

**Function:** `metadata_to_labels(metadata_path, audio_duration, ...)`

**What happens:**
1. **Calculate frames:** 20ms per frame → total_frames from audio duration
2. **Initialize tensor:** [T, I×J, M] with zeros
   - T = time frames
   - I×J = spatial grid cells (18×36 = 648)
   - M = num_classes (14)
3. **Read CSV metadata:** Parse [frame, class, source, azimuth, elevation]
4. **Map each event:**
   - Convert metadata frame (100ms) → multiple 20ms frames
   - Convert polar coords (azimuth, elevation) → grid cell (i, j)
   - Set class probability to 1.0 for active cells
5. **Set background:** All cells without events get background class (index 13)

**Verification:**
- ✅ Frame alignment: 100ms metadata → 5× 20ms frames
- ✅ Spatial mapping: polar_to_grid(phi, theta) → (i, j)
- ✅ One-hot encoding per cell
- ✅ Background class for empty cells

**Output:** `labels` tensor (time_frames, 648, 14)

---

### **PHASE 5: Dataset Creation** ✅

**File Lines:** 383-549

**Class:** `SELDDataset(Dataset)`

#### Initialization Flow:

**Step 5.1: `__init__`** (Line 384-429)
- Takes audio_files and metadata_files lists
- Calculates grid dimensions: 18×36 = 648 cells
- Calculates window parameters:
  - Window: 5s = 250 spectrogram frames
  - Hop: 1s = 50 spectrogram frames
- Calls `_load_and_concatenate_all()`
- Calls `_create_windows()`

**Step 5.2: `_load_and_concatenate_all()`** (Line 431-490)
For each audio/metadata pair:
1. Load audio → waveform
2. Convert to mel spectrogram → (4, 64, T_i)
3. Generate labels → (T_i, 648, 14)
4. Align time dimensions (crop to min if mismatch)
5. Append to lists

Then concatenate:
- All spectrograms → (4, 64, T_total)
- All labels → (T_total, 648, 14)

**Step 5.3: `_create_windows()`** (Line 492-535)
Segments concatenated data:
1. Start at frame 0
2. Extract window_length_frames (250)
3. Move by hop_length_frames (50)
4. Repeat until end of data
5. Pad last window if needed (with zeros + background class)
6. Transpose spectrogram: [C, F, T] → [T, C, F]

**Step 5.4: `__getitem__`** (Line 537-549)
Returns single window:
- Spectrogram: (250, 4, 64) - [Time, Channels, MelBins]
- Labels: (250, 648, 14) - [Time, GridCells, Classes]

**Verification:**
- ✅ All audio files processed before windowing
- ✅ Temporal alignment maintained
- ✅ Windowing creates overlapping segments (4s overlap)
- ✅ Consistent output shape for all windows
- ✅ Padding handled correctly for final window

**Output:** Dataset with N windows, each:
- Input: (250, 4, 64)
- Target: (250, 648, 14)

---

### **PHASE 6: Model Architecture** ✅

**File Lines:** 552-722

#### Building Blocks:
1. **Conv** (Line 552) - Conv2d + BatchNorm + SiLU
2. **Bottleneck** (Line 564) - Residual block
3. **C3** (Line 576) - CSP bottleneck
4. **SPPF** (Line 592) - Spatial pyramid pooling

#### Backbone: **CSPDarkNet53** (Line 609-657)
- Stem: 3×3 conv
- Stage 1: Conv + C3 (3 blocks) → 128 channels
- Stage 2: Conv + C3 (6 blocks) → 256 channels
- Stage 3: Conv + C3 (9 blocks) → 512 channels
- Stage 4: Conv + C3 (3 blocks) + SPPF → 1024 channels

Returns multi-scale features: [P2, P3, P4, P5]

#### Main Model: **SMRSELDWithCSPDarkNet** (Line 659-722)

**Forward Pass Flow:**
```
Input: [B, T, 4, 64]
   ↓
Reshape: [B×T, 4, 64, 1]  (treat time independently)
   ↓
Backbone → [P2, P3, P4, P5]
   ↓
Fusion: Use P3, P4, P5
   ├─ 1×1 conv to 256 channels each
   ├─ Upsample to same size
   └─ Concatenate → 768 channels
   ↓
Conv Fuse: 768 → 512 → 256 channels
   ↓
Grid Pool: [B×T, 256, H, W] → [B×T, 256, 648, 1]
   (Adaptive pooling to grid_cells rows)
   ↓
Classifier: 256 → 128 → 14 (per cell)
   ↓
Reshape: [B×T, 648, 14] → [B, T, 648, 14]
   ↓
Softmax: along class dimension
   ↓
Output: [B, T, 648, 14]
```

**Verification:**
- ✅ Input: [Batch, Time, Channels, Frequency]
- ✅ Processes each time frame independently
- ✅ Multi-scale feature fusion
- ✅ Spatial pooling to exact grid dimensions (648)
- ✅ Per-cell classification (14 classes)
- ✅ Output: [Batch, Time, GridCells, Classes]

---

### **PHASE 7: Loss Function** ✅

**File Lines:** 724-880

**Class:** `SMRSELDLoss(nn.Module)`

#### Three Components:

**7A. Class MSE Loss** (Line 741)
- Computes MSE between predictions and ground truth
- Applied across all cells and classes
- Weighted: handles class imbalance

**7B. AIUR Loss** (Line 747)
- Area Intersection Union Ratio (IoU-based)
- For each frame:
  1. Find cells with events in predictions (non-background)
  2. Find cells with events in ground truth
  3. Compute intersection (both have events)
  4. Compute union (either has events)
  5. IoU = intersection / union
  6. Loss = 1 - IoU
- Average across all frames and batch

**7C. Converging Localization Loss** (Line 781)
- Helps model focus on event regions
- Uses circular padding for neighborhood context
- Creates attention map from surroundings
- Encourages predictions to converge to event areas

**Forward Method** (Line 836)
```python
total_loss = w_class × class_mse + w_aiur × aiur + w_cl × cl
breakdown = {'class_mse': ..., 'aiur': ..., 'cl': ...}
return total_loss, breakdown
```

**Verification:**
- ✅ Three complementary loss components
- ✅ Per-frame computation prevents global averaging issues
- ✅ Handles class imbalance (background vs events)
- ✅ Returns detailed breakdown for monitoring

---

### **PHASE 8: Training Loop** ✅

**File Lines:** 882-1277

**Function:** `train_model(train_audio_files, train_meta_files, test_audio_files, test_meta_files, ...)`

#### Training Flow:

**Step 8.1: Dataset Creation** (Line 904-914)
```python
train_dataset = SELDDataset(train_audio_files, train_meta_files)
test_dataset = SELDDataset(test_audio_files, test_meta_files)
```

**Step 8.2: DataLoader Creation** (Line 916-929)
- Batch size: 8
- Shuffle: True (train), False (test)
- **CUDA optimizations:**
  - num_workers=4 (parallel loading)
  - pin_memory=True (fast GPU transfer)

**Step 8.3: Model & Optimizer** (Line 936-960)
```python
model = SMRSELDWithCSPDarkNet(...).to(device)
criterion = SMRSELDLoss(...)
optimizer = Adam(lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(...)
```

**Step 8.4: Training Loop** (Line 988-1096)
For each epoch:
1. **Training Phase:**
   ```python
   model.train()
   for spectrograms, labels in train_loader:
       spectrograms = spectrograms.to(device, non_blocking=True)  # CUDA opt
       labels = labels.to(device, non_blocking=True)
       
       predictions = model(spectrograms)
       loss, breakdown = criterion(predictions, labels)
       
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   ```

2. **Validation Phase:**
   ```python
   model.eval()
   with torch.no_grad():
       for spectrograms, labels in test_loader:
           spectrograms = spectrograms.to(device, non_blocking=True)
           labels = labels.to(device, non_blocking=True)
           
           predictions = model(spectrograms)
           loss, breakdown = criterion(predictions, labels)
   ```

3. **Checkpoint Saving:**
   - Save best model (based on test loss)
   - Save periodic checkpoints (every 5 epochs)
   - Keep only last 3 checkpoints

4. **Learning Rate Scheduling:**
   - ReduceLROnPlateau monitors test loss
   - Reduces LR by 0.5 if no improvement for 5 epochs

5. **Early Stopping:**
   - Stops if no improvement for 10 epochs

6. **Memory Management:**
   - Clear CUDA cache every 5 epochs

**Step 8.5: Save Results** (Line 1150-1178)
- Load best model weights
- Save training history
- Plot loss curves (saved as PNG)

**Verification:**
- ✅ Dataset created from raw audio files
- ✅ DataLoader with CUDA optimizations
- ✅ Model initialized and moved to GPU
- ✅ Loss function with all three components
- ✅ Training and validation loops separate
- ✅ Gradient computation in training only
- ✅ Checkpointing and early stopping
- ✅ LR scheduling based on validation loss
- ✅ Memory cleanup for long runs

---

### **PHASE 9: Testing Loop** ✅

**File Lines:** 1279-1517

**Function:** `test_model(test_audio_files, test_meta_files, model_path, ...)`

#### Testing Flow:

**Step 9.1: Load Test Dataset** (Line 1295-1315)
```python
test_dataset = SELDDataset(test_audio_files, test_meta_files)
test_loader = DataLoader(test_dataset, ...)
```

**Step 9.2: Load Best Model** (Line 1317-1342)
```python
checkpoint = torch.load(model_path)
model = SMRSELDWithCSPDarkNet(...).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**Step 9.3: Inference** (Line 1351-1383)
```python
with torch.no_grad():
    for spectrograms, labels in test_loader:
        spectrograms = spectrograms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        predictions = model(spectrograms)
        loss, breakdown = criterion(predictions, labels)
        
        # Store predictions and labels (on CPU)
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
```

**Step 9.4: Calculate Metrics** (Line 1385-1408)
- Overall accuracy
- Non-background accuracy
- Count frames with active events

**Step 9.5: Create Visualizations** (Line 1410-1481)
1. Find frames with active events
2. Randomly select frames for visualization
3. For each selected frame:
   ```python
   visualize_grid_predictions(
       ground_truth=frame_labels,
       predictions=frame_predictions,
       save_path=viz_dir / f"test_viz_{idx}.png"
   )
   ```
4. Creates comparison plots (ground truth vs predictions vs difference)

**Step 9.6: Return Results** (Line 1491-1506)
```python
results = {
    'test_loss': avg_test_loss,
    'class_mse': ...,
    'aiur': ...,
    'cl': ...,
    'overall_accuracy': ...,
    'non_bg_accuracy': ...,
    'num_frames_with_events': ...,
    'visualizations': [...],
    'checkpoint_epoch': ...
}
```

**Verification:**
- ✅ Uses same dataset creation as training
- ✅ Loads best saved model
- ✅ No gradient computation (eval mode)
- ✅ CUDA optimizations (non_blocking)
- ✅ Comprehensive metrics
- ✅ Visual comparisons saved to disk
- ✅ Returns detailed results dictionary

---

### **PHASE 10: Main Entry Point** ✅

**File Lines:** 1519-1638

**Function:** `main()`

#### Execution Flow:

**Step 10.1: Parse Arguments** (Line 1520-1536)
```python
--epochs: Number of training epochs
--batch-size: Batch size
--lr: Learning rate
--small-model: Use smaller backbone
--full-dataset: Use all files vs single file
```

**Step 10.2: Initialize** (Line 1538-1558)
```python
logger, log_file = setup_logging()
config = Config()
DEVICE = get_device()
```

**Step 10.3: Load Data** (Line 1560-1566)
```python
train_audio_files, train_meta_files, test_audio_files, test_meta_files = load_files(config)
```

**Step 10.4: Train** (Line 1568-1579)
```python
model, history = train_model(
    train_audio_files, train_meta_files,
    test_audio_files, test_meta_files,
    ...
)
```

**Step 10.5: Test** (Line 1581-1595)
```python
test_results = test_model(
    test_audio_files, test_meta_files,
    model_path=config.CHECKPOINT_PATH / "best_model.pth",
    ...
)
```

**Step 10.6: Report Results** (Line 1597-1616)
- Training summary
- Test performance
- Accuracy metrics
- Visualization count

**Verification:**
- ✅ CLI interface for flexibility
- ✅ Complete pipeline in single run
- ✅ Comprehensive logging
- ✅ Error handling with traceback
- ✅ Proper exit codes

---

## ✅ FINAL VERIFICATION CHECKLIST

### Data Flow:
- ✅ Raw audio files → Mel spectrograms
- ✅ Metadata CSVs → Label tensors
- ✅ Concatenation → Complete dataset
- ✅ Windowing → Fixed-size batches

### Temporal Alignment:
- ✅ Audio sample rate: 24000 Hz
- ✅ Spectrogram hop: 480 samples (20ms per frame)
- ✅ Metadata frame: 100ms (5 spectrogram frames)
- ✅ Window: 250 frames (5 seconds)
- ✅ Hop: 50 frames (1 second overlap)

### Spatial Mapping:
- ✅ Grid: 18×36 = 648 cells (10° resolution)
- ✅ Polar to Cartesian: (azimuth, elevation) → (i, j)
- ✅ One-hot encoding per cell (14 classes)

### Model Architecture:
- ✅ Input: [Batch, Time=250, Channels=4, Freq=64]
- ✅ Output: [Batch, Time=250, Cells=648, Classes=14]
- ✅ Per-frame processing
- ✅ Multi-scale features
- ✅ Softmax output (probabilities)

### Loss Function:
- ✅ Three components (MSE + AIUR + CL)
- ✅ Weighted combination
- ✅ Per-frame computation
- ✅ Handles class imbalance

### Training:
- ✅ Batch processing
- ✅ Forward + Backward + Optimize
- ✅ Validation after each epoch
- ✅ Checkpointing
- ✅ Early stopping
- ✅ LR scheduling

### Testing:
- ✅ Inference only (no gradients)
- ✅ Metric computation
- ✅ Visualization generation
- ✅ Results reporting

### CUDA Optimizations:
- ✅ CuDNN benchmarking
- ✅ Pin memory
- ✅ Multi-worker loading
- ✅ Non-blocking transfers
- ✅ Memory cleanup

---

## 🎯 SUMMARY

**The complete pipeline flow is CORRECT:**

1. ✅ Data is loaded (audio + metadata pairs)
2. ✅ Dataset converts audio to mel spectrograms
3. ✅ Labels are created from metadata (spatial + temporal)
4. ✅ Dataset creates segmented windows with overlap
5. ✅ Model architecture is properly defined
6. ✅ Loss function has all three components
7. ✅ Training loop is complete with optimization
8. ✅ Testing loop evaluates and visualizes
9. ✅ All CUDA optimizations are applied
10. ✅ Main function orchestrates everything

**No issues found. Pipeline is ready for HPC deployment! 🚀**
