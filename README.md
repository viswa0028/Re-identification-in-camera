# Re-identification

A real-time player identification and tracking system using computer vision and deep learning. This system detects players in video footage and assigns consistent IDs based on facial/appearance features.

## Features

- **Real-time Player Detection**: Uses YOLO object detection to identify players in video frames
- **Feature-based Identification**: Employs ResNet-50 for extracting deep features from player appearances
- **Similarity Matching**: Uses FAISS (Facebook AI Similarity Search) for efficient similarity search
- **Persistent ID Assignment**: Maintains consistent player IDs across video frames
- **Initialization Period**: Learns new players during the first 5 seconds of video
- **Adaptive Thresholding**: Configurable similarity threshold for ID matching

## Requirements

### Dependencies

```bash
pip install numpy torch opencv-python faiss-cpu ultralytics torchvision
```

### System Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended for faster processing)
- Sufficient RAM for video processing and feature storage

## File Structure

```
project/
├── main.py                    # Main application script
├── Best (1).pt               # Custom YOLO model weights
├── Assignment Materials 720p.mp4  # Input video file
└── README.md                 # This file
```

## Configuration

### Key Parameters

- **Similarity Threshold**: `0.87` - Minimum similarity score for player matching
- **Initialization Period**: `5 seconds` - Time window for learning new players
- **Minimum Bounding Box Area**: `1000 pixels` - Filters out small detections
- **Feature Dimension**: `2048` - ResNet-50 feature vector size

### Model Setup

1. **YOLO Model**: Place your trained YOLO weights file as `Best (1).pt`
2. **ResNet-50**: Automatically downloads pre-trained weights from torchvision
3. **Video Input**: Update the video path in the code: `Assignment Materials 720p.mp4`

## Usage

### Basic Usage

```bash
python main.py
```

### Controls

- **'q' key**: Quit the application
- **ESC**: Alternative quit method

### Workflow

1. **Initialization Phase** (First 5 seconds):
   - Detects and learns new players
   - Builds feature database
   - Assigns initial Player IDs

2. **Tracking Phase** (After 5 seconds):
   - Matches detected players against known features
   - Assigns consistent IDs to recognized players
   - Labels unknown players as "Unknown"

## How It Works

### 1. Object Detection
- Uses YOLO to detect player bounding boxes in each frame
- Filters detections based on minimum area threshold

### 2. Feature Extraction
- Crops detected regions of interest (ROI)
- Applies ResNet-50 preprocessing transformations
- Extracts 2048-dimensional feature vectors

### 3. Similarity Search
- Normalizes feature vectors for cosine similarity
- Uses FAISS IndexFlatIP for efficient similarity search
- Compares against stored player features

### 4. ID Assignment
- **Known Players**: Assigns existing ID if similarity > threshold
- **New Players**: Creates new ID during initialization period
- **Unknown Players**: Labels as "Unknown" after initialization

## Customization

### Adjusting Similarity Threshold

```python
Similarity_threshold = 0.87  # Increase for stricter matching
```

### Changing Initialization Period

```python
initialization_frames = int(fps * 5)  # 5 seconds, adjust as needed
```

### Modifying Minimum Detection Size

```python
min_area = 1000  # Minimum bounding box area in pixels
```

## Output

The system displays:
- **Bounding boxes** around detected players (green rectangles)
- **Player labels** above each bounding box
- **Real-time video** with tracking annotations

## Troubleshooting

### Common Issues

1. **YOLO Model Not Found**
   ```
   Error: [Errno 2] No such file or directory: 'Best (1).pt'
   ```
   - Ensure YOLO weights file is in the correct path

2. **Video File Not Found**
   ```
   Error: Video file not found
   ```
   - Check video file path and format

3. **CUDA Out of Memory**
   - Reduce video resolution or batch size
   - Use CPU-only mode by setting device to 'cpu'

4. **Poor Tracking Performance**
   - Adjust similarity threshold
   - Increase initialization period
   - Check lighting conditions in video

### Performance Optimization

- **GPU Acceleration**: Ensure CUDA is properly installed
- **Batch Processing**: Process multiple detections simultaneously
- **Memory Management**: Clear unused variables and tensors

## Technical Details

### Architecture

- **Detection**: YOLOv8 (or compatible version)
- **Feature Extraction**: ResNet-50 (pre-trained on ImageNet)
- **Similarity Search**: FAISS IndexFlatIP (Inner Product)
- **Preprocessing**: Standard ImageNet normalization

### Data Flow

1. Video Frame → YOLO Detection → ROI Extraction
2. ROI → ResNet-50 → Feature Vector
3. Feature Vector → FAISS Search → Similarity Score
4. Similarity Score → ID Assignment → Display

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues and pull requests to improve the system.

## Acknowledgments

- YOLO by Ultralytics
- ResNet by Microsoft Research
- FAISS by Facebook AI Research
