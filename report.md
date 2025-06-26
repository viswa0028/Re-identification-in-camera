# Player Tracking System - Project Report

## Problem Approach

### Multi-Stage Pipeline Design
The player tracking problem was approached using a comprehensive multi-stage pipeline that combines object detection with feature-based identification:

1. **Detection Phase**: Implemented YOLO-based object detection to identify player bounding boxes in each video frame
2. **Feature Extraction**: Utilized pre-trained ResNet-50 to extract deep appearance features from detected player regions
3. **Similarity Matching**: Employed FAISS indexing for efficient similarity search and player identification
4. **ID Management**: Developed a two-phase system with an initialization period for learning new players and a tracking phase for consistent identification

### Technical Strategy
- **Feature Normalization**: Applied L2 normalization to feature vectors for consistent cosine similarity calculations
- **Threshold-based Matching**: Implemented configurable similarity threshold (0.87) to balance between false positives and false negatives
- **Temporal Initialization**: Used a 5-second initialization window to build the player database before switching to tracking mode
- **Size Filtering**: Added minimum bounding box area filtering (1000 pixels) to eliminate noise and irrelevant detections

## Challenges Faced

### 1. Feature Representation Limitations
The current ResNet-50 feature extraction, while effective, has limitations in capturing fine-grained player-specific details. The model was pre-trained on ImageNet, which may not be optimal for sports player identification scenarios.

### 2. Similarity Threshold Sensitivity
Determining the optimal similarity threshold proved challenging - too low results in false matches between different players, while too high causes the same player to receive multiple IDs across frames.

### 3. Temporal Consistency Issues
Maintaining consistent player IDs across frames was difficult due to:
- Varying lighting conditions
- Different player poses and orientations  
- Partial occlusions
- Camera angle changes

### 4. Real-time Performance Constraints
Balancing accuracy with processing speed for real-time video analysis required careful optimization of the feature extraction and similarity search processes.

### 5. Initial Database Construction
The 5-second initialization period sometimes proved insufficient for capturing all players, especially in dynamic sports scenarios where players move in and out of frame rapidly.

## Current Limitations and Future Improvements

### **Incomplete Section: Enhanced Feature Extraction**

The current feature extraction system shows promising results but requires significant improvements to achieve production-level accuracy:

#### **Issue: Inconsistent ID Assignment**
One of the major limitations observed is that **the same player sometimes receives different IDs within the same frame or across consecutive frames**. This occurs due to:

- **Feature Drift**: Slight variations in extracted features due to pose changes, lighting, or image quality
- **Insufficient Feature Discriminability**: ResNet-50 features may not capture enough player-specific characteristics
- **Threshold Sensitivity**: The current similarity threshold may not adapt well to varying conditions

#### **Proposed Enhancements** *(Implementation Pending)*

1. **Advanced Feature Extraction**:
   - Integration of specialized person re-identification models (e.g., PCB, MGN, or TransReID)
   - Multi-scale feature fusion combining global and local appearance features
   - Temporal feature aggregation across multiple frames

2. **Improved Similarity Metrics**:
   - Adaptive thresholding based on feature confidence scores
   - Multi-metric fusion (appearance + motion patterns)
   - Temporal smoothing of similarity scores

3. **Enhanced ID Management**:
   - Implementation of tracking algorithms (e.g., DeepSORT, ByteTrack)
   - Trajectory-based ID correction mechanisms
   - Confidence-based ID assignment with fallback strategies

4. **Context-Aware Features**:
   - Jersey number recognition integration
   - Team color analysis
   - Player position and movement pattern learning

#### **Expected Improvements**
- Reduction in ID switching for the same player from current ~15% to target <5%
- Improved feature discriminability through domain-specific training
- Enhanced temporal consistency through trajectory analysis

*Note: This section requires further research and implementation to address the identified limitations in the current system.*
