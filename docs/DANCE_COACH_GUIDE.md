# Dance Coach Guide

Complete guide for using the Dance Coach demo to record, practice, and improve your dance movements.

## Overview

The Dance Coach uses computer vision and Dynamic Time Warping (DTW) to compare your dance performance against a reference recording. It provides real-time feedback and scoring to help you improve your movements.

## Features

- **Reference Recording**: Record any dance sequence (3-10 seconds recommended)
- **Real-time Comparison**: Get instant feedback while practicing
- **DTW Algorithm**: Handles different speeds - dance faster or slower
- **8-Joint Tracking**: Monitors major body joints
- **Scoring System**: 0-100 score with per-joint breakdown
- **Save/Load**: Store reference sequences for later practice

## Quick Start

### 1. Launch the Demo

```bash
cd /Volumes/MindDockSSD/projects/opensource/motion-tracker
source venv/bin/activate
python demos/dance_coach_demo.py
```

### 2. Record Reference Dance

1. Stand in front of the camera (full body visible)
2. Press `r` to start recording
3. Perform your dance move (3-10 seconds)
4. Press `r` again to stop recording
5. You'll see: "Reference ready: XX frames"

**Tips:**
- Keep the entire body in frame
- Move smoothly and clearly
- 5-8 seconds is ideal
- Practice the move first before recording

### 3. Practice Mode

1. Press `p` to enter practice mode
2. Perform the same dance
3. Watch real-time feedback:
   - **[OK]** Green = Good (< 15° difference)
   - **[~]** Orange = Okay (15-30° difference)
   - **[!]** Red = Needs work (> 30° difference)
4. Press `p` to stop and see your score

### 4. View Results

After stopping practice, you'll see:
```
==========================================
DANCE COMPARISON RESULTS
==========================================
Overall Score: 87.3/100

Joint Scores:
  Left Elbow          :  92.5/100
  Right Elbow         :  89.1/100
  Left Shoulder       :  85.2/100
  Right Shoulder      :  88.7/100
  Left Knee           :  90.3/100
  Right Knee          :  86.9/100
  Left Hip            :  84.5/100
  Right Hip           :  81.2/100
==========================================
```

## Controls

| Key | Action |
|-----|--------|
| `r` | Start/Stop recording reference |
| `p` | Start/Stop practice mode |
| `c` | Clear current reference |
| `s` | Save reference to file (`dance_reference.pkl`) |
| `l` | Load reference from file |
| `q` | Quit |

## Understanding the Score

### Overall Score (0-100)

- **90-100**: Excellent! Near-perfect match
- **80-89**: Great! Very close to reference
- **70-79**: Good! Minor adjustments needed
- **60-69**: Okay - practice key joints
- **< 60**: Needs work - focus on problem areas

### Per-Joint Scores

Each joint is scored individually. Low-scoring joints show where you need improvement:

- **Elbows**: Arm position and movement
- **Shoulders**: Upper body posture
- **Knees**: Leg bending and position
- **Hips**: Core and lower body movement

## Tips for Better Scores

### Recording the Reference

1. **Lighting**: Ensure good, even lighting
2. **Background**: Plain background works best
3. **Distance**: Stand 2-3 meters from camera
4. **Framing**: Keep full body in frame throughout
5. **Clarity**: Make movements clear and distinct
6. **Duration**: 5-8 seconds is optimal

### Practicing

1. **Start Slow**: Practice at slower speed first
2. **Focus on Problem Joints**: Check which joints score lowest
3. **Mirror Mode**: Video is flipped - use it like a mirror
4. **Consistent Position**: Start in the same spot as reference
5. **Multiple Attempts**: Try several times to improve

### Improving Scores

If a specific joint scores low:

- **Elbow/Wrist**: Check arm angles and timing
- **Shoulder**: Work on upper body posture
- **Hip**: Focus on core positioning
- **Knee**: Practice leg bending and extension

## Advanced Features

### Save Reference for Later

Save your reference dance:
```bash
Press 's' while reference is loaded
```

This creates `dance_reference.pkl` in the current directory.

### Load Saved Reference

Load a previously saved reference:
```bash
Press 'l' to load from dance_reference.pkl
```

### Multiple References

Save different references with different names:

```python
# In the code, you can modify the filename
filepath = "dance_hip_hop.pkl"
filepath = "dance_ballet.pkl"
filepath = "dance_jazz.pkl"
```

## Technical Details

### Dynamic Time Warping (DTW)

DTW allows the system to compare sequences of different lengths and speeds. This means:

- You can dance faster or slower than the reference
- Pauses and timing variations are handled
- The algorithm finds the best alignment

### Tracked Joints

The system monitors 8 key joints:
1. Left Elbow
2. Right Elbow
3. Left Shoulder
4. Right Shoulder
5. Left Knee
6. Right Knee
7. Left Hip
8. Right Hip

### Real-time Feedback

During practice, each joint is compared frame-by-frame:

- **< 15° difference**: [OK] Green
- **15-30° difference**: [~] Orange
- **> 30° difference**: [!] Red

### Scoring Algorithm

1. Calculate DTW distance for each joint
2. Normalize by sequence length
3. Convert to 0-100 score (higher is better)
4. Average all joint scores for overall score

## Troubleshooting

### "Sequences too short"

**Problem**: Recording less than 10 frames

**Solution**:
- Record for at least 2-3 seconds
- Ensure pose is detected (green skeleton visible)
- Check camera connection

### Low Scores Despite Good Performance

**Possible causes**:
- Reference recorded from different angle
- Lighting changes between recording and practice
- Standing in different position
- Camera moved

**Solution**:
- Re-record reference in current conditions
- Ensure consistent position
- Check camera placement

### Joints Not Detected

**Symptoms**: Missing feedback for some joints

**Solution**:
- Ensure full body visible in frame
- Improve lighting
- Move closer/further from camera
- Wear contrasting clothing

### High Latency

**Symptoms**: Delayed feedback

**Solution**:
- Use lite model: `MediaPipeBackend(model_complexity=0)`
- Reduce camera resolution
- Close other applications
- Check system resources

## Example Workflow

### Learning a New Dance Move

1. **Watch Tutorial**: Learn the move first
2. **Record Reference**:
   - Slow, clear performance
   - Press `r`, perform, press `r`
3. **Practice Session**:
   - Press `p` to start
   - Follow the move, watch feedback
   - Press `p` to see score
4. **Iterate**:
   - Focus on low-scoring joints
   - Try again (press `p`)
   - Track improvement

### Teaching a Dance Routine

1. **Instructor Records**: Teacher performs reference
2. **Save Reference**: Press `s` to save
3. **Students Practice**:
   - Load reference (press `l`)
   - Each student practices (press `p`)
   - Compare scores

### Choreography Practice

1. **Break Into Sections**: Record 5-second segments
2. **Practice Each**: Master one section at a time
3. **Combine**: String sections together
4. **Polish**: Refine low-scoring parts

## Performance Tips

### For Best Real-time Performance

```python
# Use lite model in code
estimator = MediaPipeBackend(model_complexity=0)
```

### For Best Accuracy

```python
# Use heavy model
estimator = MediaPipeBackend(model_complexity=2)
```

### Optimal Settings

- **Resolution**: 1280x720 (default)
- **Model**: Full (complexity=1) - balanced
- **Recording**: 5-8 seconds
- **Lighting**: Bright, even

## API Reference

### DanceSequence Class

```python
from demos.dance_coach_demo import DanceSequence

# Create sequence
sequence = DanceSequence("My Dance")

# Add frames
sequence.add_frame(pose_result, angles, timestamp)

# Get angle sequence
elbow_angles = sequence.get_angle_sequence('left_elbow')

# Save/load
sequence.save("my_dance.pkl")
loaded = DanceSequence.load("my_dance.pkl")
```

### DTWMatcher Class

```python
from demos.dance_coach_demo import DTWMatcher

matcher = DTWMatcher()

# Calculate distance
distance = matcher.dtw_distance(seq1, seq2)

# Get score
score = matcher.normalize_score(distance, avg_length)
```

### DanceCoach Class

```python
from demos.dance_coach_demo import DanceCoach

coach = DanceCoach()

# Recording
coach.start_recording_reference()
coach.add_reference_frame(pose_result, timestamp)
coach.stop_recording_reference()

# Practicing
coach.start_practice()
coach.add_practice_frame(pose_result, timestamp)
feedback = coach.get_real_time_feedback(pose_result)
coach.stop_practice()

# Compare
results = coach.compare_sequences()
```

## Future Enhancements

Planned features:

- [ ] Multiple reference comparison
- [ ] Video export with annotations
- [ ] Progress tracking over time
- [ ] Side-by-side comparison view
- [ ] Detailed timing analysis
- [ ] Beat/rhythm detection
- [ ] Multi-person comparison
- [ ] Mobile app version

## Support

For issues or questions:
- GitHub Issues: https://github.com/MindDock/motion-tracker/issues
- Documentation: See `README.md` and other docs

## Credits

- DTW implementation based on classic algorithm
- MediaPipe Pose for pose estimation
- OpenCV for video processing
