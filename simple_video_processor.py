#!/usr/bin/env python3
"""
Simple Video Processor with Trained YOLOv8 Model
"""

import cv2
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse
from ultralytics import YOLO
import numpy as np
from src.tracker import Tracker
from src.speed import SpeedEstimator

def process_video_simple(video_path: str, output_dir: str = "output"):
    """
    Simple video processing with trained YOLOv8 model
    """
    
    print(f"Processing video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    try:
        model = YOLO('models/traffic_violations_best.pt')
        print("Loaded trained traffic violation model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using pretrained YOLOv8n model instead")
        model = YOLO('yolov8n.pt')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer for annotated output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_dir, f"detected_{Path(video_path).stem}.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Processing variables
    frame_count = 0
    detection_count = 0
    violations_log = []
    start_time = datetime.now()

    # Initialize tracker
    tracker = Tracker()

    # Initialize speed estimator
    speed_estimator = SpeedEstimator(fps=fps)

    # Track logged violations to avoid duplicates per vehicle
    logged_violations = set()  # (track_id, violation_type) tuples

    # Class names (from your data.yaml)
    class_names = {
        0: 'helmet',
        1: 'no_helmet',
        2: 'mobile_usage',
        3: 'traffic_violation',
        4: 'overspeed',
        5: 'number_plate'
    }

    # Violation classes only (exclude compliance classes like helmet, number_plate)
    violation_classes = ['no_helmet', 'mobile_usage', 'traffic_violation', 'overspeed']
    
    print("Starting video processing...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = model.predict(frame, conf=0.5, verbose=False)

            # Convert detections to tracker format
            detections_for_tracker = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = class_names.get(class_id, f"class_{class_id}")

                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_name': class_name
                        }
                        detections_for_tracker.append(detection)

            # Update tracker
            tracked_objects = tracker.update(detections_for_tracker, frame)

            # Process detections for annotation and violation logging
            annotated_frame = frame.copy()
            frame_detections = len(detections_for_tracker)

            # Draw all detections
            for detection in detections_for_tracker:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']

                # Draw bounding box
                color = get_color_for_class(class_name)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw tracked objects and log violations once per vehicle
            for track in tracked_objects:
                track_id = track['track_id']
                bbox = track['bbox']
                class_name = track['class_name']

                # Draw track ID on tracked objects
                x1, y1, x2, y2 = bbox
                cv2.putText(annotated_frame, f"ID:{track_id}", (int(x1), int(y1) - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Log violation only if it's a violation class and not already logged for this track
                if class_name in violation_classes:
                    violation_key = (track_id, class_name)
                    if violation_key not in logged_violations:
                        logged_violations.add(violation_key)
                        violation = {
                            'id': len(violations_log) + 1,
                            'type': class_name,
                            'confidence': float(track['confidence']),
                            'bbox': bbox,
                            'timestamp': datetime.now().isoformat(),
                            'frame_number': frame_count,
                            'track_id': track_id,
                            'plate_number': None
                        }
                        violations_log.append(violation)
            
            if frame_detections > 0:
                detection_count += frame_detections
                print(f"Frame {frame_count}: {frame_detections} detections")
            
            # Save annotated frame
            out.write(annotated_frame)
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = datetime.now() - start_time
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {detection_count} total detections")
    
    except KeyboardInterrupt:
        print("\nProcessing stopped by user")
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        
        # Save violations to JSON file
        violations_file = os.path.join(output_dir, "violations_log.json")
        with open(violations_file, 'w') as f:
            json.dump(violations_log, f, indent=2)
        
        # Final statistics
        elapsed_time = datetime.now() - start_time
        print(f"\nProcessing completed!")
        print(f"Statistics:")
        print(f"   - Frames processed: {frame_count}")
        print(f"   - Total detections: {detection_count}")
        print(f"   - Processing time: {elapsed_time}")
        print(f"   - Average FPS: {frame_count / elapsed_time.total_seconds():.2f}")
        print(f"Annotated video saved: {output_video_path}")
        print(f"Violations log saved: {violations_file}")

def get_color_for_class(class_name):
    """Get color for different violation types"""
    color_map = {
        'helmet': (0, 255, 0),        # Green
        'no_helmet': (0, 0, 255),     # Red
        'mobile_usage': (0, 165, 255), # Orange
        'traffic_violation': (255, 0, 0), # Blue
        'overspeed': (0, 255, 255),   # Yellow
        'number_plate': (255, 255, 0) # Cyan
    }
    return color_map.get(class_name, (255, 255, 255))  # White default

def main():
    parser = argparse.ArgumentParser(description='Process video for traffic violations')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return
    
    # Process video
    process_video_simple(
        video_path=args.video_path,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
