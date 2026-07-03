"""
Drone Detection, Path Tracking & Target Selection System
==========================================================

A comprehensive computer vision system for detecting drones using YOLO,
tracking their paths over time, and intelligently selecting targets
based on configurable criteria (proximity, speed, threat level).

Requirements:
    pip install ultralytics opencv-python numpy scipy

Usage:
    python drone_tracker.py --source 0          # Webcam
    python drone_tracker.py --source video.mp4  # Video file
    python drone_tracker.py --source rtsp://... # RTSP stream
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
from pathlib import Path

# Try to import ultralytics; provide fallback if not available
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

class TargetPriority(Enum):
    """Priority levels for target selection."""
    CRITICAL = 4   # Very close, fast approaching
    HIGH = 3       # Close or fast moving
    MEDIUM = 2     # Moderate distance/speed
    LOW = 1        # Far away, slow
    NONE = 0       # No threat


@dataclass
class DroneTrack:
    """
    Represents a tracked drone with its history and computed metrics.
    """
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "drone"
    
    # Tracking history
    trajectory: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Computed metrics
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    direction: float = 0.0  # degrees
    distance_from_center: float = 0.0
    
    # Target selection
    priority: TargetPriority = TargetPriority.NONE
    threat_score: float = 0.0
    is_selected: bool = False
    
    # Timing
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 0
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float, 
               frame_time: float, frame_shape: Tuple[int, int]):
        """Update track with new detection."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        self.bbox = bbox
        self.center = (center_x, center_y)
        self.confidence = confidence
        self.last_seen = frame_time
        self.frame_count += 1
        
        # Add to trajectory
        self.trajectory.append((center_x, center_y))
        self.timestamps.append(frame_time)
        
        # Calculate velocity and speed
        if len(self.trajectory) >= 2:
            self._calculate_velocity(frame_shape)
        
        # Calculate distance from frame center (normalized 0-1)
        h, w = frame_shape[:2]
        frame_cx, frame_cy = w / 2.0, h / 2.0
        self.distance_from_center = np.sqrt(
            ((center_x - frame_cx) / w) ** 2 + 
            ((center_y - frame_cy) / h) ** 2
        )
    
    def _calculate_velocity(self, frame_shape: Tuple[int, int]):
        """Calculate velocity vector and speed from trajectory."""
        if len(self.trajectory) < 2:
            return
        
        # Use last N points for smoothing
        n = min(5, len(self.trajectory))
        recent_points = list(self.trajectory)[-n:]
        recent_times = list(self.timestamps)[-n:]
        
        if len(recent_times) < 2 or recent_times[-1] == recent_times[0]:
            return
        
        # Calculate displacement
        dx = recent_points[-1][0] - recent_points[0][0]
        dy = recent_points[-1][1] - recent_points[0][1]
        dt = recent_times[-1] - recent_times[0]
        
        if dt > 0:
            self.velocity = (dx / dt, dy / dt)
            self.speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            self.direction = np.degrees(np.arctan2(-dy, dx))  # -dy for image coords
        
        # Normalize speed by frame diagonal
        h, w = frame_shape[:2]
        diagonal = np.sqrt(h**2 + w**2)
        self.speed = self.speed / diagonal if diagonal > 0 else 0
    
    def calculate_threat_score(self, frame_shape: Tuple[int, int]):
        """
        Calculate composite threat score based on:
        - Proximity to center (closer = higher threat)
        - Speed (faster = higher threat)
        - Approach velocity (moving toward center = higher threat)
        - Time tracked (newer = slightly higher - less predictable)
        """
        h, w = frame_shape[:2]
        frame_cx, frame_cy = w / 2.0, h / 2.0
        
        # Proximity factor (inverse of distance from center, normalized)
        proximity = max(0, 1.0 - self.distance_from_center * 2)
        
        # Speed factor (normalized, clamped)
        speed_factor = min(self.speed * 10, 1.0)
        
        # Approach factor: is drone moving toward center?
        approach = 0.0
        if len(self.trajectory) >= 2:
            prev = self.trajectory[-2]
            curr = self.trajectory[-1]
            prev_dist = np.sqrt((prev[0] - frame_cx)**2 + (prev[1] - frame_cy)**2)
            curr_dist = np.sqrt((curr[0] - frame_cx)**2 + (curr[1] - frame_cy)**2)
            if curr_dist < prev_dist:
                approach = min((prev_dist - curr_dist) / max(prev_dist, 1), 1.0)
        
        # Confidence factor
        confidence_factor = self.confidence
        
        # Composite score (weighted)
        self.threat_score = (
            proximity * 0.35 +
            speed_factor * 0.25 +
            approach * 0.25 +
            confidence_factor * 0.15
        )
        
        # Assign priority
        if self.threat_score > 0.7:
            self.priority = TargetPriority.CRITICAL
        elif self.threat_score > 0.5:
            self.priority = TargetPriority.HIGH
        elif self.threat_score > 0.3:
            self.priority = TargetPriority.MEDIUM
        elif self.threat_score > 0.1:
            self.priority = TargetPriority.LOW
        else:
            self.priority = TargetPriority.NONE
    
    def is_lost(self, current_time: float, timeout: float = 2.0) -> bool:
        """Check if track has timed out."""
        return (current_time - self.last_seen) > timeout
    
    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen


class DroneTracker:
    """
    Main tracking system that manages multiple drone tracks,
    handles detection-to-track association, and selects targets.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        max_lost: int = 30,
        track_buffer: int = 100,
        use_byte_track: bool = True,
        drone_classes: List[str] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.track_buffer = track_buffer
        self.use_byte_track = use_byte_track
        
        # Drone class names (customize based on your model)
        self.drone_classes = drone_classes or [
            "drone", "uav", "quadcopter", "hexacopter", 
            "aircraft", "airplane", "bird"  # fallback classes
        ]
        
        # Initialize YOLO model
        self.model = None
        self._init_model(model_path)
        
        # Tracking state
        self.tracks: Dict[int, DroneTrack] = {}
        self.next_track_id = 1
        self.frame_count = 0
        self.start_time = time.time()
        
        # Selected target
        self.selected_target_id: Optional[int] = None
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        
        # Visualization settings
        self.colors = {
            TargetPriority.CRITICAL: (0, 0, 255),    # Red
            TargetPriority.HIGH: (0, 165, 255),       # Orange
            TargetPriority.MEDIUM: (0, 255, 255),     # Yellow
            TargetPriority.LOW: (0, 255, 0),          # Green
            TargetPriority.NONE: (128, 128, 128),     # Gray
            "selected": (255, 0, 255),                # Magenta
            "trajectory": (0, 255, 0),               # Green
            "prediction": (255, 0, 0),               # Blue
        }
    
    def _init_model(self, model_path: str):
        """Initialize YOLO model."""
        if not ULTRALYTICS_AVAILABLE:
            print("ERROR: ultralytics not installed. Cannot load YOLO model.")
            return
        
        try:
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model: {model_path}")
            
            # Print model info
            if hasattr(self.model, 'names'):
                print(f"Model classes: {self.model.names}")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            print("Attempting to download default YOLOv8n...")
            try:
                self.model = YOLO("yolov8n.pt")
            except Exception as e2:
                print(f"Failed to load default model: {e2}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLO detection on frame and return drone detections.
        
        Returns list of dicts with keys: bbox, confidence, class_id, class_name
        """
        if self.model is None:
            return []
        
        detections = []
        
        try:
            # Run inference
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            
            for result in results:
                if result.boxes is None:
                    continue
                
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id] if hasattr(result, 'names') else str(class_id)
                    
                    # Filter for drone-like objects
                    is_drone = any(drone_term in class_name.lower() 
                                  for drone_term in self.drone_classes)
                    
                    if is_drone or confidence > 0.6:  # High conf = probably interesting
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        })
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detections
    
    def _calculate_iou(self, bbox1: Tuple[int, ...], bbox2: Tuple[int, ...]) -> float:
        """Calculate Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _associate_detections(self, detections: List[Dict], frame_shape: Tuple[int, int]):
        """
        Associate detections with existing tracks using IoU matching.
        Simple but effective for drone tracking.
        """
        current_time = time.time()
        matched_tracks = set()
        matched_dets = set()
        
        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            # Predict next position (simple linear prediction)
            if track.speed > 0 and len(track.trajectory) >= 2:
                dt = current_time - track.last_seen
                pred_x = track.center[0] + track.velocity[0] * dt
                pred_y = track.center[1] + track.velocity[1] * dt
                
                # Create predicted bbox
                w = track.bbox[2] - track.bbox[0]
                h = track.bbox[3] - track.bbox[1]
                pred_bbox = (
                    int(pred_x - w/2), int(pred_y - h/2),
                    int(pred_x + w/2), int(pred_y + h/2)
                )
            else:
                pred_bbox = track.bbox
            
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(pred_bbox, det['bbox'])
        
        # Greedy matching by highest IoU
        while True:
            if iou_matrix.size == 0 or np.max(iou_matrix) < 0.3:
                break
            
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            tid = track_ids[i]
            
            # Update track
            self.tracks[tid].update(
                detections[j]['bbox'],
                detections[j]['confidence'],
                current_time,
                frame_shape
            )
            matched_tracks.add(tid)
            matched_dets.add(j)
            
            # Remove from matrix
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1
        
        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                track = DroneTrack(
                    track_id=self.next_track_id,
                    bbox=det['bbox'],
                    center=((det['bbox'][0] + det['bbox'][2]) / 2,
                           (det['bbox'][1] + det['bbox'][3]) / 2),
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    class_name=det['class_name']
                )
                track.first_seen = current_time
                track.last_seen = current_time
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
        
        # Remove lost tracks
        lost_ids = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks and track.is_lost(current_time):
                lost_ids.append(tid)
        
        for tid in lost_ids:
            del self.tracks[tid]
    
    def select_target(self, frame_shape: Tuple[int, int], strategy: str = "threat"):
        """
        Select primary target using specified strategy.
        
        Strategies:
            - "threat": Highest threat score (default)
            - "closest": Closest to frame center
            - "newest": Most recently detected
            - "fastest": Highest speed
        """
        if not self.tracks:
            self.selected_target_id = None
            return
        
        # Mark all as not selected
        for track in self.tracks.values():
            track.is_selected = False
        
        # Calculate threat scores
        for track in self.tracks.values():
            track.calculate_threat_score(frame_shape)
        
        # Select based on strategy
        if strategy == "threat":
            best_track = max(self.tracks.values(), key=lambda t: t.threat_score)
        elif strategy == "closest":
            best_track = min(self.tracks.values(), key=lambda t: t.distance_from_center)
        elif strategy == "newest":
            best_track = max(self.tracks.values(), key=lambda t: t.last_seen)
        elif strategy == "fastest":
            best_track = max(self.tracks.values(), key=lambda t: t.speed)
        else:
            best_track = max(self.tracks.values(), key=lambda t: t.threat_score)
        
        best_track.is_selected = True
        self.selected_target_id = best_track.track_id
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect, track, select target, visualize.
        """
        self.frame_count += 1
        frame_time = time.time()
        h, w = frame.shape[:2]
        
        # Detection
        t0 = time.time()
        detections = self.detect(frame)
        det_time = time.time() - t0
        self.detection_times.append(det_time)
        
        # Update tracks
        if detections:
            self._associate_detections(detections, (h, w))
        
        # Select target
        self.select_target((h, w), strategy="threat")
        
        # Calculate FPS
        self.fps_history.append(1.0 / (time.time() - frame_time + 1e-6))
        
        # Visualization
        output = self._visualize(frame)
        
        return output
    
    def _visualize(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracks, trajectories, and target info on frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw frame center crosshair
        cx, cy = w // 2, h // 2
        cv2.line(overlay, (cx - 20, cy), (cx + 20, cy), (200, 200, 200), 1)
        cv2.line(overlay, (cx, cy - 20), (cx, cy + 20), (200, 200, 200), 1)
        cv2.circle(overlay, (cx, cy), 5, (200, 200, 200), 1)
        
        # Draw concentric zones
        for radius in [w//6, w//4, w//3]:
            cv2.circle(overlay, (cx, cy), radius, (50, 50, 50), 1)
        
        # Draw tracks
        for track in self.tracks.values():
            x1, y1, x2, y2 = track.bbox
            color = self.colors[track.priority]
            
            # Selected target gets special treatment
            if track.is_selected:
                color = self.colors["selected"]
                # Draw target lock brackets
                bracket_len = 30
                cv2.line(overlay, (x1, y1), (x1 + bracket_len, y1), color, 3)
                cv2.line(overlay, (x1, y1), (x1, y1 + bracket_len), color, 3)
                cv2.line(overlay, (x2, y1), (x2 - bracket_len, y1), color, 3)
                cv2.line(overlay, (x2, y1), (x2, y1 + bracket_len), color, 3)
                cv2.line(overlay, (x1, y2), (x1 + bracket_len, y2), color, 3)
                cv2.line(overlay, (x1, y2), (x1, y2 - bracket_len), color, 3)
                cv2.line(overlay, (x2, y2), (x2 - bracket_len, y2), color, 3)
                cv2.line(overlay, (x2, y2), (x2, y2 - bracket_len), color, 3)
                
                # Draw line to center
                cv2.line(overlay, (int(track.center[0]), int(track.center[1])), 
                        (cx, cy), color, 2)
            else:
                # Regular bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"ID:{track.track_id} {track.class_name} {track.confidence:.2f}"
            if track.is_selected:
                label += f" [TARGET] Threat:{track.threat_score:.2f}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(overlay, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw trajectory
            if len(track.trajectory) >= 2:
                points = np.array(list(track.trajectory), dtype=np.int32)
                
                # Fade effect for trajectory
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    thickness = max(1, int(alpha * 3))
                    pt_color = tuple(int(c * alpha + 255 * (1-alpha)) for c in self.colors["trajectory"])
                    cv2.line(overlay, tuple(points[i-1]), tuple(points[i]), pt_color, thickness)
            
            # Draw velocity vector
            if track.speed > 0.01:
                end_x = int(track.center[0] + track.velocity[0] * 0.5)
                end_y = int(track.center[1] + track.velocity[1] * 0.5)
                cv2.arrowedLine(overlay, 
                               (int(track.center[0]), int(track.center[1])),
                               (end_x, end_y),
                               self.colors["prediction"], 2, tipLength=0.3)
            
            # Draw center dot
            cv2.circle(overlay, (int(track.center[0]), int(track.center[1])), 4, color, -1)
        
        # Draw HUD / Info panel
        self._draw_hud(overlay, w, h)
        
        return overlay
    
    def _draw_hud(self, frame: np.ndarray, w: int, h: int):
        """Draw heads-up display with system info."""
        # Semi-transparent info panel
        panel_h = 140
        panel = np.zeros((panel_h, 350, 3), dtype=np.uint8)
        
        # FPS
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        avg_det = np.mean(self.detection_times) * 1000 if self.detection_times else 0
        
        lines = [
            f"DRONE TRACKING SYSTEM",
            f"FPS: {avg_fps:.1f} | Det: {avg_det:.1f}ms",
            f"Active Tracks: {len(self.tracks)}",
            f"Frame: {self.frame_count}",
        ]
        
        if self.selected_target_id and self.selected_target_id in self.tracks:
            t = self.tracks[self.selected_target_id]
            lines.extend([
                f"",
                f"TARGET: ID {t.track_id}",
                f"Speed: {t.speed*100:.1f}% | Threat: {t.threat_score:.2f}",
                f"Dist Center: {t.distance_from_center:.2f}",
                f"Direction: {t.direction:.1f}°"
            ])
        
        y_offset = 20
        for line in lines:
            cv2.putText(panel, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 18
        
        # Blend panel onto frame
        frame[10:10+panel_h, 10:10+350] = cv2.addWeighted(
            frame[10:10+panel_h, 10:10+350], 0.3, panel, 0.7, 0
        )
        
        # Draw legend
        legend_y = h - 30
        legend_items = [
            (self.colors[TargetPriority.CRITICAL], "CRITICAL"),
            (self.colors[TargetPriority.HIGH], "HIGH"),
            (self.colors[TargetPriority.MEDIUM], "MEDIUM"),
            (self.colors[TargetPriority.LOW], "LOW"),
            (self.colors["selected"], "SELECTED"),
        ]
        x_offset = 10
        for color, text in legend_items:
            cv2.rectangle(frame, (x_offset, legend_y), (x_offset + 15, legend_y + 15), color, -1)
            cv2.putText(frame, text, (x_offset + 20, legend_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            x_offset += 90
    
    def get_target_info(self) -> Optional[Dict]:
        """Get information about the currently selected target."""
        if self.selected_target_id and self.selected_target_id in self.tracks:
            t = self.tracks[self.selected_target_id]
            return {
                'track_id': t.track_id,
                'bbox': t.bbox,
                'center': t.center,
                'velocity': t.velocity,
                'speed': t.speed,
                'direction': t.direction,
                'threat_score': t.threat_score,
                'priority': t.priority.name,
                'confidence': t.confidence,
                'trajectory': list(t.trajectory)
            }
        return None
    
    def export_tracking_data(self, filepath: str):
        """Export all tracking data to JSON."""
        data = {
            'frame_count': self.frame_count,
            'duration_seconds': time.time() - self.start_time,
            'tracks': []
        }
        
        for track in self.tracks.values():
            data['tracks'].append({
                'track_id': track.track_id,
                'class_name': track.class_name,
                'first_seen': track.first_seen,
                'last_seen': track.last_seen,
                'duration': track.age_seconds,
                'detections': track.frame_count,
                'max_threat_score': track.threat_score,
                'trajectory_length': len(track.trajectory),
                'final_bbox': track.bbox
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tracking data exported to {filepath}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Drone Detection & Tracking System')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source: 0=webcam, path=video file, url=stream')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--resolution', type=str, default=None,
                       help='Resolution WxH (e.g., 1280x720)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display (headless mode)')
    parser.add_argument('--save-data', type=str, default=None,
                       help='Save tracking data to JSON file')
    
    args = parser.parse_args()
    
    # Parse source
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # Initialize tracker
    tracker = DroneTracker(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    if tracker.model is None:
        print("Failed to initialize model. Exiting.")
        return
    
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Set resolution if specified
    if args.resolution:
        try:
            w, h = map(int, args.resolution.split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        except:
            print("Invalid resolution format. Use WxH (e.g., 1280x720)")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps:.1f} FPS")
    
    # Initialize video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Recording to: {args.output}")
    
    # Main loop
    print("\n" + "="*50)
    print("DRONE DETECTION & TRACKING SYSTEM")
    print("="*50)
    print("Controls:")
    print("  [Q] Quit")
    print("  [S] Save current frame")
    print("  [T] Cycle target selection strategy")
    print("  [P] Pause/Resume")
    print("  [R] Reset all tracks")
    print("="*50 + "\n")
    
    paused = False
    target_strategies = ["threat", "closest", "newest", "fastest"]
    strategy_idx = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of stream")
                    break
                
                # Process frame
                output = tracker.process_frame(frame)
                
                # Write to file
                if writer:
                    writer.write(output)
                
                # Display
                if not args.no_display:
                    cv2.imshow('Drone Tracker', output)
            
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, output)
                    print(f"Saved: {filename}")
                elif key == ord('t'):
                    strategy_idx = (strategy_idx + 1) % len(target_strategies)
                    print(f"Target strategy: {target_strategies[strategy_idx]}")
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('r'):
                    tracker.tracks.clear()
                    tracker.selected_target_id = None
                    print("Tracks reset")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Export data if requested
        if args.save_data:
            tracker.export_tracking_data(args.save_data)
        
        # Print summary
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Total frames processed: {tracker.frame_count}")
        print(f"Duration: {time.time() - tracker.start_time:.1f}s")
        print(f"Unique tracks: {tracker.next_track_id - 1}")
        print(f"Final active tracks: {len(tracker.tracks)}")
        if tracker.selected_target_id:
            t = tracker.tracks.get(tracker.selected_target_id)
            if t:
                print(f"Final target: ID {t.track_id} (Threat: {t.threat_score:.3f})")
        print("="*50)


if __name__ == "__main__":
    main()