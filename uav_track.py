"""
Drone Detection, Path Tracking & Target Selection System
==========================================================

A comprehensive computer vision system for detecting drones using YOLO,
tracking their paths over time, and intelligently selecting targets.

Features:
- Automatic frame resizing to fit screen
- Maintains aspect ratio with letterboxing
- Fullscreen mode support
- Multi-monitor support

Requirements:
    pip install ultralytics opencv-python numpy screeninfo

Usage:
    python drone_tracker.py --source your_video.mp4
    python drone_tracker.py --source your_video.mp4 --fit-screen
    python drone_tracker.py --source your_video.mp4 --fullscreen
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
import tkinter as tk
from pathlib import Path

# Try to import screeninfo for automatic screen detection
try:
    from screeninfo import get_monitors
    SCREENINFO_AVAILABLE = True
except ImportError:
    SCREENINFO_AVAILABLE = False
    print("Note: Install 'screeninfo' for automatic screen detection: pip install screeninfo")

# Try to import ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


# =============================================================================
# SCREEN UTILITIES
# =============================================================================

class ScreenFitter:
    """
    Handles automatic frame resizing to fit screen while maintaining aspect ratio.
    Supports multiple fit modes and fullscreen.
    """
    
    def __init__(self, fit_mode: str = "fit", target_width: int = None, 
                 target_height: int = None, margin: int = 80):
        """
        Args:
            fit_mode: "fit" (fit in screen), "fill" (fill screen), "none" (no resize)
            target_width: Manual target width (overrides auto-detect)
            target_height: Manual target height (overrides auto-detect)
            margin: Pixel margin from screen edges
        """
        self.fit_mode = fit_mode
        self.margin = margin
        self.target_width = target_width
        self.target_height = target_height
        self.screen_width = 1920  # defaults
        self.screen_height = 1080
        self.scale_factor = 1.0
        
        # Detect screen size
        self._detect_screen()
    
    def _detect_screen(self):
        """Detect primary screen resolution."""
        # Method 1: screeninfo library
        if SCREENINFO_AVAILABLE:
            try:
                monitors = get_monitors()
                if monitors:
                    primary = monitors[0]
                    self.screen_width = primary.width
                    self.screen_height = primary.height
                    print(f"Detected screen: {self.screen_width}x{self.screen_height}")
                    return
            except Exception as e:
                print(f"screeninfo error: {e}")
        
        # Method 2: tkinter
        try:
            root = tk.Tk()
            self.screen_width = root.winfo_screenwidth()
            self.screen_height = root.winfo_screenheight()
            root.destroy()
            print(f"Detected screen (tkinter): {self.screen_width}x{self.screen_height}")
            return
        except Exception as e:
            print(f"tkinter error: {e}")
        
        # Method 3: OpenCV fullscreen test
        try:
            test_cap = cv2.VideoCapture(0)
            if test_cap.isOpened():
                test_cap.release()
        except:
            pass
        
        print(f"Using default screen size: {self.screen_width}x{self.screen_height}")
    
    def get_target_size(self, frame_width: int, frame_height: int) -> Tuple[int, int, float]:
        """
        Calculate target display size maintaining aspect ratio.
        
        Returns:
            (target_w, target_h, scale_factor)
        """
        if self.fit_mode == "none":
            return frame_width, frame_height, 1.0
        
        # Use manual targets if provided
        max_w = self.target_width or (self.screen_width - self.margin * 2)
        max_h = self.target_height or (self.screen_height - self.margin * 2)
        
        frame_aspect = frame_width / frame_height
        screen_aspect = max_w / max_h
        
        if self.fit_mode == "fit":
            # Fit entire frame within screen (may leave black bars)
            if frame_aspect > screen_aspect:
                # Frame is wider - fit to width
                target_w = max_w
                target_h = int(target_w / frame_aspect)
            else:
                # Frame is taller - fit to height
                target_h = max_h
                target_w = int(target_h * frame_aspect)
        
        elif self.fit_mode == "fill":
            # Fill entire screen (may crop frame)
            if frame_aspect > screen_aspect:
                target_h = max_h
                target_w = int(target_h * frame_aspect)
            else:
                target_w = max_w
                target_h = int(target_w / frame_aspect)
        
        else:
            target_w, target_h = frame_width, frame_height
        
        # Calculate scale factor relative to original
        self.scale_factor = target_w / frame_width
        
        return target_w, target_h, self.scale_factor
    
    def resize_frame(self, frame: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize frame to fit screen while maintaining aspect ratio.
        Adds black letterbox/pillarbox bars if needed.
        """
        if self.fit_mode == "none":
            return frame
        
        h, w = frame.shape[:2]
        target_w, target_h, _ = self.get_target_size(w, h)
        
        # Resize frame
        resized = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)
        
        # If fit mode, add black bars to make exact screen size
        if self.fit_mode == "fit":
            max_w = self.target_width or (self.screen_width - self.margin * 2)
            max_h = self.target_height or (self.screen_height - self.margin * 2)
            
            if target_w < max_w or target_h < max_h:
                # Create black canvas
                canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                
                # Center the resized frame
                y_offset = (max_h - target_h) // 2
                x_offset = (max_w - target_w) // 2
                
                canvas[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = resized
                return canvas
        
        return resized
    
    def resize_to_window(self, frame: np.ndarray, window_name: str) -> np.ndarray:
        """
        Resize frame to fit current window size dynamically.
        """
        try:
            rect = cv2.getWindowImageRect(window_name)
            if rect and rect[2] > 0 and rect[3] > 0:
                win_w, win_h = rect[2], rect[3]
                h, w = frame.shape[:2]
                frame_aspect = w / h
                win_aspect = win_w / win_h
                
                if frame_aspect > win_aspect:
                    new_w = win_w
                    new_h = int(win_w / frame_aspect)
                else:
                    new_h = win_h
                    new_w = int(win_h * frame_aspect)
                
                return cv2.resize(frame, (new_w, new_h))
        except:
            pass
        
        return frame
    
    def get_display_info(self) -> str:
        """Get string with display configuration info."""
        return (f"Screen: {self.screen_width}x{self.screen_height} | "
                f"Mode: {self.fit_mode} | "
                f"Margin: {self.margin}px")


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

class TargetPriority(Enum):
    """Priority levels for target selection."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NONE = 0


@dataclass
class DroneTrack:
    """Represents a tracked drone with history and computed metrics."""
    
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "drone"
    
    trajectory: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    direction: float = 0.0
    distance_from_center: float = 0.0
    
    priority: TargetPriority = TargetPriority.NONE
    threat_score: float = 0.0
    is_selected: bool = False
    
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 0
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float,
               frame_time: float, frame_shape: Tuple[int, int]):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        self.bbox = bbox
        self.center = (center_x, center_y)
        self.confidence = confidence
        self.last_seen = frame_time
        self.frame_count += 1
        
        self.trajectory.append((center_x, center_y))
        self.timestamps.append(frame_time)
        
        if len(self.trajectory) >= 2:
            self._calculate_velocity(frame_shape)
        
        h, w = frame_shape[:2]
        frame_cx, frame_cy = w / 2.0, h / 2.0
        self.distance_from_center = np.sqrt(
            ((center_x - frame_cx) / w) ** 2 +
            ((center_y - frame_cy) / h) ** 2
        )
    
    def _calculate_velocity(self, frame_shape: Tuple[int, int]):
        if len(self.trajectory) < 2:
            return
        
        n = min(5, len(self.trajectory))
        recent_points = list(self.trajectory)[-n:]
        recent_times = list(self.timestamps)[-n:]
        
        if len(recent_times) < 2 or recent_times[-1] == recent_times[0]:
            return
        
        dx = recent_points[-1][0] - recent_points[0][0]
        dy = recent_points[-1][1] - recent_points[0][1]
        dt = recent_times[-1] - recent_times[0]
        
        if dt > 0:
            self.velocity = (dx / dt, dy / dt)
            self.speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            self.direction = np.degrees(np.arctan2(-dy, dx))
        
        h, w = frame_shape[:2]
        diagonal = np.sqrt(h**2 + w**2)
        self.speed = self.speed / diagonal if diagonal > 0 else 0
    
    def calculate_threat_score(self, frame_shape: Tuple[int, int]):
        h, w = frame_shape[:2]
        frame_cx, frame_cy = w / 2.0, h / 2.0
        
        proximity = max(0, 1.0 - self.distance_from_center * 2)
        speed_factor = min(self.speed * 10, 1.0)
        
        approach = 0.0
        if len(self.trajectory) >= 2:
            prev = self.trajectory[-2]
            curr = self.trajectory[-1]
            prev_dist = np.sqrt((prev[0] - frame_cx)**2 + (prev[1] - frame_cy)**2)
            curr_dist = np.sqrt((curr[0] - frame_cx)**2 + (curr[1] - frame_cy)**2)
            if curr_dist < prev_dist:
                approach = min((prev_dist - curr_dist) / max(prev_dist, 1), 1.0)
        
        confidence_factor = self.confidence
        
        self.threat_score = (
            proximity * 0.35 +
            speed_factor * 0.25 +
            approach * 0.25 +
            confidence_factor * 0.15
        )
        
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
        return (current_time - self.last_seen) > timeout
    
    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen


class DroneTracker:
    """Main tracking system."""
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        drone_classes: List[str] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        self.drone_classes = drone_classes or [
            "drone", "uav", "quadcopter", "hexacopter",
            "aircraft", "airplane", "bird"
        ]
        
        self.model = None
        self._init_model(model_path)
        
        self.tracks: Dict[int, DroneTrack] = {}
        self.next_track_id = 1
        self.frame_count = 0
        self.start_time = time.time()
        
        self.selected_target_id: Optional[int] = None
        
        self.fps_history = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        
        self.colors = {
            TargetPriority.CRITICAL: (0, 0, 255),
            TargetPriority.HIGH: (0, 165, 255),
            TargetPriority.MEDIUM: (0, 255, 255),
            TargetPriority.LOW: (0, 255, 0),
            TargetPriority.NONE: (128, 128, 128),
            "selected": (255, 0, 255),
            "trajectory": (0, 255, 0),
            "prediction": (255, 0, 0),
        }
    
    def _init_model(self, model_path: str):
        if not ULTRALYTICS_AVAILABLE:
            print("ERROR: ultralytics not installed.")
            return
        
        try:
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model: {model_path}")
            if hasattr(self.model, 'names'):
                print(f"Model classes: {self.model.names}")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            try:
                self.model = YOLO("yolov8n.pt")
            except Exception as e2:
                print(f"Failed to load default model: {e2}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        if self.model is None:
            return []
        
        detections = []
        try:
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id] if hasattr(result, 'names') else str(class_id)
                    
                    is_drone = any(drone_term in class_name.lower()
                                  for drone_term in self.drone_classes)
                    
                    if is_drone or confidence > 0.6:
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
        current_time = time.time()
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            if track.speed > 0 and len(track.trajectory) >= 2:
                dt = current_time - track.last_seen
                pred_x = track.center[0] + track.velocity[0] * dt
                pred_y = track.center[1] + track.velocity[1] * dt
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
        
        matched_tracks = set()
        matched_dets = set()
        
        while True:
            if iou_matrix.size == 0 or np.max(iou_matrix) < 0.3:
                break
            
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            tid = track_ids[i]
            
            self.tracks[tid].update(
                detections[j]['bbox'],
                detections[j]['confidence'],
                current_time,
                frame_shape
            )
            matched_tracks.add(tid)
            matched_dets.add(j)
            
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1
        
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
        
        lost_ids = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks and track.is_lost(current_time):
                lost_ids.append(tid)
        
        for tid in lost_ids:
            del self.tracks[tid]
    
    def select_target(self, frame_shape: Tuple[int, int], strategy: str = "threat"):
        if not self.tracks:
            self.selected_target_id = None
            return
        
        for track in self.tracks.values():
            track.is_selected = False
        
        for track in self.tracks.values():
            track.calculate_threat_score(frame_shape)
        
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
        self.frame_count += 1
        frame_time = time.time()
        h, w = frame.shape[:2]
        
        t0 = time.time()
        detections = self.detect(frame)
        det_time = time.time() - t0
        self.detection_times.append(det_time)
        
        if detections:
            self._associate_detections(detections, (h, w))
        
        self.select_target((h, w), strategy="threat")
        self.fps_history.append(1.0 / (time.time() - frame_time + 1e-6))
        
        return self._visualize(frame)
    
    def _visualize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        cx, cy = w // 2, h // 2
        cv2.line(overlay, (cx - 20, cy), (cx + 20, cy), (200, 200, 200), 1)
        cv2.line(overlay, (cx, cy - 20), (cx, cy + 20), (200, 200, 200), 1)
        cv2.circle(overlay, (cx, cy), 5, (200, 200, 200), 1)
        
        for radius in [w//6, w//4, w//3]:
            cv2.circle(overlay, (cx, cy), radius, (50, 50, 50), 1)
        
        for track in self.tracks.values():
            x1, y1, x2, y2 = track.bbox
            color = self.colors[track.priority]
            
            if track.is_selected:
                color = self.colors["selected"]
                bracket_len = 30
                cv2.line(overlay, (x1, y1), (x1 + bracket_len, y1), color, 3)
                cv2.line(overlay, (x1, y1), (x1, y1 + bracket_len), color, 3)
                cv2.line(overlay, (x2, y1), (x2 - bracket_len, y1), color, 3)
                cv2.line(overlay, (x2, y1), (x2, y1 + bracket_len), color, 3)
                cv2.line(overlay, (x1, y2), (x1 + bracket_len, y2), color, 3)
                cv2.line(overlay, (x1, y2), (x1, y2 - bracket_len), color, 3)
                cv2.line(overlay, (x2, y2), (x2 - bracket_len, y2), color, 3)
                cv2.line(overlay, (x2, y2), (x2, y2 - bracket_len), color, 3)
                cv2.line(overlay, (int(track.center[0]), int(track.center[1])),
                        (cx, cy), color, 2)
            else:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{track.track_id} {track.class_name} {track.confidence:.2f}"
            if track.is_selected:
                label += f" [TARGET] Threat:{track.threat_score:.2f}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(overlay, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if len(track.trajectory) >= 2:
                points = np.array(list(track.trajectory), dtype=np.int32)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    thickness = max(1, int(alpha * 3))
                    pt_color = tuple(int(c * alpha + 255 * (1-alpha)) for c in self.colors["trajectory"])
                    cv2.line(overlay, tuple(points[i-1]), tuple(points[i]), pt_color, thickness)
            
            if track.speed > 0.01:
                end_x = int(track.center[0] + track.velocity[0] * 0.5)
                end_y = int(track.center[1] + track.velocity[1] * 0.5)
                cv2.arrowedLine(overlay,
                               (int(track.center[0]), int(track.center[1])),
                               (end_x, end_y),
                               self.colors["prediction"], 2, tipLength=0.3)
            
            cv2.circle(overlay, (int(track.center[0]), int(track.center[1])), 4, color, -1)
        
        self._draw_hud(overlay, w, h)
        return overlay
    
    def _draw_hud(self, frame: np.ndarray, w: int, h: int):
        panel_h = 160
        panel = np.zeros((panel_h, 380, 3), dtype=np.uint8)
        
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
                f"Direction: {t.direction:.1f}deg"
            ])
        
        y_offset = 20
        for line in lines:
            cv2.putText(panel, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 18
        
        frame[10:10+panel_h, 10:10+380] = cv2.addWeighted(
            frame[10:10+panel_h, 10:10+380], 0.3, panel, 0.7, 0
        )
        
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
                       help='YOLO model path')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path')
    
    # ===== NEW: Screen fitting options =====
    parser.add_argument('--fit-screen', action='store_true',
                       help='Auto-resize frame to fit screen with black bars')
    parser.add_argument('--fill-screen', action='store_true',
                       help='Resize frame to fill screen (may crop)')
    parser.add_argument('--fullscreen', action='store_true',
                       help='Open window in fullscreen mode')
    parser.add_argument('--window-width', type=int, default=None,
                       help='Force window width (px)')
    parser.add_argument('--window-height', type=int, default=None,
                       help='Force window height (px)')
    parser.add_argument('--no-resize', action='store_true',
                       help='Keep original frame size (no resizing)')
    parser.add_argument('--margin', type=int, default=80,
                       help='Screen margin in pixels (default: 80)')
    
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display (headless mode)')
    parser.add_argument('--save-data', type=str, default=None,
                       help='Save tracking data to JSON file')
    
    args = parser.parse_args()
    
    # Determine fit mode
    if args.no_resize:
        fit_mode = "none"
    elif args.fill_screen:
        fit_mode = "fill"
    else:
        fit_mode = "fit"  # default
    
    # Initialize screen fitter
    screen_fitter = ScreenFitter(
        fit_mode=fit_mode,
        target_width=args.window_width,
        target_height=args.window_height,
        margin=args.margin
    )
    print(screen_fitter.get_display_info())
    
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
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original video: {orig_width}x{orig_height} @ {fps:.1f} FPS")
    
    # Calculate display size
    disp_w, disp_h, scale = screen_fitter.get_target_size(orig_width, orig_height)
    print(f"Display size: {disp_w}x{disp_h} (scale: {scale:.2f}x)")
    
    # Create window
    window_name = "Drone Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("Fullscreen mode enabled")
    else:
        # Set initial window size
        if fit_mode == "none":
            cv2.resizeWindow(window_name, orig_width, orig_height)
        else:
            cv2.resizeWindow(window_name, disp_w, disp_h)
    
    # Video writer setup
    writer = None
    if args.output:
        # IMPORTANT: Write at ORIGINAL resolution, not display size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_width, orig_height))
        print(f"Recording to: {args.output} ({orig_width}x{orig_height})")
    
    # Main loop
    print("\n" + "="*55)
    print("DRONE DETECTION & TRACKING SYSTEM")
    print("="*55)
    print("Controls:")
    print("  [Q] Quit")
    print("  [S] Save current frame")
    print("  [T] Cycle target selection strategy")
    print("  [P] Pause/Resume")
    print("  [R] Reset all tracks")
    print("  [F] Toggle fullscreen")
    print("  [+/-] Increase/decrease window size")
    print("  [0] Reset window to original size")
    print("="*55 + "\n")
    
    paused = False
    target_strategies = ["threat", "closest", "newest", "fastest"]
    strategy_idx = 0
    is_fullscreen = args.fullscreen
    current_scale = 1.0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of stream")
                    break
                
                # Process frame at ORIGINAL resolution (for detection accuracy)
                output = tracker.process_frame(frame)
                
                # Write original resolution to file
                if writer:
                    writer.write(output)
                
                # Resize for display only
                if not args.no_display:
                    display_frame = screen_fitter.resize_frame(output)
                    
                    # Handle window resize during runtime
                    if not is_fullscreen:
                        display_frame = screen_fitter.resize_to_window(display_frame, window_name)
                    
                    cv2.imshow(window_name, display_frame)
            
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"capture_{int(time.time())}.jpg"
                    # Save at display resolution
                    cv2.imwrite(filename, display_frame)
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
                elif key == ord('f'):
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, disp_w, disp_h)
                    print(f"Fullscreen: {is_fullscreen}")
                elif key == ord('+') or key == ord('='):
                    current_scale += 0.1
                    new_w = int(orig_width * current_scale)
                    new_h = int(orig_height * current_scale)
                    cv2.resizeWindow(window_name, new_w, new_h)
                    print(f"Window scale: {current_scale:.1f}x")
                elif key == ord('-'):
                    current_scale = max(0.2, current_scale - 0.1)
                    new_w = int(orig_width * current_scale)
                    new_h = int(orig_height * current_scale)
                    cv2.resizeWindow(window_name, new_w, new_h)
                    print(f"Window scale: {current_scale:.1f}x")
                elif key == ord('0'):
                    current_scale = 1.0
                    cv2.resizeWindow(window_name, orig_width, orig_height)
                    print("Window reset to original size")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        if args.save_data:
            tracker.export_tracking_data(args.save_data)
        
        print("\n" + "="*55)
        print("SESSION SUMMARY")
        print("="*55)
        print(f"Total frames processed: {tracker.frame_count}")
        print(f"Duration: {time.time() - tracker.start_time:.1f}s")
        print(f"Unique tracks: {tracker.next_track_id - 1}")
        print(f"Final active tracks: {len(tracker.tracks)}")
        if tracker.selected_target_id:
            t = tracker.tracks.get(tracker.selected_target_id)
            if t:
                print(f"Final target: ID {t.track_id} (Threat: {t.threat_score:.3f})")
        print("="*55)


if __name__ == "__main__":
    main()