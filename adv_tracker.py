"""
Fighter Jet HUD Drone Detection & Tracking System
==================================================

Features:
- Night Vision Mode (green phosphor / thermal)
- Fighter Jet-style HUD Overlay
- Real-time Path Tracking Graphs (side panel)
- ROI (Region of Interest) Selection
- Target Lock System
- Threat Assessment Display

Requirements:
    pip install ultralytics opencv-python numpy matplotlib screeninfo

Usage:
    python fighter_hud_tracker.py --source video.mp4 --night-vision
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
import math

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Graphs disabled.")

try:
    from screeninfo import get_monitors
    SCREENINFO_AVAILABLE = True
except ImportError:
    SCREENINFO_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# =============================================================================
# CONFIGURATION & ENUMS
# =============================================================================

class TargetPriority(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NONE = 0


class NightVisionMode(Enum):
    OFF = "off"
    GREEN = "green"      # Classic green phosphor
    WHITE_HOT = "white_hot"   # Thermal - white hot
    BLACK_HOT = "black_hot"   # Thermal - black hot
    AMBER = "amber"      # Amber phosphor


@dataclass
class DroneTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "drone"
    
    trajectory: deque = field(default_factory=lambda: deque(maxlen=200))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=200))
    
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    direction: float = 0.0
    distance_from_center: float = 0.0
    
    priority: TargetPriority = TargetPriority.NONE
    threat_score: float = 0.0
    is_selected: bool = False
    is_locked: bool = False
    lock_time: float = 0.0
    
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 0
    
    # Path history for graphing
    path_x: deque = field(default_factory=lambda: deque(maxlen=100))
    path_y: deque = field(default_factory=lambda: deque(maxlen=100))
    
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
        self.path_x.append(center_x)
        self.path_y.append(center_y)
        
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
        
        self.threat_score = proximity * 0.35 + speed_factor * 0.25 + approach * 0.25 + self.confidence * 0.15
        
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


# =============================================================================
# NIGHT VISION PROCESSOR
# =============================================================================

class NightVisionProcessor:
    """Applies night vision / thermal imaging effects."""
    
    def __init__(self, mode: NightVisionMode = NightVisionMode.GREEN):
        self.mode = mode
        self.noise_intensity = 15
        
        # Scanline effect
        self.scanline_offset = 0
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == NightVisionMode.OFF:
            return frame
        
        # Convert to grayscale base
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
        
        # Apply noise
        noise = np.random.normal(0, self.noise_intensity, gray.shape).astype(np.float32)
        gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Apply color map based on mode
        if self.mode == NightVisionMode.GREEN:
            # Classic green phosphor
            result = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            result[:, :, 1] = gray  # Green channel
            result[:, :, 0] = (gray * 0.3).astype(np.uint8)  # Slight blue
            result[:, :, 2] = (gray * 0.1).astype(np.uint8)  # Minimal red
        
        elif self.mode == NightVisionMode.WHITE_HOT:
            # Thermal - white hot
            result = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        
        elif self.mode == NightVisionMode.BLACK_HOT:
            # Thermal - black hot (inverted)
            result = cv2.applyColorMap(255 - gray, cv2.COLORMAP_HOT)
        
        elif self.mode == NightVisionMode.AMBER:
            # Amber phosphor
            result = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            result[:, :, 2] = gray  # Red
            result[:, :, 1] = (gray * 0.6).astype(np.uint8)  # Some green = amber
        
        # Add scanlines
        result = self._add_scanlines(result)
        
        # Add vignette
        result = self._add_vignette(result)
        
        return result
    
    def _add_scanlines(self, frame: np.ndarray) -> np.ndarray:
        """Add CRT scanline effect."""
        result = frame.copy()
        for i in range(0, frame.shape[0], 2):
            result[i, :] = (result[i, :] * 0.85).astype(np.uint8)
        return result
    
    def _add_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Add edge darkening."""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Vignette mask (1.0 at center, 0.7 at edges)
        mask = 1.0 - (dist / max_dist) * 0.3
        mask = np.clip(mask, 0.7, 1.0)
        
        result = (frame.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)
        return result
    
    def cycle_mode(self):
        modes = list(NightVisionMode)
        idx = modes.index(self.mode)
        self.mode = modes[(idx + 1) % len(modes)]
        return self.mode


# =============================================================================
# PATH GRAPH GENERATOR
# =============================================================================

class PathGraphGenerator:
    """Generates real-time path tracking graphs using matplotlib."""
    
    def __init__(self, width: int = 400, height: int = 300):
        self.width = width
        self.height = height
        self.fig = None
        self.canvas = None
        
        if MATPLOTLIB_AVAILABLE:
            self._init_figure()
    
    def _init_figure(self):
        """Initialize matplotlib figure."""
        dpi = 100
        fig_w = self.width / dpi
        fig_h = self.height / dpi
        
        self.fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor='black')
        self.ax = self.fig.add_subplot(111, facecolor='black')
        self.ax.set_facecolor('black')
        
        # Style for HUD look
        self.ax.tick_params(colors='lime', labelsize=8)
        self.ax.spines['bottom'].set_color('lime')
        self.ax.spines['top'].set_color('lime')
        self.ax.spines['left'].set_color('lime')
        self.ax.spines['right'].set_color('lime')
        
        self.canvas = FigureCanvasAgg(self.fig)
    
    def generate_path_graph(self, tracks: Dict[int, DroneTrack], 
                           selected_id: Optional[int],
                           frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Generate path tracking graph image."""
        if not MATPLOTLIB_AVAILABLE or not tracks:
            return None
        
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='lime', labelsize=7)
        
        h, w = frame_shape[:2]
        
        # Plot each track's path
        colors = {'selected': '#FF00FF', 'critical': '#FF0000', 'high': '#FFA500',
                 'medium': '#FFFF00', 'low': '#00FF00', 'none': '#808080'}
        
        for tid, track in tracks.items():
            if len(track.path_x) < 2:
                continue
            
            x_vals = list(track.path_x)
            y_vals = [h - y for y in track.path_y]  # Flip Y for graph
            
            color = colors.get(track.priority.name.lower(), '#00FF00')
            linewidth = 3 if track.is_selected else 1.5
            alpha = 1.0 if track.is_selected else 0.6
            
            self.ax.plot(x_vals, y_vals, color=color, linewidth=linewidth, 
                        alpha=alpha, label=f'ID {tid}')
            
            # Mark current position
            if track.is_selected:
                self.ax.scatter([x_vals[-1]], [y_vals[-1]], 
                             color='#FF00FF', s=100, marker='x', linewidths=3)
            else:
                self.ax.scatter([x_vals[-1]], [y_vals[-1]], 
                             color=color, s=30, marker='o')
        
        # Set limits
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.set_xlabel('X Position', color='lime', fontsize=8)
        self.ax.set_ylabel('Y Position', color='lime', fontsize=8)
        self.ax.set_title('DRONE PATH TRACKING', color='lime', fontsize=10, fontweight='bold')
        self.ax.grid(True, color='darkgreen', alpha=0.3, linestyle='--')
        
        # Tight layout
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Convert to numpy array
        buf = self.canvas.buffer_rgba()
        graph_img = np.asarray(buf)
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
        
        return graph_img
    
    def generate_velocity_graph(self, tracks: Dict[int, DroneTrack],
                               selected_id: Optional[int]) -> Optional[np.ndarray]:
        """Generate velocity over time graph."""
        if not MATPLOTLIB_AVAILABLE or not tracks:
            return None
        
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='lime', labelsize=7)
        
        for tid, track in tracks.items():
            if len(track.timestamps) < 2:
                continue
            
            times = [t - track.first_seen for t in track.timestamps]
            # Calculate instantaneous speeds
            speeds = []
            for i in range(1, len(track.trajectory)):
                dx = track.trajectory[i][0] - track.trajectory[i-1][0]
                dy = track.trajectory[i][1] - track.trajectory[i-1][1]
                dt = track.timestamps[i] - track.timestamps[i-1]
                if dt > 0:
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    speeds.append(speed)
                else:
                    speeds.append(0)
            
            times = times[1:]  # Match speed length
            
            color = '#FF00FF' if track.is_selected else '#00FF00'
            linewidth = 2.5 if track.is_selected else 1.2
            alpha = 1.0 if track.is_selected else 0.5
            
            self.ax.plot(times, speeds, color=color, linewidth=linewidth, alpha=alpha)
        
        self.ax.set_xlabel('Time (s)', color='lime', fontsize=8)
        self.ax.set_ylabel('Speed (px/s)', color='lime', fontsize=8)
        self.ax.set_title('VELOCITY PROFILE', color='lime', fontsize=10, fontweight='bold')
        self.ax.grid(True, color='darkgreen', alpha=0.3, linestyle='--')
        self.fig.tight_layout()
        self.canvas.draw()
        
        buf = self.canvas.buffer_rgba()
        graph_img = np.asarray(buf)
        return cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)


# =============================================================================
# FIGHTER HUD RENDERER
# =============================================================================

class FighterHUDRenderer:
    """Renders fighter jet-style HUD overlay."""
    
    def __init__(self):
        self.hud_color = (0, 255, 0)  # Green HUD
        self.hud_color_dim = (0, 180, 0)
        self.alert_color = (0, 0, 255)
        self.lock_color = (255, 0, 255)
        
        # HUD animation state
        self.frame_counter = 0
        self.blink_state = True
    
    def update(self):
        self.frame_counter += 1
        if self.frame_counter % 15 == 0:
            self.blink_state = not self.blink_state
    
    def draw_full_hud(self, frame: np.ndarray, tracks: Dict[int, DroneTrack],
                     selected_id: Optional[int], fps: float, 
                     night_vision: NightVisionMode) -> np.ndarray:
        """Draw complete fighter jet HUD."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cx, cy = w // 2, h // 2
        
        # Draw pitch ladder (artificial horizon reference)
        self._draw_pitch_ladder(overlay, cx, cy, w, h)
        
        # Draw center reticle
        self._draw_reticle(overlay, cx, cy, selected_id, tracks)
        
        # Draw heading tape (top)
        self._draw_heading_tape(overlay, w, h)
        
        # Draw altitude/range bars (sides)
        self._draw_side_bars(overlay, w, h, tracks, selected_id)
        
        # Draw target info boxes
        self._draw_target_boxes(overlay, tracks, selected_id, w, h)
        
        # Draw bottom data strip
        self._draw_data_strip(overlay, w, h, fps, night_vision)
        
        # Draw threat warnings
        self._draw_threat_warnings(overlay, tracks, w, h)
        
        # Draw lock indicator if target locked
        if selected_id and selected_id in tracks and tracks[selected_id].is_locked:
            self._draw_lock_indicator(overlay, cx, cy, w, h)
        
        return overlay
    
    def _draw_pitch_ladder(self, frame: np.ndarray, cx: int, cy: int, w: int, h: int):
        """Draw artificial horizon pitch ladder."""
        color = self.hud_color_dim
        
        # Horizontal reference line
        cv2.line(frame, (cx - 150, cy), (cx - 50, cy), color, 1)
        cv2.line(frame, (cx + 50, cy), (cx + 150, cy), color, 1)
        
        # Pitch marks
        for offset in [-80, -40, 40, 80]:
            y = cy + offset
            cv2.line(frame, (cx - 30, y), (cx + 30, y), color, 1)
            cv2.line(frame, (cx - 30, y), (cx - 20, y - 5), color, 1)
            cv2.line(frame, (cx + 30, y), (cx + 20, y - 5), color, 1)
    
    def _draw_reticle(self, frame: np.ndarray, cx: int, cy: int,
                     selected_id: Optional[int], tracks: Dict[int, DroneTrack]):
        """Draw center targeting reticle."""
        color = self.lock_color if (selected_id and selected_id in tracks 
                                   and tracks[selected_id].is_locked) else self.hud_color
        
        # Main circle
        cv2.circle(frame, (cx, cy), 40, color, 2)
        cv2.circle(frame, (cx, cy), 3, color, -1)
        
        # Crosshair
        cv2.line(frame, (cx - 60, cy), (cx - 45, cy), color, 2)
        cv2.line(frame, (cx + 45, cy), (cx + 60, cy), color, 2)
        cv2.line(frame, (cx, cy - 60), (cx, cy - 45), color, 2)
        cv2.line(frame, (cx, cy + 45), (cx, cy + 60), color, 2)
        
        # Corner brackets
        bracket = 25
        cv2.line(frame, (cx - 50, cy - 50), (cx - 50 + bracket, cy - 50), color, 2)
        cv2.line(frame, (cx - 50, cy - 50), (cx - 50, cy - 50 + bracket), color, 2)
        cv2.line(frame, (cx + 50, cy - 50), (cx + 50 - bracket, cy - 50), color, 2)
        cv2.line(frame, (cx + 50, cy - 50), (cx + 50, cy - 50 + bracket), color, 2)
        cv2.line(frame, (cx - 50, cy + 50), (cx - 50 + bracket, cy + 50), color, 2)
        cv2.line(frame, (cx - 50, cy + 50), (cx - 50, cy + 50 - bracket), color, 2)
        cv2.line(frame, (cx + 50, cy + 50), (cx + 50 - bracket, cy + 50), color, 2)
        cv2.line(frame, (cx + 50, cy + 50), (cx + 50, cy + 50 - bracket), color, 2)
    
    def _draw_heading_tape(self, frame: np.ndarray, w: int, h: int):
        """Draw heading indicator at top."""
        y = 40
        color = self.hud_color
        
        # Background strip
        cv2.rectangle(frame, (w//2 - 200, y - 20), (w//2 + 200, y + 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (w//2 - 200, y - 20), (w//2 + 200, y + 20), color, 1)
        
        # Center marker
        cv2.line(frame, (w//2, y - 15), (w//2, y + 15), (255, 255, 255), 2)
        cv2.putText(frame, "N", (w//2 - 7, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Tick marks
        for i in range(-4, 5):
            if i == 0:
                continue
            x = w//2 + i * 40
            cv2.line(frame, (x, y - 10), (x, y + 10), color, 1)
            heading = (i * 10) % 360
            cv2.putText(frame, str(heading), (x - 10, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def _draw_side_bars(self, frame: np.ndarray, w: int, h: int,
                       tracks: Dict[int, DroneTrack], selected_id: Optional[int]):
        """Draw altitude/range bars on sides."""
        color = self.hud_color_dim
        
        # Left bar - Altitude/Range
        bar_x = 30
        cv2.line(frame, (bar_x, 100), (bar_x, h - 100), color, 1)
        
        # Ticks
        for i in range(10):
            y = 100 + i * (h - 200) // 10
            cv2.line(frame, (bar_x, y), (bar_x + 10, y), color, 1)
        
        # Right bar - Speed
        bar_x = w - 30
        cv2.line(frame, (bar_x, 100), (bar_x, h - 100), color, 1)
        
        for i in range(10):
            y = 100 + i * (h - 200) // 10
            cv2.line(frame, (bar_x - 10, y), (bar_x, y), color, 1)
        
        # Selected target speed indicator
        if selected_id and selected_id in tracks:
            speed = tracks[selected_id].speed * 100
            y_pos = h - 100 - int(speed * (h - 200) / 100)
            y_pos = max(100, min(h - 100, y_pos))
            cv2.circle(frame, (w - 30, y_pos), 5, self.lock_color, -1)
    
    def _draw_target_boxes(self, frame: np.ndarray, tracks: Dict[int, DroneTrack],
                          selected_id: Optional[int], w: int, h: int):
        """Draw target acquisition boxes."""
        priority_colors = {
            TargetPriority.CRITICAL: (0, 0, 255),
            TargetPriority.HIGH: (0, 165, 255),
            TargetPriority.MEDIUM: (0, 255, 255),
            TargetPriority.LOW: (0, 255, 0),
            TargetPriority.NONE: (128, 128, 128),
        }
        
        for tid, track in tracks.items():
            x1, y1, x2, y2 = track.bbox
            color = priority_colors[track.priority]
            
            if track.is_selected:
                color = self.lock_color
                
                # Lock brackets (animated)
                bracket_len = 35 + (5 if self.blink_state else 0)
                thickness = 3
                
                # Top-left
                cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, thickness)
                cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, thickness)
                # Top-right
                cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, thickness)
                cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, thickness)
                # Bottom-left
                cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, thickness)
                cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, thickness)
                # Bottom-right
                cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, thickness)
                cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, thickness)
                
                # Lock diamond
                diamond_size = 10
                cx_t, cy_t = int(track.center[0]), int(track.center[1])
                diamond = np.array([
                    [cx_t, cy_t - diamond_size],
                    [cx_t + diamond_size, cy_t],
                    [cx_t, cy_t + diamond_size],
                    [cx_t - diamond_size, cy_t]
                ], np.int32)
                cv2.polylines(frame, [diamond], True, self.lock_color, 2)
                
                # Target data box
                self._draw_target_data_box(frame, track, x2 + 10, y1, w, h)
                
                # Line to center reticle
                cv2.line(frame, (cx_t, cy_t), (w//2, h//2), color, 1, cv2.LINE_AA)
            else:
                # Simple box for non-selected
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            # Track ID
            label = f"[{tid}] {track.class_name.upper()}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, 1)
            cv2.putText(frame, label, (x1 + 2, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Trajectory
            if len(track.trajectory) >= 2:
                points = np.array(list(track.trajectory), dtype=np.int32)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    pt_color = tuple(int(c * alpha + (255 if j == 1 else 0) * (1-alpha)) 
                                   for j, c in enumerate(color))
                    cv2.line(frame, tuple(points[i-1]), tuple(points[i]), pt_color, 1)
    
    def _draw_target_data_box(self, frame: np.ndarray, track: DroneTrack, 
                             x: int, y: int, w: int, h: int):
        """Draw detailed target data box."""
        color = self.lock_color
        box_w, box_h = 180, 100
        
        # Clamp position
        x = min(x, w - box_w - 10)
        y = max(y, 10)
        
        # Box background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 20, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 1)
        
        # Data lines
        lines = [
            f"TARGET: ID-{track.track_id}",
            f"RNG: {track.distance_from_center*100:.1f}%",
            f"SPD: {track.speed*100:.1f}% | DIR: {track.direction:.0f}°",
            f"THREAT: {track.threat_score:.2f} [{track.priority.name}]",
            f"CONF: {track.confidence:.2f}",
        ]
        
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x + 5, y + 18 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_data_strip(self, frame: np.ndarray, w: int, h: int, 
                        fps: float, night_vision: NightVisionMode):
        """Draw bottom data strip."""
        y = h - 35
        color = self.hud_color
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y - 5), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        
        # Separator line
        cv2.line(frame, (0, y - 5), (w, y - 5), color, 1)
        
        # Left side - System info
        info_left = f"SYS: ONLINE | FPS: {fps:.1f} | NV: {night_vision.value.upper()}"
        cv2.putText(frame, info_left, (10, y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Center - Mode
        mode_text = "MODE: TRACKING"
        (tw, _), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, mode_text, ((w - tw) // 2, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Right side - Time
        time_str = time.strftime("%H:%M:%S")
        (tw, _), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, time_str, (w - tw - 10, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_threat_warnings(self, frame: np.ndarray, tracks: Dict[int, DroneTrack],
                            w: int, h: int):
        """Draw threat warning indicators."""
        critical_count = sum(1 for t in tracks.values() if t.priority == TargetPriority.CRITICAL)
        high_count = sum(1 for t in tracks.values() if t.priority == TargetPriority.HIGH)
        
        if critical_count > 0 and self.blink_state:
            # Flashing warning
            warning = f"!!! THREAT DETECTED: {critical_count} CRITICAL !!!"
            (tw, th), _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x = (w - tw) // 2
            y = h // 2 - 100
            
            # Background flash
            cv2.rectangle(frame, (x - 10, y - th - 5), (x + tw + 10, y + 5), 
                         (0, 0, 255), -1)
            cv2.putText(frame, warning, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        elif high_count > 0 and self.frame_counter % 30 < 15:
            warning = f"CAUTION: {high_count} HIGH THREAT"
            (tw, th), _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x = (w - tw) // 2
            y = h // 2 - 100
            cv2.putText(frame, warning, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    def _draw_lock_indicator(self, frame: np.ndarray, cx: int, cy: int, w: int, h: int):
        """Draw target lock confirmation."""
        if self.blink_state:
            text = "<<<< LOCKED >>>>"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x = (w - tw) // 2
            y = cy + 80
            
            cv2.putText(frame, text, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.lock_color, 2)


# =============================================================================
# ROI SELECTOR
# =============================================================================

class ROISelector:
    """Handles Region of Interest selection."""
    
    def __init__(self):
        self.roi = None
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.active = False
    
    def start_selection(self, x: int, y: int):
        self.selecting = True
        self.start_point = (x, y)
        self.active = True
    
    def update_selection(self, x: int, y: int):
        if self.selecting:
            self.end_point = (x, y)
    
    def end_selection(self):
        if self.selecting and self.start_point and self.end_point:
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            self.roi = (x1, y1, x2, y2)
        self.selecting = False
    
    def clear(self):
        self.roi = None
        self.active = False
        self.selecting = False
        self.start_point = None
        self.end_point = None
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI on frame."""
        if self.selecting and self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "SELECTING ROI...", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        elif self.roi:
            x1, y1, x2, y2 = self.roi
            # Animated border
            dash_len = 10
            color = (0, 255, 255)
            
            # Draw dashed rectangle
            for i in range(x1, x2, dash_len * 2):
                cv2.line(frame, (i, y1), (min(i + dash_len, x2), y1), color, 2)
                cv2.line(frame, (i, y2), (min(i + dash_len, x2), y2), color, 2)
            for i in range(y1, y2, dash_len * 2):
                cv2.line(frame, (x1, i), (x1, min(i + dash_len, y2)), color, 2)
                cv2.line(frame, (x2, i), (x2, min(i + dash_len, y2)), color, 2)
            
            cv2.putText(frame, "ROI ACTIVE", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def is_in_roi(self, bbox: Tuple[int, ...]) -> bool:
        """Check if bounding box is within ROI."""
        if not self.roi:
            return True
        
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = self.roi
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return (rx1 <= center_x <= rx2) and (ry1 <= center_y <= ry2)


# =============================================================================
# SCREEN FITTER
# =============================================================================

class ScreenFitter:
    def __init__(self, fit_mode: str = "fit", target_width: int = None,
                 target_height: int = None, margin: int = 80):
        self.fit_mode = fit_mode
        self.margin = margin
        self.target_width = target_width
        self.target_height = target_height
        self.screen_width = 1920
        self.screen_height = 1080
        self.scale_factor = 1.0
        self._detect_screen()
    
    def _detect_screen(self):
        if SCREENINFO_AVAILABLE:
            try:
                monitors = get_monitors()
                if monitors:
                    primary = monitors[0]
                    self.screen_width = primary.width
                    self.screen_height = primary.height
                    return
            except:
                pass
        
        try:
            root = tk.Tk()
            self.screen_width = root.winfo_screenwidth()
            self.screen_height = root.winfo_screenheight()
            root.destroy()
            return
        except:
            pass
    
    def get_target_size(self, frame_width: int, frame_height: int) -> Tuple[int, int, float]:
        if self.fit_mode == "none":
            return frame_width, frame_height, 1.0
        
        max_w = self.target_width or (self.screen_width - self.margin * 2)
        max_h = self.target_height or (self.screen_height - self.margin * 2)
        
        frame_aspect = frame_width / frame_height
        screen_aspect = max_w / max_h
        
        if self.fit_mode == "fit":
            if frame_aspect > screen_aspect:
                target_w = max_w
                target_h = int(target_w / frame_aspect)
            else:
                target_h = max_h
                target_w = int(target_h * frame_aspect)
        elif self.fit_mode == "fill":
            if frame_aspect > screen_aspect:
                target_h = max_h
                target_w = int(target_h * frame_aspect)
            else:
                target_w = max_w
                target_h = int(target_w / frame_aspect)
        else:
            target_w, target_h = frame_width, frame_height
        
        self.scale_factor = target_w / frame_width
        return target_w, target_h, self.scale_factor
    
    def resize_frame(self, frame: np.ndarray, graph_width: int = 0) -> np.ndarray:
        if self.fit_mode == "none":
            return frame
        
        h, w = frame.shape[:2]
        target_w, target_h, _ = self.get_target_size(w + graph_width, h)
        
        # Adjust for graph panel
        if graph_width > 0:
            target_w = max(target_w - graph_width, w // 2)
        
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return resized


# =============================================================================
# DRONE TRACKER
# =============================================================================

class DroneTracker:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.drone_classes = ["drone", "uav", "quadcopter", "hexacopter", "aircraft", "airplane", "bird"]
        self.model = None
        self._init_model(model_path)
        
        self.tracks: Dict[int, DroneTrack] = {}
        self.next_track_id = 1
        self.frame_count = 0
        self.start_time = time.time()
        self.selected_target_id: Optional[int] = None
        self.fps_history = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
    
    def _init_model(self, model_path: str):
        if not ULTRALYTICS_AVAILABLE:
            return
        try:
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            try:
                self.model = YOLO("yolov8n.pt")
            except:
                pass
    
    def detect(self, frame: np.ndarray, roi_selector: ROISelector = None) -> List[Dict]:
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
                    
                    is_drone = any(drone_term in class_name.lower() for drone_term in self.drone_classes)
                    
                    if is_drone or confidence > 0.6:
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        # Filter by ROI
                        if roi_selector is None or roi_selector.is_in_roi(bbox):
                            detections.append({
                                'bbox': bbox,
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': class_name
                            })
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detections
    
    def _calculate_iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
        return inter / union if union > 0 else 0.0
    
    def _associate_detections(self, detections, frame_shape):
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
                pred_bbox = (int(pred_x - w/2), int(pred_y - h/2), int(pred_x + w/2), int(pred_y + h/2))
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
            self.tracks[tid].update(detections[j]['bbox'], detections[j]['confidence'],
                                   current_time, frame_shape)
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
        
        lost_ids = [tid for tid, track in self.tracks.items() 
                   if tid not in matched_tracks and track.is_lost(current_time)]
        for tid in lost_ids:
            del self.tracks[tid]
    
    def select_target(self, frame_shape, strategy="threat"):
        if not self.tracks:
            self.selected_target_id = None
            return
        
        for track in self.tracks.values():
            track.is_selected = False
        
        for track in self.tracks.values():
            track.calculate_threat_score(frame_shape)
        
        if strategy == "threat":
            best = max(self.tracks.values(), key=lambda t: t.threat_score)
        elif strategy == "closest":
            best = min(self.tracks.values(), key=lambda t: t.distance_from_center)
        elif strategy == "newest":
            best = max(self.tracks.values(), key=lambda t: t.last_seen)
        elif strategy == "fastest":
            best = max(self.tracks.values(), key=lambda t: t.speed)
        else:
            best = max(self.tracks.values(), key=lambda t: t.threat_score)
        
        best.is_selected = True
        self.selected_target_id = best.track_id
    
    def toggle_lock(self):
        """Toggle target lock on selected target."""
        if self.selected_target_id and self.selected_target_id in self.tracks:
            track = self.tracks[self.selected_target_id]
            track.is_locked = not track.is_locked
            if track.is_locked:
                track.lock_time = time.time()
            return track.is_locked
        return False
    
    def process_frame(self, frame: np.ndarray, roi_selector: ROISelector = None) -> np.ndarray:
        self.frame_count += 1
        frame_time = time.time()
        h, w = frame.shape[:2]
        
        t0 = time.time()
        detections = self.detect(frame, roi_selector)
        self.detection_times.append(time.time() - t0)
        
        if detections:
            self._associate_detections(detections, (h, w))
        
        self.select_target((h, w), strategy="threat")
        self.fps_history.append(1.0 / (time.time() - frame_time + 1e-6))
        
        return frame  # Visualization handled by HUD renderer


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fighter HUD Drone Tracker')
    parser.add_argument('--source', type=str, default='0', help='Video source')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    
    # Display options
    parser.add_argument('--night-vision', action='store_true', help='Enable night vision mode')
    parser.add_argument('--nv-mode', type=str, default='green', 
                       choices=['green', 'white_hot', 'black_hot', 'amber'],
                       help='Night vision mode')
    parser.add_argument('--fullscreen', action='store_true', help='Fullscreen mode')
    parser.add_argument('--fit-screen', action='store_true', help='Fit to screen')
    parser.add_argument('--show-graphs', action='store_true', help='Show path graphs')
    parser.add_argument('--graph-width', type=int, default=400, help='Graph panel width')
    parser.add_argument('--no-display', action='store_true', help='Headless mode')
    parser.add_argument('--save-data', type=str, default=None, help='Export JSON data')
    
    args = parser.parse_args()
    
    # Initialize components
    night_vision = NightVisionProcessor(
        mode=NightVisionMode(args.nv_mode) if args.night_vision else NightVisionMode.OFF
    )
    
    hud = FighterHUDRenderer()
    roi_selector = ROISelector()
    graph_gen = PathGraphGenerator(width=args.graph_width, height=300)
    
    fit_mode = "fit" if args.fit_screen else "none"
    screen_fitter = ScreenFitter(fit_mode=fit_mode)
    
    # Parse source
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Initialize tracker
    tracker = DroneTracker(model_path=args.model, confidence_threshold=args.conf)
    if tracker.model is None:
        print("Failed to initialize model. Exiting.")
        return
    
    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {orig_w}x{orig_h} @ {fps:.1f} FPS")
    
    # Window setup
    window_name = "Fighter HUD - Drone Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Mouse callback for ROI selection
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_selector.start_selection(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and roi_selector.selecting:
            roi_selector.update_selection(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_selector.end_selection()
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Video writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))
    
    # Main loop
    print("\n" + "="*60)
    print("FIGHTER HUD DRONE TRACKING SYSTEM")
    print("="*60)
    print("Controls:")
    print("  [Q] Quit    [P] Pause    [S] Save frame")
    print("  [T] Target strategy    [R] Reset tracks")
    print("  [N] Toggle night vision    [M] Cycle NV mode")
    print("  [L] Toggle target lock    [C] Clear ROI")
    print("  [Mouse] Drag to select ROI")
    print("="*60 + "\n")
    
    paused = False
    strategy_idx = 0
    strategies = ["threat", "closest", "newest", "fastest"]
    show_graphs = args.show_graphs
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of stream")
                    break
                
                # Apply night vision
                if night_vision.mode != NightVisionMode.OFF:
                    frame = night_vision.apply(frame)
                
                # Process tracking
                tracker.process_frame(frame, roi_selector)
                
                # Update HUD animation
                hud.update()
                
                # Draw HUD overlay
                avg_fps = np.mean(tracker.fps_history) if tracker.fps_history else 0
                frame = hud.draw_full_hud(frame, tracker.tracks, tracker.selected_target_id,
                                         avg_fps, night_vision.mode)
                
                # Draw ROI
                frame = roi_selector.draw(frame)
                
                # Write original resolution
                if writer:
                    writer.write(frame)
                
                # Prepare display
                if not args.no_display:
                    display_frame = frame.copy()
                    
                    # Add graph panel if enabled
                    if show_graphs and MATPLOTLIB_AVAILABLE:
                        graph_img = graph_gen.generate_path_graph(
                            tracker.tracks, tracker.selected_target_id, (orig_h, orig_w))
                        
                        if graph_img is not None:
                            # Resize graph to match frame height
                            graph_h = display_frame.shape[0]
                            graph_ratio = graph_h / graph_img.shape[0]
                            new_w = int(graph_img.shape[1] * graph_ratio)
                            graph_img = cv2.resize(graph_img, (new_w, graph_h))
                            
                            # Concatenate horizontally
                            display_frame = np.hstack([display_frame, graph_img])
                    
                    # Fit to screen
                    if args.fit_screen:
                        display_frame = screen_fitter.resize_frame(display_frame)
                    
                    cv2.imshow(window_name, display_frame)
            
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('s'):
                    fname = f"hud_capture_{int(time.time())}.jpg"
                    cv2.imwrite(fname, frame)
                    print(f"Saved: {fname}")
                elif key == ord('t'):
                    strategy_idx = (strategy_idx + 1) % len(strategies)
                    print(f"Strategy: {strategies[strategy_idx]}")
                elif key == ord('r'):
                    tracker.tracks.clear()
                    tracker.selected_target_id = None
                    print("Tracks reset")
                elif key == ord('n'):
                    if night_vision.mode == NightVisionMode.OFF:
                        night_vision.mode = NightVisionMode.GREEN
                        print("Night vision: ON")
                    else:
                        night_vision.mode = NightVisionMode.OFF
                        print("Night vision: OFF")
                elif key == ord('m'):
                    new_mode = night_vision.cycle_mode()
                    print(f"NV Mode: {new_mode.value}")
                elif key == ord('l'):
                    locked = tracker.toggle_lock()
                    print(f"Target lock: {'ON' if locked else 'OFF'}")
                elif key == ord('c'):
                    roi_selector.clear()
                    print("ROI cleared")
                elif key == ord('g'):
                    show_graphs = not show_graphs
                    print(f"Graphs: {'ON' if show_graphs else 'OFF'}")
                elif key == ord('f'):
                    # Toggle fullscreen
                    is_fs = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == 1
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                         cv2.WINDOW_NORMAL if is_fs else cv2.WINDOW_FULLSCREEN)
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        if args.save_data:
            tracker.export_tracking_data(args.save_data)
        
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print(f"Frames: {tracker.frame_count}")
        print(f"Tracks: {tracker.next_track_id - 1}")
        print("="*60)


if __name__ == "__main__":
    main()