"""
Fighter HUD Drone Detection & Tracking System with 2D/3D Path Maps
===================================================================

Features:
- Night Vision Mode (green phosphor / thermal)
- Fighter Jet-style HUD Overlay
- Real-time 2D Path Maps (Top-down + Side view)
- Real-time 3D Path Visualization
- ROI Selection
- Target Lock System
- Normal Mode Toggle

Requirements:
    pip install ultralytics opencv-python numpy matplotlib screeninfo

Usage:
    python fighter_hud_tracker.py --source video.mp4 --night-vision --show-maps
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
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Maps disabled.")

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
    GREEN = "green"
    WHITE_HOT = "white_hot"
    BLACK_HOT = "black_hot"
    AMBER = "amber"


class DisplayMode(Enum):
    NORMAL = "normal"           # Standard view
    HUD = "hud"                 # Fighter jet HUD only
    MAP_2D = "map_2d"          # 2D path maps side panel
    MAP_3D = "map_3d"          # 3D path visualization
    SPLIT = "split"             # Video + 2D + 3D split view
    FULL_MAP = "full_map"       # Full screen maps


@dataclass
class DroneTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "drone"
    
    # 2D trajectory
    trajectory: deque = field(default_factory=lambda: deque(maxlen=500))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # 3D path (simulated altitude based on size/approach)
    path_3d: deque = field(default_factory=lambda: deque(maxlen=500))
    
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    direction: float = 0.0
    distance_from_center: float = 0.0
    
    # Estimated altitude (simulated from bounding box size changes)
    altitude: float = 100.0  # meters
    altitude_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    priority: TargetPriority = TargetPriority.NONE
    threat_score: float = 0.0
    is_selected: bool = False
    is_locked: bool = False
    lock_time: float = 0.0
    
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 0
    
    # Initial bbox size for altitude estimation
    initial_bbox_area: float = 0.0
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float,
               frame_time: float, frame_shape: Tuple[int, int]):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        bbox_area = (x2 - x1) * (y2 - y1)
        
        self.bbox = bbox
        self.center = (center_x, center_y)
        self.confidence = confidence
        self.last_seen = frame_time
        self.frame_count += 1
        
        # Store initial size for altitude estimation
        if self.initial_bbox_area == 0:
            self.initial_bbox_area = bbox_area
        
        # Estimate altitude from apparent size (larger = closer/lower)
        # Simulated: altitude inversely proportional to apparent size
        size_ratio = self.initial_bbox_area / max(bbox_area, 1)
        self.altitude = max(10, min(500, 100 * size_ratio))
        self.altitude_history.append(self.altitude)
        
        # 2D trajectory
        self.trajectory.append((center_x, center_y))
        self.timestamps.append(frame_time)
        
        # 3D path (x, y, z=altitude)
        self.path_3d.append((center_x, center_y, self.altitude))
        
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
        
        # Altitude factor (lower altitude = higher threat)
        altitude_factor = max(0, 1.0 - (self.altitude / 300))
        
        self.threat_score = (
            proximity * 0.30 +
            speed_factor * 0.20 +
            approach * 0.20 +
            self.confidence * 0.15 +
            altitude_factor * 0.15
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


# =============================================================================
# NIGHT VISION PROCESSOR
# =============================================================================

class NightVisionProcessor:
    def __init__(self, mode: NightVisionMode = NightVisionMode.GREEN):
        self.mode = mode
        self.noise_intensity = 15
        self.scanline_offset = 0
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == NightVisionMode.OFF:
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
        
        noise = np.random.normal(0, self.noise_intensity, gray.shape).astype(np.float32)
        gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        if self.mode == NightVisionMode.GREEN:
            result = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            result[:, :, 1] = gray
            result[:, :, 0] = (gray * 0.3).astype(np.uint8)
            result[:, :, 2] = (gray * 0.1).astype(np.uint8)
        elif self.mode == NightVisionMode.WHITE_HOT:
            result = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        elif self.mode == NightVisionMode.BLACK_HOT:
            result = cv2.applyColorMap(255 - gray, cv2.COLORMAP_HOT)
        elif self.mode == NightVisionMode.AMBER:
            result = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            result[:, :, 2] = gray
            result[:, :, 1] = (gray * 0.6).astype(np.uint8)
        
        result = self._add_scanlines(result)
        result = self._add_vignette(result)
        return result
    
    def _add_scanlines(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        for i in range(0, frame.shape[0], 2):
            result[i, :] = (result[i, :] * 0.85).astype(np.uint8)
        return result
    
    def _add_vignette(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        mask = 1.0 - (dist / max_dist) * 0.3
        mask = np.clip(mask, 0.7, 1.0)
        return (frame.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)
    
    def cycle_mode(self):
        modes = list(NightVisionMode)
        idx = modes.index(self.mode)
        self.mode = modes[(idx + 1) % len(modes)]
        return self.mode


# =============================================================================
# PATH MAP GENERATOR (2D & 3D)
# =============================================================================

class PathMapGenerator:
    """Generates 2D and 3D path visualization maps."""
    
    def __init__(self, width: int = 400, height: int = 350):
        self.width = width
        self.height = height
        self.fig_2d = None
        self.fig_3d = None
        self.canvas_2d = None
        self.canvas_3d = None
        
        if MATPLOTLIB_AVAILABLE:
            self._init_figures()
    
    def _init_figures(self):
        dpi = 100
        
        # 2D Figure (Top-down view)
        self.fig_2d = plt.figure(figsize=(self.width/dpi, self.height/dpi), dpi=dpi, facecolor='black')
        self.ax_2d = self.fig_2d.add_subplot(111, facecolor='black')
        self._style_axis(self.ax_2d)
        self.ax_2d.set_title('2D TOP-DOWN VIEW', color='lime', fontsize=9, fontweight='bold')
        self.canvas_2d = FigureCanvasAgg(self.fig_2d)
        
        # 3D Figure
        self.fig_3d = plt.figure(figsize=(self.width/dpi, self.height/dpi), dpi=dpi, facecolor='black')
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d', facecolor='black')
        self.ax_3d.set_facecolor('black')
        self.ax_3d.tick_params(colors='lime', labelsize=6)
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        self.ax_3d.xaxis.pane.set_edgecolor('lime')
        self.ax_3d.yaxis.pane.set_edgecolor('lime')
        self.ax_3d.zaxis.pane.set_edgecolor('lime')
        self.ax_3d.xaxis.pane.set_alpha(0.1)
        self.ax_3d.yaxis.pane.set_alpha(0.1)
        self.ax_3d.zaxis.pane.set_alpha(0.1)
        self.ax_3d.set_title('3D FLIGHT PATH', color='lime', fontsize=9, fontweight='bold')
        self.canvas_3d = FigureCanvasAgg(self.fig_3d)
    
    def _style_axis(self, ax):
        ax.set_facecolor('black')
        ax.tick_params(colors='lime', labelsize=7)
        ax.spines['bottom'].set_color('lime')
        ax.spines['top'].set_color('lime')
        ax.spines['left'].set_color('lime')
        ax.spines['right'].set_color('lime')
        ax.grid(True, color='darkgreen', alpha=0.3, linestyle='--')
    
    def generate_2d_map(self, tracks: Dict[int, DroneTrack], selected_id: Optional[int],
                       frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Generate 2D top-down path map."""
        if not MATPLOTLIB_AVAILABLE or not tracks:
            return None
        
        self.ax_2d.clear()
        self._style_axis(self.ax_2d)
        self.ax_2d.set_title('2D TOP-DOWN VIEW', color='lime', fontsize=9, fontweight='bold')
        
        h, w = frame_shape[:2]
        
        # Draw frame boundary
        self.ax_2d.plot([0, w, w, 0, 0], [0, 0, h, h, 0], 'g--', alpha=0.3, linewidth=1)
        self.ax_2d.fill([0, w, w, 0], [0, 0, h, h], color='darkgreen', alpha=0.05)
        
        # Draw center cross
        self.ax_2d.axhline(y=h/2, color='darkgreen', alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax_2d.axvline(x=w/2, color='darkgreen', alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Draw concentric zones
        for radius in [w/6, w/4, w/3]:
            circle = plt.Circle((w/2, h/2), radius, fill=False, color='darkgreen', 
                               alpha=0.2, linestyle='--')
            self.ax_2d.add_patch(circle)
        
        colors = {
            TargetPriority.CRITICAL: '#FF0000',
            TargetPriority.HIGH: '#FFA500',
            TargetPriority.MEDIUM: '#FFFF00',
            TargetPriority.LOW: '#00FF00',
            TargetPriority.NONE: '#808080',
        }
        
        for tid, track in tracks.items():
            if len(track.trajectory) < 2:
                continue
            
            x_vals = [p[0] for p in track.trajectory]
            y_vals = [h - p[1] for p in track.trajectory]  # Flip Y
            
            color = colors.get(track.priority, '#00FF00')
            linewidth = 3 if track.is_selected else 1.5
            alpha = 1.0 if track.is_selected else 0.6
            
            # Plot path with gradient
            for i in range(len(x_vals) - 1):
                path_alpha = alpha * (0.3 + 0.7 * (i / len(x_vals)))
                self.ax_2d.plot(x_vals[i:i+2], y_vals[i:i+2], 
                               color=color, linewidth=linewidth, alpha=path_alpha)
            
            # Current position
            if track.is_selected:
                self.ax_2d.scatter([x_vals[-1]], [y_vals[-1]], color='#FF00FF', 
                                  s=120, marker='X', linewidths=2, edgecolors='white')
                # Draw prediction line
                if track.speed > 0.01:
                    pred_x = x_vals[-1] + track.velocity[0] * 2
                    pred_y = y_vals[-1] - track.velocity[1] * 2
                    self.ax_2d.plot([x_vals[-1], pred_x], [y_vals[-1], pred_y],
                                   'm--', linewidth=2, alpha=0.7)
            else:
                self.ax_2d.scatter([x_vals[-1]], [y_vals[-1]], color=color, s=50, marker='o')
            
            # Start point
            self.ax_2d.scatter([x_vals[0]], [y_vals[0]], color=color, s=30, marker='^', alpha=0.5)
        
        self.ax_2d.set_xlim(-50, w + 50)
        self.ax_2d.set_ylim(-50, h + 50)
        self.ax_2d.set_xlabel('X (pixels)', color='lime', fontsize=7)
        self.ax_2d.set_ylabel('Y (pixels)', color='lime', fontsize=7)
        self.ax_2d.set_aspect('equal')
        
        self.fig_2d.tight_layout()
        self.canvas_2d.draw()
        
        buf = self.canvas_2d.buffer_rgba()
        img = np.asarray(buf)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    def generate_3d_map(self, tracks: Dict[int, DroneTrack], selected_id: Optional[int],
                       frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Generate 3D flight path visualization."""
        if not MATPLOTLIB_AVAILABLE or not tracks:
            return None
        
        self.ax_3d.clear()
        self.ax_3d.set_facecolor('black')
        self.ax_3d.tick_params(colors='lime', labelsize=6)
        self.ax_3d.set_title('3D FLIGHT PATH', color='lime', fontsize=9, fontweight='bold')
        
        h, w = frame_shape[:2]
        
        colors = {
            TargetPriority.CRITICAL: '#FF0000',
            TargetPriority.HIGH: '#FFA500',
            TargetPriority.MEDIUM: '#FFFF00',
            TargetPriority.LOW: '#00FF00',
            TargetPriority.NONE: '#808080',
        }
        
        # Draw ground plane grid
        xx, yy = np.meshgrid(np.linspace(0, w, 10), np.linspace(0, h, 10))
        zz = np.zeros_like(xx)
        self.ax_3d.plot_surface(xx, yy, zz, alpha=0.05, color='green')
        
        # Draw reference lines
        self.ax_3d.plot([0, w], [h/2, h/2], [0, 0], 'g--', alpha=0.2, linewidth=0.5)
        self.ax_3d.plot([w/2, w/2], [0, h], [0, 0], 'g--', alpha=0.2, linewidth=0.5)
        
        for tid, track in tracks.items():
            if len(track.path_3d) < 2:
                continue
            
            points = list(track.path_3d)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] for p in points]
            
            color = colors.get(track.priority, '#00FF00')
            linewidth = 3 if track.is_selected else 1.5
            alpha = 1.0 if track.is_selected else 0.6
            
            # 3D path line
            self.ax_3d.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=alpha)
            
            # Project path to ground
            self.ax_3d.plot(xs, ys, [0]*len(xs), color=color, linewidth=0.5, alpha=0.2)
            
            # Vertical lines to ground
            for i in range(0, len(xs), max(1, len(xs)//5)):
                self.ax_3d.plot([xs[i], xs[i]], [ys[i], ys[i]], [0, zs[i]], 
                               color=color, linewidth=0.5, alpha=0.3)
            
            # Current position
            if track.is_selected:
                self.ax_3d.scatter([xs[-1]], [ys[-1]], [zs[-1]], 
                                  color='#FF00FF', s=100, marker='X', 
                                  edgecolors='white', linewidths=2)
                # Altitude indicator
                self.ax_3d.text(xs[-1], ys[-1], zs[-1] + 20, 
                               f'{zs[-1]:.0f}m', color='magenta', fontsize=8)
            else:
                self.ax_3d.scatter([xs[-1]], [ys[-1]], [zs[-1]], 
                                  color=color, s=40, marker='o')
            
            # Start point
            self.ax_3d.scatter([xs[0]], [ys[0]], [zs[0]], 
                              color=color, s=30, marker='^', alpha=0.5)
        
        self.ax_3d.set_xlim(0, w)
        self.ax_3d.set_ylim(0, h)
        self.ax_3d.set_zlim(0, 500)
        self.ax_3d.set_xlabel('X', color='lime', fontsize=7)
        self.ax_3d.set_ylabel('Y', color='lime', fontsize=7)
        self.ax_3d.set_zlabel('ALT (m)', color='lime', fontsize=7)
        
        # Set view angle
        self.ax_3d.view_init(elev=25, azim=-60)
        
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
        
        buf = self.canvas_3d.buffer_rgba()
        img = np.asarray(buf)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    def generate_combined_map(self, tracks: Dict[int, DroneTrack], selected_id: Optional[int],
                             frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Generate combined 2D + 3D map panel."""
        map_2d = self.generate_2d_map(tracks, selected_id, frame_shape)
        map_3d = self.generate_3d_map(tracks, selected_id, frame_shape)
        
        if map_2d is None or map_3d is None:
            return None
        
        # Resize to same height
        h = max(map_2d.shape[0], map_3d.shape[0])
        map_2d = cv2.resize(map_2d, (int(map_2d.shape[1] * h / map_2d.shape[0]), h))
        map_3d = cv2.resize(map_3d, (int(map_3d.shape[1] * h / map_3d.shape[0]), h))
        
        # Stack vertically
        combined = np.vstack([map_2d, map_3d])
        return combined
    
    def generate_altitude_profile(self, tracks: Dict[int, DroneTrack],
                                  selected_id: Optional[int]) -> Optional[np.ndarray]:
        """Generate altitude over time graph."""
        if not MATPLOTLIB_AVAILABLE or not tracks:
            return None
        
        fig = plt.figure(figsize=(self.width/100, self.height/200), facecolor='black')
        ax = fig.add_subplot(111, facecolor='black')
        self._style_axis(ax)
        ax.set_title('ALTITUDE PROFILE', color='lime', fontsize=9, fontweight='bold')
        
        colors = {
            TargetPriority.CRITICAL: '#FF0000',
            TargetPriority.HIGH: '#FFA500',
            TargetPriority.MEDIUM: '#FFFF00',
            TargetPriority.LOW: '#00FF00',
            TargetPriority.NONE: '#808080',
        }
        
        for tid, track in tracks.items():
            if len(track.altitude_history) < 2:
                continue
            
            times = [t - track.first_seen for t in track.timestamps]
            alts = list(track.altitude_history)
            
            # Pad if needed
            while len(alts) < len(times):
                alts.append(alts[-1] if alts else 100)
            times = times[:len(alts)]
            
            color = colors.get(track.priority, '#00FF00')
            linewidth = 2.5 if track.is_selected else 1.2
            alpha = 1.0 if track.is_selected else 0.6
            
            ax.plot(times, alts, color=color, linewidth=linewidth, alpha=alpha,
                   label=f'ID-{tid}')
            
            if track.is_selected:
                ax.scatter([times[-1]], [alts[-1]], color='#FF00FF', s=80, marker='X')
                ax.axhline(y=alts[-1], color='magenta', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Time (s)', color='lime', fontsize=7)
        ax.set_ylabel('Altitude (m)', color='lime', fontsize=7)
        ax.set_ylim(0, 500)
        ax.grid(True, color='darkgreen', alpha=0.3, linestyle='--')
        fig.tight_layout()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close(fig)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


# =============================================================================
# FIGHTER HUD RENDERER
# =============================================================================

class FighterHUDRenderer:
    def __init__(self):
        self.hud_color = (0, 255, 0)
        self.hud_color_dim = (0, 180, 0)
        self.alert_color = (0, 0, 255)
        self.lock_color = (255, 0, 255)
        self.frame_counter = 0
        self.blink_state = True
    
    def update(self):
        self.frame_counter += 1
        if self.frame_counter % 15 == 0:
            self.blink_state = not self.blink_state
    
    def draw_full_hud(self, frame: np.ndarray, tracks: Dict[int, DroneTrack],
                     selected_id: Optional[int], fps: float,
                     night_vision: NightVisionMode, display_mode: DisplayMode) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cx, cy = w // 2, h // 2
        
        if display_mode in [DisplayMode.HUD, DisplayMode.SPLIT]:
            self._draw_pitch_ladder(overlay, cx, cy, w, h)
            self._draw_reticle(overlay, cx, cy, selected_id, tracks)
            self._draw_heading_tape(overlay, w, h)
            self._draw_side_bars(overlay, w, h, tracks, selected_id)
            self._draw_target_boxes(overlay, tracks, selected_id, w, h)
            self._draw_threat_warnings(overlay, tracks, w, h)
            
            if selected_id and selected_id in tracks and tracks[selected_id].is_locked:
                self._draw_lock_indicator(overlay, cx, cy, w, h)
        
        if display_mode in [DisplayMode.NORMAL, DisplayMode.HUD, DisplayMode.SPLIT]:
            self._draw_data_strip(overlay, w, h, fps, night_vision, display_mode, tracks)
        
        return overlay
    
    def draw_normal_overlay(self, frame: np.ndarray, tracks: Dict[int, DroneTrack],
                           selected_id: Optional[int], fps: float) -> np.ndarray:
        """Draw minimal overlay for normal mode."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Simple crosshair
        cx, cy = w // 2, h // 2
        cv2.line(overlay, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
        cv2.line(overlay, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
        
        # Draw tracks with simple boxes
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
                color = (255, 0, 255)
                cv2.rectangle(overlay, (x1-2, y1-2), (x2+2, y2+2), color, 2)
                
                # Target info
                info = f"ID:{tid} T:{track.threat_score:.2f} A:{track.altitude:.0f}m"
                cv2.putText(overlay, info, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
                cv2.putText(overlay, f"ID:{tid}", (x1, y1 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Trajectory
            if len(track.trajectory) >= 2:
                points = np.array(list(track.trajectory), dtype=np.int32)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    thickness = max(1, int(alpha * 2))
                    cv2.line(overlay, tuple(points[i-1]), tuple(points[i]), color, thickness)
        
        # Simple info bar
        cv2.putText(overlay, f"FPS:{fps:.1f} | Tracks:{len(tracks)} | [V]HUD [B]2D [N]3D [M]Split",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return overlay
    
    def _draw_pitch_ladder(self, frame, cx, cy, w, h):
        color = self.hud_color_dim
        cv2.line(frame, (cx - 150, cy), (cx - 50, cy), color, 1)
        cv2.line(frame, (cx + 50, cy), (cx + 150, cy), color, 1)
        for offset in [-80, -40, 40, 80]:
            y = cy + offset
            cv2.line(frame, (cx - 30, y), (cx + 30, y), color, 1)
            cv2.line(frame, (cx - 30, y), (cx - 20, y - 5), color, 1)
            cv2.line(frame, (cx + 30, y), (cx + 20, y - 5), color, 1)
    
    def _draw_reticle(self, frame, cx, cy, selected_id, tracks):
        color = self.lock_color if (selected_id and selected_id in tracks and 
                                    tracks[selected_id].is_locked) else self.hud_color
        cv2.circle(frame, (cx, cy), 40, color, 2)
        cv2.circle(frame, (cx, cy), 3, color, -1)
        cv2.line(frame, (cx - 60, cy), (cx - 45, cy), color, 2)
        cv2.line(frame, (cx + 45, cy), (cx + 60, cy), color, 2)
        cv2.line(frame, (cx, cy - 60), (cx, cy - 45), color, 2)
        cv2.line(frame, (cx, cy + 45), (cx, cy + 60), color, 2)
        
        bracket = 25
        cv2.line(frame, (cx - 50, cy - 50), (cx - 50 + bracket, cy - 50), color, 2)
        cv2.line(frame, (cx - 50, cy - 50), (cx - 50, cy - 50 + bracket), color, 2)
        cv2.line(frame, (cx + 50, cy - 50), (cx + 50 - bracket, cy - 50), color, 2)
        cv2.line(frame, (cx + 50, cy - 50), (cx + 50, cy - 50 + bracket), color, 2)
        cv2.line(frame, (cx - 50, cy + 50), (cx - 50 + bracket, cy + 50), color, 2)
        cv2.line(frame, (cx - 50, cy + 50), (cx - 50, cy + 50 - bracket), color, 2)
        cv2.line(frame, (cx + 50, cy + 50), (cx + 50 - bracket, cy + 50), color, 2)
        cv2.line(frame, (cx + 50, cy + 50), (cx + 50, cy + 50 - bracket), color, 2)
    
    def _draw_heading_tape(self, frame, w, h):
        y = 40
        color = self.hud_color
        cv2.rectangle(frame, (w//2 - 200, y - 20), (w//2 + 200, y + 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (w//2 - 200, y - 20), (w//2 + 200, y + 20), color, 1)
        cv2.line(frame, (w//2, y - 15), (w//2, y + 15), (255, 255, 255), 2)
        cv2.putText(frame, "N", (w//2 - 7, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for i in range(-4, 5):
            if i == 0: continue
            x = w//2 + i * 40
            cv2.line(frame, (x, y - 10), (x, y + 10), color, 1)
            heading = (i * 10) % 360
            cv2.putText(frame, str(heading), (x - 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def _draw_side_bars(self, frame, w, h, tracks, selected_id):
        color = self.hud_color_dim
        bar_x = 30
        cv2.line(frame, (bar_x, 100), (bar_x, h - 100), color, 1)
        for i in range(10):
            y = 100 + i * (h - 200) // 10
            cv2.line(frame, (bar_x, y), (bar_x + 10, y), color, 1)
        
        bar_x = w - 30
        cv2.line(frame, (bar_x, 100), (bar_x, h - 100), color, 1)
        for i in range(10):
            y = 100 + i * (h - 200) // 10
            cv2.line(frame, (bar_x - 10, y), (bar_x, y), color, 1)
        
        if selected_id and selected_id in tracks:
            speed = tracks[selected_id].speed * 100
            y_pos = h - 100 - int(speed * (h - 200) / 100)
            y_pos = max(100, min(h - 100, y_pos))
            cv2.circle(frame, (w - 30, y_pos), 5, self.lock_color, -1)
    
    def _draw_target_boxes(self, frame, tracks, selected_id, w, h):
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
                bracket_len = 35 + (5 if self.blink_state else 0)
                thickness = 3
                
                cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, thickness)
                cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, thickness)
                cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, thickness)
                cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, thickness)
                cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, thickness)
                cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, thickness)
                cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, thickness)
                cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, thickness)
                
                diamond_size = 10
                cx_t, cy_t = int(track.center[0]), int(track.center[1])
                diamond = np.array([
                    [cx_t, cy_t - diamond_size],
                    [cx_t + diamond_size, cy_t],
                    [cx_t, cy_t + diamond_size],
                    [cx_t - diamond_size, cy_t]
                ], np.int32)
                cv2.polylines(frame, [diamond], True, self.lock_color, 2)
                
                self._draw_target_data_box(frame, track, x2 + 10, y1, w, h)
                cv2.line(frame, (cx_t, cy_t), (w//2, h//2), color, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            label = f"[{tid}] {track.class_name.upper()}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, 1)
            cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            if len(track.trajectory) >= 2:
                points = np.array(list(track.trajectory), dtype=np.int32)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    pt_color = tuple(int(c * alpha + (255 if j == 1 else 0) * (1-alpha)) 
                                   for j, c in enumerate(color))
                    cv2.line(frame, tuple(points[i-1]), tuple(points[i]), pt_color, 1)
    
    def _draw_target_data_box(self, frame, track, x, y, w, h):
        color = self.lock_color
        box_w, box_h = 200, 120
        x = min(x, w - box_w - 10)
        y = max(y, 10)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 20, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 1)
        
        lines = [
            f"TARGET: ID-{track.track_id}",
            f"RNG: {track.distance_from_center*100:.1f}% | ALT: {track.altitude:.0f}m",
            f"SPD: {track.speed*100:.1f}% | DIR: {track.direction:.0f}deg",
            f"THREAT: {track.threat_score:.2f} [{track.priority.name}]",
            f"CONF: {track.confidence:.2f}",
        ]
        
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x + 5, y + 18 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_data_strip(self, frame, w, h, fps, night_vision, display_mode, tracks):
        y = h - 35
        color = self.hud_color
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y - 5), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        cv2.line(frame, (0, y - 5), (w, y - 5), color, 1)
        
        mode_str = display_mode.value.upper()
        nv_str = night_vision.value.upper() if night_vision != NightVisionMode.OFF else "OFF"
        info_left = f"SYS:ONLINE | FPS:{fps:.1f} | MODE:{mode_str} | NV:{nv_str}"
        cv2.putText(frame, info_left, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        track_info = f"TRACKS: {len(tracks)}"
        (tw, _), _ = cv2.getTextSize(track_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, track_info, ((w - tw) // 2, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        time_str = time.strftime("%H:%M:%S")
        (tw, _), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, time_str, (w - tw - 10, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_threat_warnings(self, frame, tracks, w, h):
        critical_count = sum(1 for t in tracks.values() if t.priority == TargetPriority.CRITICAL)
        high_count = sum(1 for t in tracks.values() if t.priority == TargetPriority.HIGH)
        
        if critical_count > 0 and self.blink_state:
            warning = f"!!! THREAT DETECTED: {critical_count} CRITICAL !!!"
            (tw, th), _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x = (w - tw) // 2
            y = h // 2 - 100
            cv2.rectangle(frame, (x - 10, y - th - 5), (x + tw + 10, y + 5), (0, 0, 255), -1)
            cv2.putText(frame, warning, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif high_count > 0 and self.frame_counter % 30 < 15:
            warning = f"CAUTION: {high_count} HIGH THREAT"
            (tw, th), _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x = (w - tw) // 2
            y = h // 2 - 100
            cv2.putText(frame, warning, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    def _draw_lock_indicator(self, frame, cx, cy, w, h):
        if self.blink_state:
            text = "<<<< LOCKED >>>>"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x = (w - tw) // 2
            y = cy + 80
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.lock_color, 2)


# =============================================================================
# ROI SELECTOR
# =============================================================================

class ROISelector:
    def __init__(self):
        self.roi = None
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.active = False
    
    def start_selection(self, x, y):
        self.selecting = True
        self.start_point = (x, y)
        self.active = True
    
    def update_selection(self, x, y):
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
    
    def draw(self, frame):
        if self.selecting and self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "SELECTING ROI...", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        elif self.roi:
            x1, y1, x2, y2 = self.roi
            dash_len = 10
            color = (0, 255, 255)
            for i in range(x1, x2, dash_len * 2):
                cv2.line(frame, (i, y1), (min(i + dash_len, x2), y1), color, 2)
                cv2.line(frame, (i, y2), (min(i + dash_len, x2), y2), color, 2)
            for i in range(y1, y2, dash_len * 2):
                cv2.line(frame, (x1, i), (x1, min(i + dash_len, y2)), color, 2)
                cv2.line(frame, (x2, i), (x2, min(i + dash_len, y2)), color, 2)
            cv2.putText(frame, "ROI ACTIVE", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
    
    def is_in_roi(self, bbox):
        if not self.roi:
            return True
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = self.roi
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (rx1 <= center_x <= rx2) and (ry1 <= center_y <= ry2)


# =============================================================================
# DRONE TRACKER
# =============================================================================

class DroneTracker:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.drone_classes = ["drone", "uav", "quadcopter", "hexacopter", "aircraft", "airplane", "bird"]
        self.model = None
        self._init_model(model_path)
        self.tracks = {}
        self.next_track_id = 1
        self.frame_count = 0
        self.start_time = time.time()
        self.selected_target_id = None
        self.fps_history = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
    
    def _init_model(self, model_path):
        if not ULTRALYTICS_AVAILABLE:
            return
        try:
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model: {model_path}")
        except Exception as e:
            print(f"Error: {e}")
            try:
                self.model = YOLO("yolov8n.pt")
            except:
                pass
    
    def detect(self, frame, roi_selector=None):
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
                        if roi_selector is None or roi_selector.is_in_roi(bbox):
                            detections.append({
                                'bbox': bbox, 'confidence': confidence,
                                'class_id': class_id, 'class_name': class_name
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
        if self.selected_target_id and self.selected_target_id in self.tracks:
            track = self.tracks[self.selected_target_id]
            track.is_locked = not track.is_locked
            if track.is_locked:
                track.lock_time = time.time()
            return track.is_locked
        return False
    
    def process_frame(self, frame, roi_selector=None):
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
        return frame


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fighter HUD Drone Tracker with 2D/3D Maps')
    parser.add_argument('--source', type=str, default='0', help='Video source')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    
    # Display modes
    parser.add_argument('--mode', type=str, default='hud',
                       choices=['normal', 'hud', 'map_2d', 'map_3d', 'split', 'full_map'],
                       help='Display mode')
    parser.add_argument('--night-vision', action='store_true', help='Enable night vision')
    parser.add_argument('--nv-mode', type=str, default='green',
                       choices=['green', 'white_hot', 'black_hot', 'amber'],
                       help='Night vision mode')
    parser.add_argument('--fullscreen', action='store_true', help='Fullscreen mode')
    parser.add_argument('--fit-screen', action='store_true', help='Fit to screen')
    parser.add_argument('--map-width', type=int, default=450, help='Map panel width')
    parser.add_argument('--no-display', action='store_true', help='Headless mode')
    parser.add_argument('--save-data', type=str, default=None, help='Export JSON data')
    
    args = parser.parse_args()
    
    # Initialize components
    night_vision = NightVisionProcessor(
        mode=NightVisionMode(args.nv_mode) if args.night_vision else NightVisionMode.OFF
    )
    
    hud = FighterHUDRenderer()
    roi_selector = ROISelector()
    map_gen = PathMapGenerator(width=args.map_width, height=350)
    
    display_mode = DisplayMode(args.mode)
    
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
    
    # Mouse callback for ROI
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
    print("\n" + "="*65)
    print("FIGHTER HUD DRONE TRACKING SYSTEM - 2D/3D PATH MAPS")
    print("="*65)
    print("CONTROLS:")
    print("  [Q] Quit        [P] Pause         [S] Save frame")
    print("  [T] Strategy    [R] Reset tracks  [L] Toggle lock")
    print("  [N] Night vision  [M] Cycle NV mode  [C] Clear ROI")
    print("  [V] Normal mode   [B] HUD mode       [2] 2D Map")
    print("  [3] 3D Map        [4] Split view     [5] Full maps")
    print("  [Mouse] Drag to select ROI")
    print("="*65 + "\n")
    
    paused = False
    strategy_idx = 0
    strategies = ["threat", "closest", "newest", "fastest"]
    
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
                hud.update()
                
                # Apply overlay based on display mode
                avg_fps = np.mean(tracker.fps_history) if tracker.fps_history else 0
                
                if display_mode == DisplayMode.NORMAL:
                    frame = hud.draw_normal_overlay(frame, tracker.tracks, 
                                                   tracker.selected_target_id, avg_fps)
                else:
                    frame = hud.draw_full_hud(frame, tracker.tracks, tracker.selected_target_id,
                                             avg_fps, night_vision.mode, display_mode)
                
                # Draw ROI
                frame = roi_selector.draw(frame)
                
                # Write original
                if writer:
                    writer.write(frame)
                
                # Prepare display
                if not args.no_display:
                    display_frame = frame.copy()
                    
                    # Add map panels based on mode
                    if display_mode in [DisplayMode.MAP_2D, DisplayMode.SPLIT, DisplayMode.FULL_MAP]:
                        if MATPLOTLIB_AVAILABLE:
                            map_2d = map_gen.generate_2d_map(
                                tracker.tracks, tracker.selected_target_id, (orig_h, orig_w))
                            if map_2d is not None:
                                # Resize to match frame height
                                map_h = display_frame.shape[0]
                                map_ratio = map_h / map_2d.shape[0]
                                new_w = int(map_2d.shape[1] * map_ratio)
                                map_2d = cv2.resize(map_2d, (new_w, map_h))
                                display_frame = np.hstack([display_frame, map_2d])
                    
                    if display_mode in [DisplayMode.MAP_3D, DisplayMode.SPLIT, DisplayMode.FULL_MAP]:
                        if MATPLOTLIB_AVAILABLE:
                            if display_mode == DisplayMode.MAP_3D:
                                # Replace video with 3D map
                                map_3d = map_gen.generate_3d_map(
                                    tracker.tracks, tracker.selected_target_id, (orig_h, orig_w))
                                if map_3d is not None:
                                    display_frame = map_3d
                            elif display_mode == DisplayMode.SPLIT:
                                # Add 3D map below or beside
                                map_3d = map_gen.generate_3d_map(
                                    tracker.tracks, tracker.selected_target_id, (orig_h, orig_w))
                                if map_3d is not None:
                                    # Resize to match width
                                    target_w = display_frame.shape[1]
                                    map_ratio = target_w / map_3d.shape[1]
                                    new_h = int(map_3d.shape[0] * map_ratio)
                                    map_3d = cv2.resize(map_3d, (target_w, new_h))
                                    display_frame = np.vstack([display_frame, map_3d])
                            elif display_mode == DisplayMode.FULL_MAP:
                                # Combined 2D + 3D + altitude
                                map_combined = map_gen.generate_combined_map(
                                    tracker.tracks, tracker.selected_target_id, (orig_h, orig_w))
                                alt_profile = map_gen.generate_altitude_profile(
                                    tracker.tracks, tracker.selected_target_id)
                                
                                if map_combined is not None:
                                    display_frame = map_combined
                                if alt_profile is not None:
                                    # Stack altitude below
                                    target_w = display_frame.shape[1]
                                    alt_ratio = target_w / alt_profile.shape[1]
                                    new_h = int(alt_profile.shape[0] * alt_ratio)
                                    alt_profile = cv2.resize(alt_profile, (target_w, new_h))
                                    display_frame = np.vstack([display_frame, alt_profile])
                    
                    # Fit to screen
                    if args.fit_screen:
                        screen_h = 1080  # approximate
                        if display_frame.shape[0] > screen_h - 100:
                            ratio = (screen_h - 100) / display_frame.shape[0]
                            new_w = int(display_frame.shape[1] * ratio)
                            display_frame = cv2.resize(display_frame, (new_w, screen_h - 100))
                    
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
                    cv2.imwrite(fname, display_frame)
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
                elif key == ord('v'):
                    display_mode = DisplayMode.NORMAL
                    print("Mode: NORMAL")
                elif key == ord('b'):
                    display_mode = DisplayMode.HUD
                    print("Mode: HUD")
                elif key == ord('2'):
                    display_mode = DisplayMode.MAP_2D
                    print("Mode: 2D MAP")
                elif key == ord('3'):
                    display_mode = DisplayMode.MAP_3D
                    print("Mode: 3D MAP")
                elif key == ord('4'):
                    display_mode = DisplayMode.SPLIT
                    print("Mode: SPLIT VIEW")
                elif key == ord('5'):
                    display_mode = DisplayMode.FULL_MAP
                    print("Mode: FULL MAPS")
                elif key == ord('f'):
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
        
        print("\n" + "="*65)
        print("SESSION SUMMARY")
        print(f"Frames: {tracker.frame_count}")
        print(f"Tracks: {tracker.next_track_id - 1}")
        print("="*65)


if __name__ == "__main__":
    main()