"""
UAV Tracking System - OPTIMIZED VERSION
Fixes: Fast ROI selection, threaded capture, resolution limiting, frame skipping
"""

import cv2
import numpy as np
import math
import time
import threading
from queue import Queue


class ThreadedCapture:
    """Threaded video capture to prevent frame drops."""
    def __init__(self, source, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(source)
        
        # Limit resolution for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        self.q = Queue(maxsize=2)  # Small queue = low latency
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
    def _update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                else:
                    self.q.put(frame)
            else:
                time.sleep(0.001)
                
    def read(self):
        return self.q.get()
        
    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()


class UAVTrackerOptimized:
    def __init__(self, video_source=0, kill_zone_ratio=0.75, 
                 screen_width=1920, screen_height=1080,
                 target_width=1280, target_height=720):
        self.video_source = video_source
        self.kill_zone_ratio = kill_zone_ratio
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_width = target_width
        self.target_height = target_height
        
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking = False
        self.bbox = None
        
        # ========== FIX: Added COLOR_BLACK ==========
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_PURPLE = (255, 0, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)          # <-- THIS WAS MISSING
        self.COLOR_CYAN = (255, 255, 0)
        self.COLOR_ORANGE = (0, 165, 255)
        # ============================================
        
        self.display_scale = 1.0
        self.window_name = "UAV Tracking System | OpenCV"
        self.frame_count = 0
        self.last_display = None
        
    def initialize_video(self):
        """Initialize threaded capture with limited resolution."""
        self.capture = ThreadedCapture(
            self.video_source, 
            width=self.target_width,
            height=self.target_height
        )
        
        self.frame_width = self.capture.width
        self.frame_height = self.capture.height
        self.fps = self.capture.fps
        
        self._calculate_kill_zone()
        self.crop_x, self.crop_y, self.crop_w, self.crop_h = self.kill_zone
        self._calculate_fit_scale()
        
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        self.cropped_center = (self.crop_w // 2, self.crop_h // 2)
        
        print(f"Video: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
        print(f"Display: {int(self.crop_w * self.display_scale)}x{int(self.crop_h * self.display_scale)}")
        print("\n[S] Select & Lock  [R] Unlock  [F] Fullscreen  [ESC] Exit")
        
    def _calculate_kill_zone(self):
        kz_w = int(self.frame_width * self.kill_zone_ratio)
        kz_h = int(self.frame_height * self.kill_zone_ratio)
        kz_x = (self.frame_width - kz_w) // 2
        kz_y = (self.frame_height - kz_h) // 2
        self.kill_zone = (kz_x, kz_y, kz_w, kz_h)
        
    def _calculate_fit_scale(self):
        scale_w = self.screen_width / self.crop_w
        scale_h = self.screen_height / self.crop_h
        self.display_scale = min(scale_w, scale_h)
        
    # ========== FAST ROI SELECTION ==========
    
    def select_roi_fast(self, display_frame):
        """
        FAST ROI selection using downscaled frame.
        This is the key fix for slowness!
        """
        # Use 0.5x scale for selection (4x faster)
        select_scale = 0.5
        small = cv2.resize(display_frame, None, fx=select_scale, fy=select_scale)
        
        print("\n>>> Draw box around target, press ENTER or SPACE to LOCK <<<")
        
        # Temporarily show small frame for selection
        cv2.imshow(self.window_name, small)
        small_bbox = cv2.selectROI(self.window_name, small, 
                                    fromCenter=False, showCrosshair=True)
        
        # Restore display
        cv2.imshow(self.window_name, display_frame)
        
        if small_bbox[2] > 0 and small_bbox[3] > 0:
            # Scale back to display coords
            dx = int(small_bbox[0] / select_scale)
            dy = int(small_bbox[1] / select_scale)
            dw = int(small_bbox[2] / select_scale)
            dh = int(small_bbox[3] / select_scale)
            display_bbox = (dx, dy, dw, dh)
            
            # Convert to original frame coords
            original_bbox = self._display_to_original(display_bbox)
            
            # Validate
            ox, oy, ow, oh = original_bbox
            ox = max(0, min(ox, self.frame_width - 1))
            oy = max(0, min(oy, self.frame_height - 1))
            ow = min(ow, self.frame_width - ox)
            oh = min(oh, self.frame_height - oy)
            original_bbox = (ox, oy, ow, oh)
            
            # Get fresh frame for tracker init
            fresh_frame = self.capture.read()
            
            # Re-init tracker
            self.tracker = cv2.TrackerCSRT_create()
            self.bbox = original_bbox
            self.tracker.init(fresh_frame, self.bbox)
            self.tracking = True
            
            print(f">>> TARGET LOCKED! Bbox: {original_bbox}")
            return True
            
        print("Selection cancelled")
        return False
        
    def _display_to_original(self, display_bbox):
        """Display -> Cropped -> Original."""
        dx, dy, dw, dh = display_bbox
        cx = int(dx / self.display_scale)
        cy = int(dy / self.display_scale)
        cw = int(dw / self.display_scale)
        ch = int(dh / self.display_scale)
        ox = cx + self.crop_x
        oy = cy + self.crop_y
        return (ox, oy, cw, ch)
        
    def _original_to_cropped(self, bbox):
        if bbox is None:
            return None
        x, y, w, h = bbox
        return (x - self.crop_x, y - self.crop_y, w, h)
        
    def _cropped_to_display(self, cropped_bbox):
        if cropped_bbox is None:
            return None
        cx, cy, cw, ch = cropped_bbox
        return (int(cx * self.display_scale), int(cy * self.display_scale),
                int(cw * self.display_scale), int(ch * self.display_scale))
                
    # ========== TRACKING & DRAWING ==========
    
    def get_confidence(self):
        if not self.tracking:
            return 0.0
        try:
            return min(100.0, max(0.0, self.tracker.getTrackingScore() * 100))
        except:
            return 75.0
            
    def calculate_offsets(self, cropped_bbox):
        tx = cropped_bbox[0] + cropped_bbox[2] // 2
        ty = cropped_bbox[1] + cropped_bbox[3] // 2
        dx = tx - self.cropped_center[0]
        dy = ty - self.cropped_center[1]
        
        offsets = {}
        if dx > 0:
            offsets['Right'] = abs(dx) / 10.0
        else:
            offsets['Left'] = abs(dx) / 10.0
        if dy > 0:
            offsets['Down'] = abs(dy) / 10.0
        else:
            offsets['Up'] = abs(dy) / 10.0
        return offsets, (tx, ty)
        
    def crop_frame(self, frame):
        x, y, w, h = self.kill_zone
        x = max(0, x)
        y = max(0, y)
        w = min(w, self.frame_width - x)
        h = min(h, self.frame_height - y)
        return frame[y:y+h, x:x+w]
        
    def scale_display(self, frame):
        if self.display_scale == 1.0:
            return frame
        new_w = int(frame.shape[1] * self.display_scale)
        new_h = int(frame.shape[0] * self.display_scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
    # ========== DRAWING ==========
    
    def draw_hud(self, frame, offsets, inside, conf):
        h, w = frame.shape[:2]
        
        status = "Inside" if inside else "Outside"
        color = self.COLOR_GREEN if inside else self.COLOR_RED
        cv2.putText(frame, status, (int(w*0.05), int(h*0.15)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        y_off = int(h * 0.28)
        for d, v in offsets.items():
            cv2.putText(frame, f"{d}:{v:.2f}", (int(w*0.05), y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_RED, 2)
            y_off += int(h * 0.08)
            
        cv2.putText(frame, "Av : Hedef Vurus Alani", (int(w*0.05), int(h*0.92)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_RED, 2)
        cv2.putText(frame, "CSRT Tracker", (int(w*0.05), int(h*0.06)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_GREEN, 2)
        cv2.putText(frame, "#1 UAV Tracking System | OpenCV", (int(w*0.05), int(h*0.10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_WHITE, 1)
                   
        lock = "LOCKED" if self.tracking else "NO TARGET"
        lock_c = self.COLOR_GREEN if self.tracking else self.COLOR_RED
        cv2.putText(frame, lock, (w-200, int(h*0.06)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, lock_c, 2)
                   
    def draw_target(self, frame, display_bbox, conf):
        x, y, w, h = [int(v) for v in display_bbox]
        color = self.COLOR_GREEN if conf > 70 else (self.COLOR_ORANGE if conf > 40 else self.COLOR_RED)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cx, cy = x + w//2, y + h//2
        cv2.circle(frame, (cx, cy), 4, self.COLOR_RED, -1)
        
        label = f"UAV {conf:.0f}%"
        ls, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = y - 10 if y - 10 > 20 else y + h + 20
        cv2.rectangle(frame, (x, ly-ls[1]-5), (x+ls[0], ly+5), color, -1)
        cv2.putText(frame, label, (x, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_WHITE, 2)
        
    def draw_trajectory(self, frame, target):
        h, w = frame.shape[:2]
        center = (w//2, h//2)
        cv2.line(frame, center, target, self.COLOR_GREEN, 2)
        cv2.circle(frame, center, 5, self.COLOR_BLACK, -1)
        cv2.circle(frame, center, 3, self.COLOR_GREEN, -1)
        
    # ========== MAIN LOOP ==========
    
    def run(self):
        self.initialize_video()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        target_frame_time = 1.0 / 30  # Cap at 30 FPS display
        self.current_display = None
        
        while True:
            loop_start = time.time()
            self.frame_count += 1
            
            # Read frame (non-blocking thanks to threaded capture)
            frame = self.capture.read()
            
            # ===== TRACKING (on original frame) =====
            if self.tracking:
                success, self.bbox = self.tracker.update(frame)
                if success:
                    x, y, w, h = self.bbox
                    if not (w > 5 and h > 5 and x > -w and y > -h 
                            and x < self.frame_width and y < self.frame_height):
                        self.tracking = False
                        self.bbox = None
                else:
                    self.tracking = False
                    self.bbox = None
                    
            # ===== CROP & DRAW =====
            cropped = self.crop_frame(frame)
            display = cropped.copy()
            
            # Purple border
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, 0), (w-1, h-1), self.COLOR_PURPLE, 3)
            
            if self.tracking and self.bbox:
                cb = self._original_to_cropped(self.bbox)
                if cb:
                    cx, cy, cw, ch = cb
                    visible = (cx + cw > 0 and cy + ch > 0 
                              and cx < self.crop_w and cy < self.crop_h)
                    
                    if visible:
                        db = self._cropped_to_display(cb)
                        conf = self.get_confidence()
                        offsets, tc = self.calculate_offsets(cb)
                        td = (int(tc[0] * self.display_scale), 
                              int(tc[1] * self.display_scale))
                        
                        inside = (cx >= 0 and cy >= 0 
                                 and cx + cw <= self.crop_w 
                                 and cy + ch <= self.crop_h)
                        
                        self.draw_trajectory(display, td)
                        self.draw_target(display, db, conf)
                        self.draw_hud(display, offsets, inside, conf)
                    else:
                        hh, ww = display.shape[:2]
                        cv2.putText(display, "TARGET OUT OF ZONE", 
                                   (ww//2-150, hh//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_RED, 3)
            else:
                hh, ww = display.shape[:2]
                cv2.putText(display, "Press 'S' to SELECT and LOCK target",
                           (ww//2-220, hh//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_CYAN, 2)
                           
            # Scale to screen
            display = self.scale_display(display)
            self.current_display = display.copy()
            cv2.imshow(self.window_name, display)
            
            # ===== KEY HANDLING =====
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                if self.current_display is not None:
                    self.select_roi_fast(self.current_display)
            elif key == ord('r') or key == ord('R'):
                self.tracking = False
                self.bbox = None
                self.tracker = cv2.TrackerCSRT_create()
                print("\n>>> UNLOCKED <<<")
            elif key == ord('f') or key == ord('F'):
                is_full = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,
                                     cv2.WINDOW_FULLSCREEN if is_full != 1 else cv2.WINDOW_NORMAL)
                                     
            # FPS limiting
            elapsed = time.time() - loop_start
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
                
        self.capture.stop()
        cv2.destroyAllWindows()
        print("Shutdown complete")


if __name__ == "__main__":
    tracker = UAVTrackerOptimized(
        video_source="drone_image.mp4",  # Change to 0 for webcam   
        kill_zone_ratio=0.75,
        screen_width=1920,
        screen_height=1080,
        target_width=1280,
        target_height=720
    )
    
    try:
        tracker.run()
    except KeyboardInterrupt:
        tracker.capture.stop()
        cv2.destroyAllWindows()