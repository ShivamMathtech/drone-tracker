"""
3D Robotic Arm Simulator with OpenCV
=====================================
A complete course with 6 modules:
M1: 3D Arm Visualization (OpenCV 3D projection)
M2: Forward Kinematics (DH parameters)
M3: Inverse Kinematics (analytical + numerical)
M4: Gesture Control (MediaPipe/OpenCV)
M5: Object Tracking Control
M6: Full Simulation Environment
"""

import cv2
import numpy as np
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
from collections import deque


# ============================================================================
# MODULE 1: 3D MATH UTILITIES & CAMERA SYSTEM
# ============================================================================

@dataclass
class Vec3:
    """3D Vector with numpy backend."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, 1.0])
    
    def from_array(self, arr: np.ndarray):
        self.x, self.y, self.z = arr[0], arr[1], arr[2]
        return self
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vec3':
        l = self.length()
        if l < 1e-6:
            return Vec3(0, 0, 1)
        return Vec3(self.x/l, self.y/l, self.z/l)


class Camera3D:
    """
    3D Camera for projecting world coordinates to 2D screen.
    Supports orbit, pan, zoom controls.
    """
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        
        # Camera position (spherical coordinates)
        self.distance = 800.0
        self.azimuth = math.radians(45)    # Horizontal angle
        self.elevation = math.radians(30)  # Vertical angle
        
        # Look-at target
        self.target = Vec3(0, 200, 0)
        
        # Projection parameters
        self.fov = math.radians(60)
        self.near = 10.0
        self.far = 5000.0
        
        # Mouse control state
        self.rotating = False
        self.panning = False
        self.last_mouse = (0, 0)
        
    def get_position(self) -> Vec3:
        """Calculate camera position from spherical coords."""
        x = self.target.x + self.distance * math.cos(self.elevation) * math.sin(self.azimuth)
        y = self.target.y + self.distance * math.sin(self.elevation)
        z = self.target.z + self.distance * math.cos(self.elevation) * math.cos(self.azimuth)
        return Vec3(x, y, z)
    
    def get_view_matrix(self) -> np.ndarray:
        """Generate view matrix (look-at)."""
        eye = self.get_position()
        up = Vec3(0, 1, 0)
        
        # Forward vector (from eye to target)
        f = (self.target - eye).normalize()
        # Right vector
        s = Vec3(
            f.y * up.z - f.z * up.y,
            f.z * up.x - f.x * up.z,
            f.x * up.y - f.y * up.x
        ).normalize()
        # True up vector
        u = Vec3(
            s.y * f.z - s.z * f.y,
            s.z * f.x - s.x * f.z,
            s.x * f.y - s.y * f.x
        )
        
        # Build view matrix
        view = np.array([
            [s.x, s.y, s.z, -s.x * eye.x - s.y * eye.y - s.z * eye.z],
            [u.x, u.y, u.z, -u.x * eye.x - u.y * eye.y - u.z * eye.z],
            [-f.x, -f.y, -f.z, f.x * eye.x + f.y * eye.y + f.z * eye.z],
            [0, 0, 0, 1]
        ])
        return view
    
    def get_projection_matrix(self) -> np.ndarray:
        """Generate perspective projection matrix."""
        aspect = self.width / self.height
        f = 1.0 / math.tan(self.fov / 2)
        
        proj = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far + self.near) / (self.near - self.far), 
             (2 * self.far * self.near) / (self.near - self.far)],
            [0, 0, -1, 0]
        ])
        return proj
    
    def project(self, world_pos: Vec3) -> Tuple[int, int, float]:
        """
        Project 3D world position to 2D screen coordinates.
        Returns (screen_x, screen_y, depth) or None if behind camera.
        """
        # World -> View -> Clip -> NDC -> Screen
        view = self.get_view_matrix()
        proj = self.get_projection_matrix()
        
        world = world_pos.to_array()
        view_pos = view @ world
        clip_pos = proj @ view_pos
        
        # Perspective divide
        if abs(clip_pos[3]) < 1e-6:
            return None
        ndc_x = clip_pos[0] / clip_pos[3]
        ndc_y = clip_pos[1] / clip_pos[3]
        ndc_z = clip_pos[2] / clip_pos[3]
        
        # NDC to screen (OpenCV: y is down)
        screen_x = int((ndc_x + 1) * 0.5 * self.width)
        screen_y = int((1 - ndc_y) * 0.5 * self.height)
        
        return (screen_x, screen_y, ndc_z)
    
    def handle_mouse(self, event: int, x: int, y: int, flags: int):
        """Handle mouse events for camera control."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rotating = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.rotating = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.panning = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.panning = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom
            if flags > 0:
                self.distance = max(100, self.distance * 0.9)
            else:
                self.distance = min(2000, self.distance * 1.1)
        elif event == cv2.EVENT_MOUSEMOVE:
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            
            if self.rotating:
                self.azimuth -= dx * 0.01
                self.elevation = max(-math.pi/2 + 0.1, 
                                    min(math.pi/2 - 0.1, 
                                        self.elevation + dy * 0.01))
            elif self.panning:
                # Pan target
                right = math.cos(self.azimuth)
                forward = -math.sin(self.azimuth)
                self.target.x += (right * dx - forward * dy) * 0.5
                self.target.z += (forward * dx + right * dy) * 0.5
                
            self.last_mouse = (x, y)


# ============================================================================
# MODULE 2: ROBOTIC ARM KINEMATICS
# ============================================================================

@dataclass
class DHParameter:
    """Denavit-Hartenberg parameter for a joint."""
    theta: float   # Joint angle (variable for revolute)
    d: float       # Link offset
    a: float       # Link length
    alpha: float   # Link twist
    joint_type: str = 'revolute'  # 'revolute' or 'prismatic'
    
    def get_transform(self) -> np.ndarray:
        """Get transformation matrix for this joint."""
        ct = math.cos(self.theta)
        st = math.sin(self.theta)
        ca = math.cos(self.alpha)
        sa = math.sin(self.alpha)
        
        return np.array([
            [ct, -st * ca, st * sa, self.a * ct],
            [st, ct * ca, -ct * sa, self.a * st],
            [0, sa, ca, self.d],
            [0, 0, 0, 1]
        ])


class RoboticArm:
    """
    6-DOF Robotic Arm with DH parameters.
    Based on a generic industrial arm design.
    """
    def __init__(self):
        # DH parameters for each joint (theta, d, a, alpha)
        # These define a typical 6-DOF industrial arm
        self.dh_params = [
            DHParameter(0, 150, 0, math.pi/2),      # Base rotation
            DHParameter(0, 0, 250, 0),               # Shoulder
            DHParameter(0, 0, 200, math.pi/2),       # Elbow
            DHParameter(0, 180, 0, -math.pi/2),      # Wrist 1
            DHParameter(0, 0, 0, math.pi/2),         # Wrist 2
            DHParameter(0, 80, 0, 0),                # Wrist 3 (tool)
        ]
        
        # Joint limits (min, max) in radians
        self.joint_limits = [
            (-math.pi, math.pi),
            (-math.pi/2, math.pi/2),
            (-math.pi/2, math.pi/2),
            (-math.pi, math.pi),
            (-math.pi/2, math.pi/2),
            (-math.pi, math.pi),
        ]
        
        # Joint names
        self.joint_names = ['Base', 'Shoulder', 'Elbow', 
                           'Wrist1', 'Wrist2', 'Wrist3']
        
        # Current joint angles
        self.joint_angles = [0.0] * 6
        
        # Forward kinematics cache
        self.transforms = [np.eye(4)] * 7  # Including base
        
        # Visual properties
        self.link_color = (100, 150, 200)
        self.joint_color = (50, 200, 50)
        self.end_effector_color = (200, 50, 50)
        self.link_radius = 15
        
        # Target position for IK
        self.target_pos = Vec3(300, 400, 200)
        self.target_orientation = np.eye(3)
        
        # Trajectory
        self.trajectory = deque(maxlen=100)
        
    def set_joint_angles(self, angles: List[float]):
        """Set joint angles with limit checking."""
        for i, angle in enumerate(angles):
            if i < len(self.joint_limits):
                min_lim, max_lim = self.joint_limits[i]
                self.joint_angles[i] = max(min_lim, min(max_lim, angle))
            else:
                self.joint_angles[i] = angle
                
        self._update_dh()
        self._compute_forward_kinematics()
        
    def _update_dh(self):
        """Update DH parameters with current joint angles."""
        for i, angle in enumerate(self.joint_angles):
            self.dh_params[i].theta = angle
            
    def _compute_forward_kinematics(self):
        """Compute forward kinematics - all joint transforms."""
        self.transforms[0] = np.eye(4)  # Base
        
        for i, dh in enumerate(self.dh_params):
            T = dh.get_transform()
            self.transforms[i + 1] = self.transforms[i] @ T
            
        # Store end effector position
        ee = self.transforms[6]
        pos = Vec3(ee[0, 3], ee[1, 3], ee[2, 3])
        self.trajectory.append(pos)
        
    def get_joint_positions(self) -> List[Vec3]:
        """Get world positions of all joints."""
        positions = []
        for T in self.transforms:
            positions.append(Vec3(T[0, 3], T[1, 3], T[2, 3]))
        return positions
    
    def get_end_effector_pose(self) -> Tuple[Vec3, np.ndarray]:
        """Get end effector position and rotation matrix."""
        T = self.transforms[6]
        pos = Vec3(T[0, 3], T[1, 3], T[2, 3])
        rot = T[:3, :3]
        return pos, rot
    
    def inverse_kinematics_analytical(self, target: Vec3) -> Optional[List[float]]:
        """
        Analytical IK for a simplified 6-DOF arm.
        Returns joint angles or None if unreachable.
        """
        # Simplified analytical solution for demonstration
        # For a real arm, use numerical methods or specialized solvers
        
        x, y, z = target.x, target.y, target.z
        
        # Base rotation (project target to XY plane)
        theta1 = math.atan2(x, z)
        
        # Distance from base to target in XZ plane
        r = math.sqrt(x**2 + z**2)
        
        # Height difference
        h = y - self.dh_params[0].d
        
        # 2-link planar IK for shoulder and elbow
        L1 = self.dh_params[1].a  # Upper arm
        L2 = self.dh_params[2].a  # Forearm
        
        # Distance to wrist center (subtract wrist offsets)
        d = math.sqrt(r**2 + h**2)
        d = max(0.1, min(d, L1 + L2 - 0.1))  # Clamp to reachable
        
        # Law of cosines
        cos_theta3 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta3 = max(-1, min(1, cos_theta3))
        theta3 = math.acos(cos_theta3) - math.pi/2  # Elbow up
        
        # Shoulder angle
        alpha = math.atan2(h, r)
        beta = math.acos((L1**2 + d**2 - L2**2) / (2 * L1 * d))
        theta2 = alpha + beta - math.pi/2
        
        # Wrist angles (simplified - just point down)
        theta4 = 0
        theta5 = -theta2 - theta3
        theta6 = -theta1
        
        angles = [theta1, theta2, theta3, theta4, theta5, theta6]
        
        # Check limits
        for i, (angle, (min_lim, max_lim)) in enumerate(zip(angles, self.joint_limits)):
            if not (min_lim <= angle <= max_lim):
                return None
                
        return angles
    
    def inverse_kinematics_numerical(self, target: Vec3, 
                                      max_iter: int = 100,
                                      tolerance: float = 1.0) -> bool:
        """
        Numerical IK using Jacobian transpose method.
        More robust than analytical for complex arms.
        """
        dt = 0.5  # Step size
        
        for _ in range(max_iter):
            self._compute_forward_kinematics()
            ee_pos, _ = self.get_end_effector_pose()
            
            # Error
            error = target - ee_pos
            dist = error.length()
            
            if dist < tolerance:
                return True
                
            # Compute Jacobian numerically
            J = self._compute_jacobian()
            
            # Jacobian transpose method (simpler than pseudo-inverse)
            delta_pos = np.array([error.x, error.y, error.z])
            delta_theta = J.T @ delta_pos * dt
            
            # Update angles
            new_angles = [a + delta_theta[i] for i, a in enumerate(self.joint_angles)]
            self.set_joint_angles(new_angles)
            
        return False
    
    def _compute_jacobian(self) -> np.ndarray:
        """Compute 3x6 position Jacobian numerically."""
        J = np.zeros((3, 6))
        delta = 0.001
        
        current_pos, _ = self.get_end_effector_pose()
        
        for i in range(6):
            # Perturb joint i
            angles = self.joint_angles.copy()
            angles[i] += delta
            self.set_joint_angles(angles)
            new_pos, _ = self.get_end_effector_pose()
            
            # Restore
            self.set_joint_angles(self.joint_angles)
            
            # Column of Jacobian
            J[0, i] = (new_pos.x - current_pos.x) / delta
            J[1, i] = (new_pos.y - current_pos.y) / delta
            J[2, i] = (new_pos.z - current_pos.z) / delta
            
        return J


# ============================================================================
# MODULE 3: 3D RENDERER (OpenCV-based)
# ============================================================================

class Renderer3D:
    """
    3D Renderer using OpenCV - no OpenGL required!
    Projects 3D geometry to 2D and draws with OpenCV primitives.
    """
    def __init__(self, camera: Camera3D, width: int = 1280, height: int = 720):
        self.camera = camera
        self.width = width
        self.height = height
        
        # Z-buffer for simple occlusion
        self.z_buffer = None
        
        # Background
        self.bg_color = (30, 30, 40)
        
        # Grid
        self.grid_size = 1000
        self.grid_spacing = 50
        
    def create_frame(self) -> np.ndarray:
        """Create fresh frame with background."""
        frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        self.z_buffer = np.full((self.height, self.width), float('inf'))
        return frame
    
    def draw_grid(self, frame: np.ndarray):
        """Draw floor grid."""
        color = (60, 60, 70)
        
        # X lines
        for z in range(-self.grid_size, self.grid_size + 1, self.grid_spacing):
            p1 = self.camera.project(Vec3(-self.grid_size, 0, z))
            p2 = self.camera.project(Vec3(self.grid_size, 0, z))
            if p1 and p2:
                cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), color, 1)
                
        # Z lines
        for x in range(-self.grid_size, self.grid_size + 1, self.grid_spacing):
            p1 = self.camera.project(Vec3(x, 0, -self.grid_size))
            p2 = self.camera.project(Vec3(x, 0, self.grid_size))
            if p1 and p2:
                cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), color, 1)
                
    def draw_axes(self, frame: np.ndarray, origin: Vec3 = Vec3(0, 0, 0), 
                  scale: float = 100.0):
        """Draw coordinate axes."""
        axes = [
            (Vec3(scale, 0, 0), (0, 0, 255)),   # X - Red
            (Vec3(0, scale, 0), (0, 255, 0)),   # Y - Green
            (Vec3(0, 0, scale), (255, 0, 0)),   # Z - Blue
        ]
        
        o = self.camera.project(origin)
        if not o:
            return
            
        for dir_vec, color in axes:
            end = self.camera.project(origin + dir_vec)
            if end:
                cv2.line(frame, (o[0], o[1]), (end[0], end[1]), color, 2)
                cv2.putText(frame, 'XYZ'[axes.index((dir_vec, color))], 
                           (end[0] + 5, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
                           
    def draw_line_3d(self, frame: np.ndarray, p1: Vec3, p2: Vec3, 
                     color: Tuple[int, int, int], thickness: int = 2):
        """Draw 3D line with depth-based thickness."""
        proj1 = self.camera.project(p1)
        proj2 = self.camera.project(p2)
        
        if proj1 and proj2:
            # Depth-based alpha (darker = farther)
            avg_depth = (proj1[2] + proj2[2]) / 2
            alpha = max(0.3, min(1.0, 1.0 - (avg_depth + 1) / 2))
            faded = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, (proj1[0], proj1[1]), (proj2[0], proj2[1]), 
                    faded, thickness)
                    
    def draw_sphere(self, frame: np.ndarray, center: Vec3, radius: float,
                    color: Tuple[int, int, int], segments: int = 16):
        """Draw sphere as wireframe."""
        proj_center = self.camera.project(center)
        if not proj_center:
            return
            
        # Projected radius (approximate)
        edge = self.camera.project(center + Vec3(radius, 0, 0))
        if edge:
            r_2d = int(abs(edge[0] - proj_center[0]))
        else:
            r_2d = int(radius * 0.5)
            
        cv2.circle(frame, (proj_center[0], proj_center[1]), r_2d, color, 2)
        
        # Draw cross
        cv2.line(frame, 
                (proj_center[0] - r_2d, proj_center[1]),
                (proj_center[0] + r_2d, proj_center[1]), color, 1)
        cv2.line(frame,
                (proj_center[0], proj_center[1] - r_2d),
                (proj_center[0], proj_center[1] + r_2d), color, 1)
                
    def draw_cylinder(self, frame: np.ndarray, p1: Vec3, p2: Vec3, 
                      radius: float, color: Tuple[int, int, int]):
        """Draw cylinder (simplified as line with varying thickness)."""
        proj1 = self.camera.project(p1)
        proj2 = self.camera.project(p2)
        
        if not (proj1 and proj2):
            return
            
        # Thickness based on depth and radius
        avg_depth = (proj1[2] + proj2[2]) / 2
        thickness = max(2, int(radius * (1 - avg_depth) * 0.1))
        
        cv2.line(frame, (proj1[0], proj1[1]), (proj2[0], proj2[1]), 
                color, thickness)
                
    def draw_text_3d(self, frame: np.ndarray, pos: Vec3, text: str,
                     color: Tuple[int, int, int] = (255, 255, 255)):
        """Draw text at 3D position."""
        proj = self.camera.project(pos)
        if proj:
            cv2.putText(frame, text, (proj[0], proj[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ============================================================================
# MODULE 4: ARM VISUALIZER
# ============================================================================

class ArmVisualizer:
    """Visualizes the robotic arm in 3D."""
    
    def __init__(self, arm: RoboticArm, camera: Camera3D, renderer: Renderer3D):
        self.arm = arm
        self.camera = camera
        self.renderer = renderer
        
        # Visual settings
        self.show_trajectory = True
        self.show_joints = True
        self.show_coordinates = True
        self.show_target = True
        
    def render(self, frame: np.ndarray):
        """Render the complete arm scene."""
        # Draw environment
        self.renderer.draw_grid(frame)
        self.renderer.draw_axes(frame, Vec3(0, 0, 0), 150)
        
        # Get joint positions
        positions = self.arm.get_joint_positions()
        
        # Draw links (cylinders)
        for i in range(len(positions) - 1):
            self.renderer.draw_cylinder(
                frame, positions[i], positions[i + 1],
                self.arm.link_radius,
                self.arm.link_color
            )
            
        # Draw joints (spheres)
        if self.show_joints:
            for i, pos in enumerate(positions[:-1]):  # Exclude end effector
                self.renderer.draw_sphere(
                    frame, pos, 20, self.arm.joint_color
                )
                # Joint label
                self.renderer.draw_text_3d(
                    frame, pos + Vec3(0, 30, 0),
                    self.arm.joint_names[i],
                    (200, 200, 200)
                )
                
        # Draw end effector
        ee_pos = positions[-1]
        self.renderer.draw_sphere(
            frame, ee_pos, 15, self.arm.end_effector_color
        )
        
        # Draw target
        if self.show_target:
            self.renderer.draw_sphere(
                frame, self.arm.target_pos, 10, (255, 255, 0)
            )
            # Line from EE to target
            self.renderer.draw_line_3d(
                frame, ee_pos, self.arm.target_pos,
                (255, 255, 0), 1
            )
            
        # Draw trajectory
        if self.show_trajectory and len(self.arm.trajectory) > 1:
            traj_list = list(self.arm.trajectory)
            for i in range(len(traj_list) - 1):
                # Fade older points
                alpha = i / len(traj_list)
                color = (int(255 * alpha), int(100 * (1-alpha)), int(100))
                self.renderer.draw_line_3d(
                    frame, traj_list[i], traj_list[i+1], color, 2
                )
                
        # Draw HUD
        self._draw_hud(frame)
        
    def _draw_hud(self, frame: np.ndarray):
        """Draw heads-up display with arm info."""
        h, w = frame.shape[:2]
        
        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "6-DOF Robotic Arm Simulator", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Joint angles
        y = 70
        for i, (name, angle) in enumerate(zip(self.arm.joint_names, 
                                               self.arm.joint_angles)):
            deg = math.degrees(angle)
            bar = int((deg + 180) / 360 * 100)
            cv2.putText(frame, f"{name:10s}: {deg:7.2f}°", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            # Mini bar
            cv2.rectangle(frame, (180, y-10), (280, y-2), (50, 50, 50), -1)
            cv2.rectangle(frame, (180, y-10), (180 + bar, y-2), 
                         (0, 200, 0), -1)
            y += 25
            
        # End effector position
        ee_pos, _ = self.arm.get_end_effector_pose()
        cv2.putText(frame, f"EE Pos: ({ee_pos.x:.1f}, {ee_pos.y:.1f}, {ee_pos.z:.1f})", 
                   (20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Target position
        t = self.arm.target_pos
        cv2.putText(frame, f"Target: ({t.x:.1f}, {t.y:.1f}, {t.z:.1f})",
                   (20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Controls help
        help_y = h - 80
        cv2.putText(frame, "Controls:", (w - 300, help_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "LMB+Drag: Rotate | RMB+Drag: Pan | Scroll: Zoom", 
                   (w - 300, help_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "1-6: Joint control | T: Toggle traj | I: IK mode", 
                   (w - 300, help_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "G: Gesture mode | O: Object track | ESC: Exit", 
                   (w - 300, help_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


# ============================================================================
# MODULE 5: GESTURE CONTROL (MediaPipe)
# ============================================================================

class GestureController:
    """
    Control robotic arm using hand gestures via MediaPipe.
    Requires: pip install mediapipe
    """
    def __init__(self):
        self.enabled = False
        self.cap = None
        self.mp_hands = None
        self.hands = None
        
        # Gesture state
        self.hand_position = Vec3(0, 0, 0)  # Normalized 0-1
        self.gripper_open = True
        self.gesture_mode = 'position'  # 'position', 'joint', 'gripper'
        
        # Smoothing
        self.position_history = deque(maxlen=10)
        
    def initialize(self):
        """Initialize MediaPipe hands."""
        try:
            import mediapipe as mp
            self.mp_hands = mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.cap = cv2.VideoCapture(0)
            self.enabled = True
            print("Gesture control initialized")
            return True
        except ImportError:
            print("MediaPipe not installed. Run: pip install mediapipe")
            return False
            
    def process(self, arm: RoboticArm) -> Optional[np.ndarray]:
        """
        Process hand gesture and update arm.
        Returns camera frame for display or None.
        """
        if not self.enabled or self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Mirror frame
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        h, w = frame.shape[:2]
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = hand.landmark
            
            # Index finger tip (landmark 8)
            index_tip = landmarks[8]
            # Thumb tip (landmark 4)
            thumb_tip = landmarks[4]
            # Wrist (landmark 0)
            wrist = landmarks[0]
            
            # Map hand position to arm workspace
            # X: left-right, Y: up-down, Z: forward-back (using hand depth)
            x = (1 - index_tip.x) * 600 - 300  # -300 to 300
            y = (1 - index_tip.y) * 600 + 100  # 100 to 700
            z = 200 + (wrist.z + 0.5) * 200   # 100 to 300
            
            # Smooth
            self.position_history.append(Vec3(x, y, z))
            avg_pos = Vec3(0, 0, 0)
            for p in self.position_history:
                avg_pos = avg_pos + p
            avg_pos = avg_pos * (1.0 / len(self.position_history))
            
            # Update arm target
            if self.gesture_mode == 'position':
                arm.target_pos = avg_pos
                
            # Gripper from thumb-index distance
            thumb_idx_dist = math.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2
            )
            self.gripper_open = thumb_idx_dist > 0.1
            
            # Visual feedback
            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"Target: ({avg_pos.x:.0f}, {avg_pos.y:.0f}, {avg_pos.z:.0f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gripper: {'OPEN' if self.gripper_open else 'CLOSED'}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if self.gripper_open else (0, 0, 255), 2)
                       
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
        return frame
        
    def cleanup(self):
        if self.cap:
            self.cap.release()
        self.enabled = False


# ============================================================================
# MODULE 6: OBJECT TRACKING CONTROL
# ============================================================================

class ObjectTrackerController:
    """
    Control arm to track and follow colored objects.
    """
    def __init__(self):
        self.enabled = False
        self.cap = None
        
        # HSV color range for tracking (default: red)
        self.lower_hsv = np.array([0, 100, 100])
        self.upper_hsv = np.array([10, 255, 255])
        
        # Tracking state
        self.object_pos = None  # 3D position estimate
        self.tracking = False
        
    def initialize(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.enabled = True
        print("Object tracker initialized")
        return True
        
    def set_color_range(self, lower: np.ndarray, upper: np.ndarray):
        """Set HSV color range to track."""
        self.lower_hsv = lower
        self.upper_hsv = upper
        
    def process(self, arm: RoboticArm) -> Optional[np.ndarray]:
        """Process frame and update arm target to follow object."""
        if not self.enabled or self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        fh, fw = frame.shape[:2]
        
        if contours:
            # Largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area > 500:  # Minimum area threshold
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Map to 3D workspace
                    # x: based on horizontal position
                    # y: based on vertical position (lower = closer)
                    # z: based on contour area (larger = closer)
                    x = (cx / fw - 0.5) * 600
                    y = 700 - (cy / fh) * 500
                    z = 400 - min(area / 50, 300)
                    
                    self.object_pos = Vec3(x, y, z)
                    arm.target_pos = self.object_pos
                    self.tracking = True
                    
                    # Draw
                    cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Tracking: ({x:.0f}, {y:.0f}, {z:.0f})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.tracking = False
                cv2.putText(frame, "Object too small", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            self.tracking = False
            cv2.putText(frame, "No object found", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
        # Show mask
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([frame, mask_color])
        
        return combined
        
    def cleanup(self):
        if self.cap:
            self.cap.release()
        self.enabled = False


# ============================================================================
# MAIN SIMULATOR APPLICATION
# ============================================================================

class RobotArmSimulator:
    """
    Main application integrating all modules.
    """
    def __init__(self):
        # Core components
        self.camera = Camera3D(width=1280, height=720)
        self.arm = RoboticArm()
        self.renderer = Renderer3D(self.camera, 1280, 720)
        self.visualizer = ArmVisualizer(self.arm, self.camera, self.renderer)
        
        # Controllers
        self.gesture = GestureController()
        self.object_tracker = ObjectTrackerController()
        
        # State
        self.running = True
        self.mode = 'manual'  # 'manual', 'ik', 'gesture', 'object'
        self.selected_joint = 0
        self.joint_increment = math.radians(5)
        
        # Window
        self.window_name = "3D Robotic Arm Simulator"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        # Mouse callback
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
    def _mouse_callback(self, event, x, y, flags, param):
        self.camera.handle_mouse(event, x, y, flags)
        
    def handle_keys(self, key: int):
        """Handle keyboard input."""
        if key == 27:  # ESC
            self.running = False
            
        # Joint control (keys 1-6)
        elif ord('1') <= key <= ord('6'):
            self.selected_joint = key - ord('1')
            print(f"Selected joint: {self.arm.joint_names[self.selected_joint]}")
            
        # Joint angle adjustment
        elif key == ord('q'):
            angles = self.arm.joint_angles.copy()
            angles[self.selected_joint] += self.joint_increment
            self.arm.set_joint_angles(angles)
            self.mode = 'manual'
        elif key == ord('e'):
            angles = self.arm.joint_angles.copy()
            angles[self.selected_joint] -= self.joint_increment
            self.arm.set_joint_angles(angles)
            self.mode = 'manual'
            
        # IK mode
        elif key == ord('i'):
            self.mode = 'ik'
            print("Mode: Inverse Kinematics")
            # Solve IK
            success = self.arm.inverse_kinematics_numerical(self.arm.target_pos)
            print(f"IK {'succeeded' if success else 'failed'}")
            
        # Toggle trajectory
        elif key == ord('t'):
            self.visualizer.show_trajectory = not self.visualizer.show_trajectory
            
        # Reset
        elif key == ord('r'):
            self.arm.set_joint_angles([0] * 6)
            self.arm.target_pos = Vec3(300, 400, 200)
            self.mode = 'manual'
            
        # Gesture mode
        elif key == ord('g'):
            if not self.gesture.enabled:
                if self.gesture.initialize():
                    self.mode = 'gesture'
                    print("Mode: Gesture Control")
            else:
                self.gesture.cleanup()
                self.mode = 'manual'
                
        # Object tracking mode
        elif key == ord('o'):
            if not self.object_tracker.enabled:
                if self.object_tracker.initialize():
                    self.mode = 'object'
                    print("Mode: Object Tracking")
            else:
                self.object_tracker.cleanup()
                self.mode = 'manual'
                
        # Preset poses
        elif key == ord('h'):  # Home
            self.arm.set_joint_angles([0, 0, math.pi/4, 0, math.pi/4, 0])
            self.mode = 'manual'
        elif key == ord('p'):  # Pick
            self.arm.target_pos = Vec3(200, 200, 300)
            self.mode = 'ik'
            self.arm.inverse_kinematics_numerical(self.arm.target_pos)
            
    def run(self):
        """Main simulation loop."""
        print("\n" + "="*50)
        print("3D ROBOTIC ARM SIMULATOR")
        print("="*50)
        print("Controls:")
        print("  1-6    : Select joint")
        print("  Q/E    : Adjust joint angle")
        print("  I      : Inverse kinematics mode")
        print("  T      : Toggle trajectory")
        print("  R      : Reset arm")
        print("  H      : Home position")
        print("  P      : Pick position")
        print("  G      : Toggle gesture control")
        print("  O      : Toggle object tracking")
        print("  LMB    : Rotate camera")
        print("  RMB    : Pan camera")
        print("  Scroll : Zoom")
        print("  ESC    : Exit")
        print("="*50 + "\n")
        
        # Set initial pose
        self.arm.set_joint_angles([0, math.pi/6, -math.pi/4, 0, math.pi/3, 0])
        
        while self.running:
            loop_start = time.time()
            
            # Create frame
            frame = self.renderer.create_frame()
            
            # Update based on mode
            if self.mode == 'ik':
                # Continuously solve IK for target
                self.arm.inverse_kinematics_numerical(self.arm.target_pos, 
                                                       max_iter=5)
            elif self.mode == 'gesture':
                gesture_frame = self.gesture.process(self.arm)
                if gesture_frame is not None:
                    # Show gesture feed in corner
                    small = cv2.resize(gesture_frame, (320, 240))
                    frame[10:250, 10:330] = small
                    
            elif self.mode == 'object':
                track_frame = self.object_tracker.process(self.arm)
                if track_frame is not None:
                    small = cv2.resize(track_frame, (320, 240))
                    frame[10:250, 10:330] = small
                    
            # Render arm
            self.visualizer.render(frame)
            
            # Mode indicator
            mode_colors = {
                'manual': (200, 200, 200),
                'ik': (255, 255, 0),
                'gesture': (0, 255, 255),
                'object': (255, 0, 255)
            }
            cv2.putText(frame, f"MODE: {self.mode.upper()}", (20, 680),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_colors.get(self.mode, (255,255,255)), 2)
            
            # Selected joint indicator
            if self.mode == 'manual':
                cv2.putText(frame, f"Joint: {self.arm.joint_names[self.selected_joint]} [{self.selected_joint+1}]", 
                           (20, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show
            cv2.imshow(self.window_name, frame)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            self.handle_keys(key)
            
            # FPS cap
            elapsed = time.time() - loop_start
            if elapsed < 0.033:  # ~30 FPS
                time.sleep(0.033 - elapsed)
                
        # Cleanup
        self.gesture.cleanup()
        self.object_tracker.cleanup()
        cv2.destroyAllWindows()
        print("Simulator closed.")


# ============================================================================
# COURSE EXERCISES (Additional practice modules)
# ============================================================================

"""
EXERCISE 1: Forward Kinematics
------------------------------
Implement a function that calculates end-effector position given joint angles.
Verify by comparing with the built-in FK.

EXERCISE 2: Jacobian
--------------------
Implement numerical Jacobian calculation and use it for velocity control.
Move the end-effector in a straight line using Jacobian control.

EXERCISE 3: Path Planning
-------------------------
Implement linear interpolation between two poses.
Add circular arc motion for welding simulation.

EXERCISE 4: Collision Detection
-------------------------------
Add simple sphere-sphere collision between arm links and obstacles.
Visualize collision zones.

EXERCISE 5: Gripper Simulation
------------------------------
Add a parallel jaw gripper to the end effector.
Simulate grasping by detecting when object is between fingers.

EXERCISE 6: Camera Calibration
------------------------------
Add a virtual camera to the end effector.
Project a checkerboard pattern and perform calibration.

EXERCISE 7: Force Control
-------------------------
Simulate force/torque sensors at joints.
Implement compliance control for safe interaction.
"""


if __name__ == "__main__":
    simulator = RobotArmSimulator()
    simulator.run()