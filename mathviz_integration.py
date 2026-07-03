
"""
MathViz - Comprehensive Mathematical Integration Visualization Library
=====================================================================
A complete toolkit for visualizing 2D and 3D mathematical integrations
with multiple views, projections, and mathematical annotations.

Features:
- 2D Integration: Area under curves, between curves, with Riemann sums
- 3D Integration: Volume under surfaces, triple integrals
- Multiple Views: Standard, contour, wireframe, filled, projection views
- Projections: XY, XZ, YZ plane projections with proper mathematical notation
- Mathematical Annotations: Proper LaTeX-style labels, grid lines, axis labels

Dependencies: numpy, matplotlib, mpl_toolkits.mplot3d, scipy (optional for advanced)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings('ignore')

# Set default style for professional mathematical visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 120


class MathViz2D:
    """
    2D Integration Visualization Toolkit
    ====================================
    Visualize single and double integrals with various methods and views.
    """

    def __init__(self, figsize=(14, 10)):
        self.figsize = figsize
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'fill': '#2E86AB33',
            'fill_secondary': '#A23B7233',
            'grid': '#CCCCCC',
            'text': '#333333'
        }

    def integrate_1d(self, f, a, b, method='riemann', n=50, 
                     title=None, show_grid=True, show_legend=True,
                     fill_alpha=0.3, line_width=2):
        """
        Visualize definite integral of f(x) from a to b.

        Parameters:
        -----------
        f : callable
            Function to integrate: f(x)
        a, b : float
            Integration bounds
        method : str
            Visualization method: 'riemann', 'trapezoid', 'simpson', 'filled'
        n : int
            Number of subdivisions
        title : str
            Plot title (auto-generated if None)
        show_grid : bool
            Show grid lines
        show_legend : bool
            Show legend
        fill_alpha : float
            Opacity of filled region
        line_width : float
            Width of function curve

        Returns:
        --------
        fig, ax : matplotlib objects
        """
        x = np.linspace(a, b, 1000)
        y = f(x)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot function curve
        ax.plot(x, y, color=self.colors['primary'], linewidth=line_width, 
                label=r'$f(x)$', zorder=3)

        # Plot integration methods
        if method == 'riemann':
            self._plot_riemann(ax, f, a, b, n, fill_alpha)
        elif method == 'trapezoid':
            self._plot_trapezoid(ax, f, a, b, n, fill_alpha)
        elif method == 'simpson':
            self._plot_simpson(ax, f, a, b, n, fill_alpha)
        elif method == 'filled':
            self._plot_filled(ax, f, a, b, fill_alpha)

        # Mathematical annotations
        ax.axhline(y=0, color='black', linewidth=0.8, zorder=1)
        ax.axvline(x=0, color='black', linewidth=0.8, zorder=1)

        # Bounds annotations
        ax.axvline(x=a, color=self.colors['accent'], linestyle='--', alpha=0.7, zorder=2)
        ax.axvline(x=b, color=self.colors['accent'], linestyle='--', alpha=0.7, zorder=2)
        ax.plot([a, a], [0, f(a)], 'k--', alpha=0.5, zorder=2)
        ax.plot([b, b], [0, f(b)], 'k--', alpha=0.5, zorder=2)

        # Labels at bounds
        ax.annotate(rf'$a={a}$', xy=(a, 0), xytext=(a, -0.1*(max(y)-min(y))),
                   ha='center', fontsize=11, color=self.colors['accent'])
        ax.annotate(rf'$b={b}$', xy=(b, 0), xytext=(b, -0.1*(max(y)-min(y))),
                   ha='center', fontsize=11, color=self.colors['accent'])

        # Integral notation
        if title is None:
            title = r'$\int_{' + str(a) + '}^{' + str(b) + '} f(x)\,dx$'
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel(r'$x$', fontsize=13)
        ax.set_ylabel(r'$f(x)$', fontsize=13)

        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='-', color=self.colors['grid'])
        if show_legend:
            ax.legend(loc='best', fontsize=11, framealpha=0.9)

        ax.set_xlim(min(a-0.1*(b-a), min(x)), max(b+0.1*(b-a), max(x)))
        y_min, y_max = min(y), max(y)
        ax.set_ylim(y_min - 0.15*(y_max-y_min), y_max + 0.15*(y_max-y_min))

        plt.tight_layout()
        return fig, ax

    def _plot_riemann(self, ax, f, a, b, n, alpha):
        """Plot Riemann sum rectangles."""
        dx = (b - a) / n
        x_rect = np.linspace(a, b, n+1)

        # Left Riemann
        for i in range(n):
            height = f(x_rect[i])
            rect = plt.Rectangle((x_rect[i], 0), dx, height, 
                                fill=True, facecolor=self.colors['primary'], 
                                alpha=alpha*0.5, edgecolor=self.colors['primary'],
                                linewidth=0.5, zorder=2)
            ax.add_patch(rect)

        # Right Riemann (offset for comparison)
        for i in range(n):
            height = f(x_rect[i+1])
            rect = plt.Rectangle((x_rect[i], 0), dx, height, 
                                fill=True, facecolor=self.colors['secondary'], 
                                alpha=alpha*0.5, edgecolor=self.colors['secondary'],
                                linewidth=0.5, linestyle='--', zorder=2)
            ax.add_patch(rect)

        # Add legend entries using proxy artists
        from matplotlib.patches import Patch
        left_patch = Patch(color=self.colors['primary'], alpha=alpha*0.5, 
                          label=f'Left Riemann ($n={n}$)')
        right_patch = Patch(color=self.colors['secondary'], alpha=alpha*0.5, 
                             label=f'Right Riemann ($n={n}$)')
        ax.legend(handles=[left_patch, right_patch], loc='best', fontsize=11, framealpha=0.9)

    def _plot_trapezoid(self, ax, f, a, b, n, alpha):
        """Plot trapezoidal rule."""
        x_trap = np.linspace(a, b, n+1)
        y_trap = f(x_trap)

        for i in range(n):
            verts = [(x_trap[i], 0), (x_trap[i], y_trap[i]), 
                    (x_trap[i+1], y_trap[i+1]), (x_trap[i+1], 0)]
            poly = plt.Polygon(verts, closed=True, facecolor=self.colors['primary'],
                             alpha=alpha, edgecolor=self.colors['primary'],
                             linewidth=0.5, zorder=2)
            ax.add_patch(poly)

        ax.plot(x_trap, y_trap, 'o-', color=self.colors['accent'], 
               markersize=4, linewidth=1, label=f'Trapezoid Rule ($n={n}$)', zorder=3)

    def _plot_simpson(self, ax, f, a, b, n, alpha):
        """Plot Simpson's rule parabolic approximations."""
        if n % 2 != 0:
            n += 1
        x_simp = np.linspace(a, b, n+1)
        y_simp = f(x_simp)

        for i in range(0, n, 2):
            x_seg = np.linspace(x_simp[i], x_simp[i+2], 50)
            # Quadratic interpolation through 3 points
            coeffs = np.polyfit([x_simp[i], x_simp[i+1], x_simp[i+2]], 
                               [y_simp[i], y_simp[i+1], y_simp[i+2]], 2)
            y_seg = np.polyval(coeffs, x_seg)
            ax.fill_between(x_seg, 0, y_seg, alpha=alpha*0.7, 
                          color=self.colors['primary'], zorder=2)
            ax.plot(x_seg, y_seg, '--', color=self.colors['accent'], 
                   linewidth=1, zorder=3)

        ax.plot(x_simp, y_simp, 'o', color=self.colors['accent'], 
               markersize=5, label=f"Simpson's Rule ($n={n}$)", zorder=4)

    def _plot_filled(self, ax, f, a, b, alpha):
        """Simple filled area under curve."""
        x = np.linspace(a, b, 1000)
        y = f(x)
        ax.fill_between(x, 0, y, alpha=alpha, color=self.colors['primary'], 
                       label=r'$\int f(x)\,dx$', zorder=2)

    def integrate_2d_region(self, f, x_bounds, y_bounds, 
                            title=None, cmap='viridis', n=100,
                            show_contours=True, show_projection=True,
                            projection_plane='z'):
        """
        Visualize double integral over a 2D region with projections.

        Parameters:
        -----------
        f : callable
            Function f(x, y) to integrate
        x_bounds : tuple (xmin, xmax)
            X integration bounds
        y_bounds : tuple of callables or tuple of floats
            Y bounds: either (ymin_func, ymax_func) or (ymin, ymax) constants
        title : str
            Plot title
        cmap : str
            Colormap for surface
        n : int
            Grid resolution
        show_contours : bool
            Show contour lines
        show_projection : bool
            Show projection onto a plane
        projection_plane : str
            Projection plane: 'z' (xy), 'x' (yz), 'y' (xz)

        Returns:
        --------
        fig, axes : matplotlib objects
        """
        xmin, xmax = x_bounds
        x = np.linspace(xmin, xmax, n)

        # Handle y bounds - check if tuple of functions or constants
        if isinstance(y_bounds, tuple) and len(y_bounds) == 2:
            if callable(y_bounds[0]) and callable(y_bounds[1]):
                ymin_func, ymax_func = y_bounds
            else:
                # Constant bounds
                ymin_func = lambda x: y_bounds[0]
                ymax_func = lambda x: y_bounds[1]
        else:
            raise ValueError("y_bounds must be a tuple of (ymin, ymax) or (ymin_func, ymax_func)")

        # Create grid
        y_min_global = min(ymin_func(x))
        y_max_global = max(ymax_func(x))
        X, Y = np.meshgrid(x, np.linspace(y_min_global, y_max_global, n))
        Z = f(X, Y)

        # Mask region outside bounds
        mask = np.zeros_like(X, dtype=bool)
        for i in range(n):
            for j in range(n):
                if Y[j, i] < ymin_func(X[j, i]) or Y[j, i] > ymax_func(X[j, i]):
                    mask[j, i] = True
        Z_masked = np.ma.array(Z, mask=mask)

        # Create figure with multiple views
        fig = plt.figure(figsize=(18, 12))

        # Main 3D view
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        surf = ax1.plot_surface(X, Y, Z_masked, cmap=cmap, alpha=0.8,
                               linewidth=0, antialiased=True, zorder=2)
        ax1.set_title('3D Surface View', fontsize=12, pad=10)
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_zlabel(r'$f(x,y)$')

        # Contour view (top-down)
        ax2 = fig.add_subplot(2, 3, 2)
        contour = ax2.contourf(X, Y, Z_masked, levels=20, cmap=cmap, alpha=0.8)
        if show_contours:
            ax2.contour(X, Y, Z_masked, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        ax2.set_title('Contour View (Top-Down)', fontsize=12)
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        plt.colorbar(contour, ax=ax2, shrink=0.8)

        # Filled region view
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.pcolormesh(X, Y, Z_masked, cmap=cmap, shading='auto', alpha=0.8)
        # Draw boundary
        x_boundary = np.linspace(xmin, xmax, 200)
        y_min_boundary = ymin_func(x_boundary)
        y_max_boundary = ymax_func(x_boundary)
        ax3.fill_between(x_boundary, y_min_boundary, y_max_boundary, 
                        alpha=0.1, color='red', label='Integration Region')
        ax3.plot(x_boundary, y_min_boundary, 'r--', linewidth=2, label=r'$y_{min}(x)$')
        ax3.plot(x_boundary, y_max_boundary, 'r--', linewidth=2, label=r'$y_{max}(x)$')
        ax3.set_title('Integration Region', fontsize=12)
        ax3.set_xlabel(r'$x$')
        ax3.set_ylabel(r'$y$')
        ax3.legend(loc='best')
        ax3.set_aspect('equal', adjustable='box')

        # XY Projection (looking down z-axis)
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        ax4.plot_surface(X, Y, Z_masked, cmap=cmap, alpha=0.3, zorder=1)
        # Project onto z=0 plane
        z_proj = np.min(Z_masked) - 0.1*(np.max(Z_masked)-np.min(Z_masked))
        ax4.contourf(X, Y, Z_masked, zdir='z', offset=z_proj, 
                    cmap=cmap, levels=20, alpha=0.6, zorder=3)
        ax4.set_title('XY Projection (onto z=const)', fontsize=12, pad=10)
        ax4.set_xlabel(r'$x$')
        ax4.set_ylabel(r'$y$')
        ax4.set_zlabel(r'$f(x,y)$')

        # XZ Projection (looking from y-axis)
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        ax5.plot_surface(X, Y, Z_masked, cmap=cmap, alpha=0.3, zorder=1)
        y_proj = np.max(Y) + 0.1*(np.max(Y)-np.min(Y))
        ax5.contourf(X, Y, Z_masked, zdir='y', offset=y_proj,
                    cmap=cmap, levels=20, alpha=0.6, zorder=3)
        ax5.set_title('XZ Projection (onto y=const)', fontsize=12, pad=10)
        ax5.set_xlabel(r'$x$')
        ax5.set_ylabel(r'$y$')
        ax5.set_zlabel(r'$f(x,y)$')

        # YZ Projection (looking from x-axis)
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.plot_surface(X, Y, Z_masked, cmap=cmap, alpha=0.3, zorder=1)
        x_proj = np.min(X) - 0.1*(np.max(X)-np.min(X))
        ax6.contourf(X, Y, Z_masked, zdir='x', offset=x_proj,
                    cmap=cmap, levels=20, alpha=0.6, zorder=3)
        ax6.set_title('YZ Projection (onto x=const)', fontsize=12, pad=10)
        ax6.set_xlabel(r'$x$')
        ax6.set_ylabel(r'$y$')
        ax6.set_zlabel(r'$f(x,y)$')

        if title:
            fig.suptitle(title, fontsize=16, y=0.98)

        plt.tight_layout()
        return fig, [ax1, ax2, ax3, ax4, ax5, ax6]

    def compare_methods(self, f, a, b, n_values=[10, 50, 100, 500]):
        """
        Compare different integration methods side by side.

        Parameters:
        -----------
        f : callable
            Function to integrate
        a, b : float
            Bounds
        n_values : list
            Different subdivision counts to compare

        Returns:
        --------
        fig, axes : matplotlib objects
        """
        fig, axes = plt.subplots(2, len(n_values), figsize=(5*len(n_values), 10))
        if len(n_values) == 1:
            axes = axes.reshape(2, 1)

        x = np.linspace(a, b, 1000)
        y = f(x)

        for idx, n in enumerate(n_values):
            # Left Riemann
            ax_top = axes[0, idx]
            ax_top.plot(x, y, color=self.colors['primary'], linewidth=2, label=r'$f(x)$')
            dx = (b - a) / n
            x_rect = np.linspace(a, b, n+1)
            for i in range(n):
                height = f(x_rect[i])
                rect = plt.Rectangle((x_rect[i], 0), dx, height,
                                   fill=True, facecolor=self.colors['primary'],
                                   alpha=0.3, edgecolor=self.colors['primary'],
                                   linewidth=0.5)
                ax_top.add_patch(rect)
            ax_top.set_title(f'Left Riemann ($n={n}$)', fontsize=11)
            ax_top.set_xlabel(r'$x$')
            ax_top.set_ylabel(r'$f(x)$')
            ax_top.set_xlim(a, b)
            ax_top.grid(True, alpha=0.3)

            # Trapezoid
            ax_bot = axes[1, idx]
            ax_bot.plot(x, y, color=self.colors['primary'], linewidth=2, label=r'$f(x)$')
            x_trap = np.linspace(a, b, n+1)
            y_trap = f(x_trap)
            for i in range(n):
                verts = [(x_trap[i], 0), (x_trap[i], y_trap[i]),
                        (x_trap[i+1], y_trap[i+1]), (x_trap[i+1], 0)]
                poly = plt.Polygon(verts, closed=True, facecolor=self.colors['secondary'],
                                 alpha=0.3, edgecolor=self.colors['secondary'],
                                 linewidth=0.5)
                ax_bot.add_patch(poly)
            ax_bot.plot(x_trap, y_trap, 'o-', color=self.colors['accent'],
                       markersize=3, linewidth=1)
            ax_bot.set_title(f'Trapezoid Rule ($n={n}$)', fontsize=11)
            ax_bot.set_xlabel(r'$x$')
            ax_bot.set_ylabel(r'$f(x)$')
            ax_bot.set_xlim(a, b)
            ax_bot.grid(True, alpha=0.3)

        fig.suptitle(r'Comparison of Integration Methods: $\int_{' + 
                    f'{a}' + '}^{' + f'{b}' + '} f(x)\,dx$', fontsize=14)
        plt.tight_layout()
        return fig, axes


class MathViz3D:
    """
    3D Integration Visualization Toolkit
    ====================================
    Visualize triple integrals, volume integrals, and 3D regions.
    """

    def __init__(self, figsize=(14, 10)):
        self.figsize = figsize
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'surface': '#2E86AB40',
            'wireframe': '#2E86AB80',
            'grid': '#CCCCCC'
        }

    def volume_integral(self, f, x_bounds, y_bounds, z_bounds,
                       title=None, cmap='viridis', n=50,
                       show_wireframe=True, show_slices=True,
                       show_projections=True, alpha=0.6):
        """
        Visualize triple integral / volume under a 3D surface.

        Parameters:
        -----------
        f : callable or None
            Function f(x,y,z) for integrand, or None for just region visualization
        x_bounds, y_bounds, z_bounds : tuples
            (min, max) for each dimension
        title : str
            Plot title
        cmap : str
            Colormap
        n : int
            Grid resolution (use lower for performance)
        show_wireframe : bool
            Show wireframe overlay
        show_slices : bool
            Show cross-sectional slices
        show_projections : bool
            Show 2D projections
        alpha : float
            Surface opacity

        Returns:
        --------
        fig, axes : matplotlib objects
        """
        xmin, xmax = x_bounds
        ymin, ymax = y_bounds
        zmin, zmax = z_bounds

        x = np.linspace(xmin, xmax, n)
        y = np.linspace(ymin, ymax, n)
        z = np.linspace(zmin, zmax, n)

        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(20, 14))

        # Main 3D isometric view
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        if f is not None:
            Z_surface = f(X, Y)
            # Clip to bounds
            Z_surface = np.clip(Z_surface, zmin, zmax)
            surf = ax1.plot_surface(X, Y, Z_surface, cmap=cmap, alpha=alpha,
                                   linewidth=0, antialiased=True, zorder=2)
            if show_wireframe:
                ax1.plot_wireframe(X, Y, Z_surface, color='black', 
                                  alpha=0.2, linewidth=0.5, rstride=5, cstride=5)

        # Draw bounding box
        self._draw_bounding_box(ax1, x_bounds, y_bounds, z_bounds)
        ax1.set_title('3D Isometric View', fontsize=12, pad=10)
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_zlabel(r'$z$')

        # XY Projection (top view)
        ax2 = fig.add_subplot(2, 3, 2)
        if f is not None:
            Z_top = f(X, Y)
            contour = ax2.contourf(X, Y, Z_top, levels=20, cmap=cmap, alpha=0.8)
            ax2.contour(X, Y, Z_top, levels=20, colors='black', alpha=0.3, linewidths=0.5)
            plt.colorbar(contour, ax=ax2, shrink=0.8)
        ax2.set_title('XY Projection (Top View)', fontsize=12)
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax2.set_aspect('equal')

        # XZ Projection (side view from y)
        ax3 = fig.add_subplot(2, 3, 3)
        X_xz, Z_xz = np.meshgrid(x, z)
        if f is not None:
            # Approximate: evaluate at y midpoint
            Y_mid = (ymin + ymax) / 2
            F_xz = f(X_xz, np.full_like(X_xz, Y_mid))
            contour = ax3.contourf(X_xz, Z_xz, F_xz, levels=20, cmap=cmap, alpha=0.8)
            ax3.contour(X_xz, Z_xz, F_xz, levels=20, colors='black', alpha=0.3, linewidths=0.5)
            plt.colorbar(contour, ax=ax3, shrink=0.8)
        ax3.set_title('XZ Projection (Side View)', fontsize=12)
        ax3.set_xlabel(r'$x$')
        ax3.set_ylabel(r'$z$')
        ax3.set_aspect('equal')

        # YZ Projection (side view from x)
        ax4 = fig.add_subplot(2, 3, 4)
        Y_yz, Z_yz = np.meshgrid(y, z)
        if f is not None:
            X_mid = (xmin + xmax) / 2
            F_yz = f(np.full_like(Y_yz, X_mid), Y_yz)
            contour = ax4.contourf(Y_yz, Z_yz, F_yz, levels=20, cmap=cmap, alpha=0.8)
            ax4.contour(Y_yz, Z_yz, F_yz, levels=20, colors='black', alpha=0.3, linewidths=0.5)
            plt.colorbar(contour, ax=ax4, shrink=0.8)
        ax4.set_title('YZ Projection (Side View)', fontsize=12)
        ax4.set_xlabel(r'$y$')
        ax4.set_ylabel(r'$z$')
        ax4.set_aspect('equal')

        # Cross-sections
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        if f is not None and show_slices:
            # X-slice at midpoint
            x_mid = (xmin + xmax) / 2
            Y_slice, Z_slice = np.meshgrid(y, z)
            # Create slice plane
            X_slice = np.full_like(Y_slice, x_mid)
            F_slice = f(X_slice, Y_slice)
            ax5.plot_surface(X_slice, Y_slice, F_slice, alpha=0.5, 
                           color=self.colors['accent'], zorder=3)

            # Y-slice at midpoint
            y_mid = (ymin + ymax) / 2
            X_slice2, Z_slice2 = np.meshgrid(x, z)
            Y_slice2 = np.full_like(X_slice2, y_mid)
            F_slice2 = f(X_slice2, Y_slice2)
            ax5.plot_surface(X_slice2, Y_slice2, F_slice2, alpha=0.5,
                           color=self.colors['secondary'], zorder=3)

        self._draw_bounding_box(ax5, x_bounds, y_bounds, z_bounds)
        ax5.set_title('Cross-Sections', fontsize=12, pad=10)
        ax5.set_xlabel(r'$x$')
        ax5.set_ylabel(r'$y$')
        ax5.set_zlabel(r'$z$')

        # Volume rendering approximation
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        if f is not None:
            # Create filled volume using multiple surfaces
            levels = np.linspace(zmin, zmax, 10)
            for level in levels:
                Z_level = np.full_like(X, level)
                # Only show where surface is above this level
                mask = f(X, Y) >= level
                if np.any(mask):
                    ax6.contourf(X, Y, Z_level, zdir='z', offset=level,
                               levels=[level-0.01, level+0.01], 
                               colors=[plt.cm.viridis((level-zmin)/(zmax-zmin))],
                               alpha=0.3)

        self._draw_bounding_box(ax6, x_bounds, y_bounds, z_bounds)
        ax6.set_title('Volume Rendering', fontsize=12, pad=10)
        ax6.set_xlabel(r'$x$')
        ax6.set_ylabel(r'$y$')
        ax6.set_zlabel(r'$z$')

        if title:
            fig.suptitle(title, fontsize=16, y=0.98)

        plt.tight_layout()
        return fig, [ax1, ax2, ax3, ax4, ax5, ax6]

    def _draw_bounding_box(self, ax, x_bounds, y_bounds, z_bounds):
        """Draw a wireframe bounding box."""
        xmin, xmax = x_bounds
        ymin, ymax = y_bounds
        zmin, zmax = z_bounds

        # Bottom face
        ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmin], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([xmax, xmin], [ymax, ymax], [zmin, zmin], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([xmin, xmin], [ymax, ymin], [zmin, zmin], 'k-', alpha=0.3, linewidth=0.5)

        # Top face
        ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([xmax, xmax], [ymin, ymax], [zmax, zmax], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([xmax, xmin], [ymax, ymax], [zmax, zmax], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([xmin, xmin], [ymax, ymin], [zmax, zmax], 'k-', alpha=0.3, linewidth=0.5)

        # Vertical edges
        ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], 'k-', alpha=0.3, linewidth=0.5)
        ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], 'k-', alpha=0.3, linewidth=0.5)

    def spherical_integral(self, f_spherical, r_bounds, theta_bounds, phi_bounds,
                          title=None, n=30):
        """
        Visualize integration in spherical coordinates.

        Parameters:
        -----------
        f_spherical : callable
            Function f(r, theta, phi) in spherical coordinates
        r_bounds : tuple
            (r_min, r_max)
        theta_bounds : tuple
            (theta_min, theta_max) in radians
        phi_bounds : tuple
            (phi_min, phi_max) in radians
        title : str
            Plot title
        n : int
            Grid resolution

        Returns:
        --------
        fig, axes : matplotlib objects
        """
        r_min, r_max = r_bounds
        theta_min, theta_max = theta_bounds
        phi_min, phi_max = phi_bounds

        r = np.linspace(r_min, r_max, n)
        theta = np.linspace(theta_min, theta_max, n)
        phi = np.linspace(phi_min, phi_max, n)

        fig = plt.figure(figsize=(18, 12))

        # Convert to Cartesian for visualization
        R, Theta, Phi = np.meshgrid(r, theta, phi, indexing='ij')
        X = R * np.sin(Phi) * np.cos(Theta)
        Y = R * np.sin(Phi) * np.sin(Theta)
        Z = R * np.cos(Phi)

        # Main 3D spherical view
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        # Plot outer shell
        R_outer, Theta_outer = np.meshgrid(
            np.linspace(r_min, r_max, n),
            np.linspace(theta_min, theta_max, n)
        )
        Phi_fixed = (phi_min + phi_max) / 2
        X_shell = R_outer * np.sin(Phi_fixed) * np.cos(Theta_outer)
        Y_shell = R_outer * np.sin(Phi_fixed) * np.sin(Theta_outer)
        Z_shell = R_outer * np.cos(Phi_fixed)
        ax1.plot_surface(X_shell, Y_shell, Z_shell, alpha=0.5, cmap='coolwarm')
        ax1.set_title('Spherical Shell View', fontsize=12, pad=10)
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_zlabel(r'$z$')

        # Theta-Phi cross-section
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        Theta_sec, Phi_sec = np.meshgrid(theta, phi)
        r_fixed = (r_min + r_max) / 2
        X_sec = r_fixed * np.sin(Phi_sec) * np.cos(Theta_sec)
        Y_sec = r_fixed * np.sin(Phi_sec) * np.sin(Theta_sec)
        Z_sec = r_fixed * np.cos(Phi_sec)
        ax2.plot_surface(X_sec, Y_sec, Z_sec, alpha=0.5, cmap='viridis')
        ax2.set_title(r'$\theta$-$\phi$ Cross-Section', fontsize=12, pad=10)
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax2.set_zlabel(r'$z$')

        # R-Theta cross-section
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        R_sec, Theta_sec2 = np.meshgrid(r, theta)
        phi_fixed = (phi_min + phi_max) / 2
        X_sec2 = R_sec * np.sin(phi_fixed) * np.cos(Theta_sec2)
        Y_sec2 = R_sec * np.sin(phi_fixed) * np.sin(Theta_sec2)
        Z_sec2 = R_sec * np.cos(phi_fixed)
        ax3.plot_surface(X_sec2, Y_sec2, Z_sec2, alpha=0.5, cmap='plasma')
        ax3.set_title(r'$r$-$\theta$ Cross-Section', fontsize=12, pad=10)
        ax3.set_xlabel(r'$x$')
        ax3.set_ylabel(r'$y$')
        ax3.set_zlabel(r'$z$')

        # Spherical coordinate grid
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        for r_val in np.linspace(r_min, r_max, 5):
            theta_line = np.linspace(theta_min, theta_max, 100)
            phi_line = np.full_like(theta_line, (phi_min + phi_max) / 2)
            x_line = r_val * np.sin(phi_line) * np.cos(theta_line)
            y_line = r_val * np.sin(phi_line) * np.sin(theta_line)
            z_line = r_val * np.cos(phi_line)
            ax4.plot(x_line, y_line, z_line, 'b-', alpha=0.5, linewidth=0.5)

        for theta_val in np.linspace(theta_min, theta_max, 8):
            r_line = np.linspace(r_min, r_max, 100)
            phi_line = np.full_like(r_line, (phi_min + phi_max) / 2)
            x_line = r_line * np.sin(phi_line) * np.cos(theta_val)
            y_line = r_line * np.sin(phi_line) * np.sin(theta_val)
            z_line = r_line * np.cos(phi_line)
            ax4.plot(x_line, y_line, z_line, 'r-', alpha=0.5, linewidth=0.5)

        ax4.set_title('Spherical Grid Lines', fontsize=12, pad=10)
        ax4.set_xlabel(r'$x$')
        ax4.set_ylabel(r'$y$')
        ax4.set_zlabel(r'$z$')

        # Volume element visualization
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        dr = (r_max - r_min) / 5
        dtheta = (theta_max - theta_min) / 5
        dphi = (phi_max - phi_min) / 5

        for i in range(5):
            for j in range(5):
                for k in range(5):
                    r0 = r_min + i * dr
                    t0 = theta_min + j * dtheta
                    p0 = phi_min + k * dphi
                    # Draw small volume element
                    self._draw_volume_element(ax5, r0, t0, p0, dr, dtheta, dphi)

        ax5.set_title(r'Volume Elements $dV = r^2 \sin\phi \,dr\,d\theta\,d\phi$', 
                     fontsize=12, pad=10)
        ax5.set_xlabel(r'$x$')
        ax5.set_ylabel(r'$y$')
        ax5.set_zlabel(r'$z$')

        # Projection onto xy plane
        ax6 = fig.add_subplot(2, 3, 6)
        theta_proj = np.linspace(theta_min, theta_max, 200)
        for r_val in np.linspace(r_min, r_max, 10):
            x_proj = r_val * np.sin((phi_min + phi_max) / 2) * np.cos(theta_proj)
            y_proj = r_val * np.sin((phi_min + phi_max) / 2) * np.sin(theta_proj)
            ax6.plot(x_proj, y_proj, 'b-', alpha=0.3, linewidth=0.5)

        for theta_val in np.linspace(theta_min, theta_max, 16):
            r_proj = np.linspace(r_min, r_max, 100)
            x_proj = r_proj * np.sin((phi_min + phi_max) / 2) * np.cos(theta_val)
            y_proj = r_proj * np.sin((phi_min + phi_max) / 2) * np.sin(theta_val)
            ax6.plot(x_proj, y_proj, 'r-', alpha=0.3, linewidth=0.5)

        ax6.set_title('Polar Projection (XY Plane)', fontsize=12)
        ax6.set_xlabel(r'$x$')
        ax6.set_ylabel(r'$y$')
        ax6.set_aspect('equal')
        ax6.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=16, y=0.98)

        plt.tight_layout()
        return fig, [ax1, ax2, ax3, ax4, ax5, ax6]

    def _draw_volume_element(self, ax, r, theta, phi, dr, dtheta, dphi):
        """Draw a small spherical volume element."""
        # 8 corners of the volume element
        corners = []
        for dr_i in [0, dr]:
            for dt_i in [0, dtheta]:
                for dp_i in [0, dphi]:
                    r_c = r + dr_i
                    t_c = theta + dt_i
                    p_c = phi + dp_i
                    x = r_c * np.sin(p_c) * np.cos(t_c)
                    y = r_c * np.sin(p_c) * np.sin(t_c)
                    z = r_c * np.cos(p_c)
                    corners.append([x, y, z])

        # Draw edges
        edges = [
            [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
            [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
        ]
        for edge in edges:
            pts = np.array([corners[edge[0]], corners[edge[1]]])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'g-', alpha=0.3, linewidth=0.5)


class MathVizAdvanced:
    """
    Advanced Mathematical Visualization Features
    ==========================================
    Line integrals, surface integrals, flux, and vector fields.
    """

    def __init__(self, figsize=(14, 10)):
        self.figsize = figsize

    def vector_field_integral(self, F, path, t_bounds, 
                             title=None, n_field=20, n_path=100):
        """
        Visualize line integral of vector field along a path.

        Parameters:
        -----------
        F : callable
            Vector field F(x, y) -> (Fx, Fy)
        path : callable
            Parametric path r(t) -> (x(t), y(t))
        t_bounds : tuple
            (t_min, t_max)
        title : str
            Plot title
        n_field : int
            Vector field grid density
        n_path : int
            Path resolution

        Returns:
        --------
        fig, axes : matplotlib objects
        """
        t_min, t_max = t_bounds

        # Create grid for vector field
        x = np.linspace(-3, 3, n_field)
        y = np.linspace(-3, 3, n_field)
        X, Y = np.meshgrid(x, y)

        U, V = F(X, Y)

        # Create path
        t = np.linspace(t_min, t_max, n_path)
        x_path, y_path = path(t)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Vector field with path
        ax1 = axes[0]
        magnitude = np.sqrt(U**2 + V**2)
        ax1.quiver(X, Y, U, V, magnitude, cmap='viridis', scale=50, width=0.003)
        ax1.plot(x_path, y_path, 'r-', linewidth=3, label=r'Path $C$', zorder=5)
        ax1.plot(x_path[0], y_path[0], 'go', markersize=10, label='Start', zorder=6)
        ax1.plot(x_path[-1], y_path[-1], 'mo', markersize=10, label='End', zorder=6)

        # Draw tangent vectors along path
        dt = t[1] - t[0]
        for i in range(0, len(t), 10):
            if i < len(t) - 1:
                dx = x_path[i+1] - x_path[i]
                dy = y_path[i+1] - y_path[i]
                ax1.arrow(x_path[i], y_path[i], dx*5, dy*5, 
                         head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)

        ax1.set_title(r'Vector Field $\mathbf{F}$ and Path $C$', fontsize=12)
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.legend(loc='best')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Work visualization
        ax2 = axes[1]
        # Compute F along path and dot with tangent
        Fx_path, Fy_path = F(x_path, y_path)
        dx = np.gradient(x_path)
        dy = np.gradient(y_path)
        work_density = Fx_path * dx + Fy_path * dy

        # Color path by work density
        points = np.array([x_path, y_path]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        from matplotlib.collections import LineCollection
        norm = plt.Normalize(work_density.min(), work_density.max())
        lc = LineCollection(segments, cmap='RdYlBu_r', norm=norm)
        lc.set_array(work_density[:-1])
        lc.set_linewidth(4)
        ax2.add_collection(lc)

        ax2.set_xlim(x_path.min()-0.5, x_path.max()+0.5)
        ax2.set_ylim(y_path.min()-0.5, y_path.max()+0.5)
        ax2.set_title(r'Work Density $\mathbf{F} \cdot d\mathbf{r}$', fontsize=12)
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax2, shrink=0.8)
        cbar.set_label('Work Density', rotation=270, labelpad=15)

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        return fig, axes

    def surface_integral(self, param_surface, u_bounds, v_bounds,
                        title=None, n=50, show_normal_vectors=True):
        """
        Visualize surface integral with normal vectors.

        Parameters:
        -----------
        param_surface : callable
            Parametric surface r(u, v) -> (x, y, z)
        u_bounds, v_bounds : tuples
            Parameter bounds
        title : str
            Plot title
        n : int
            Grid resolution
        show_normal_vectors : bool
            Show normal vector field

        Returns:
        --------
        fig, axes : matplotlib objects
        """
        u_min, u_max = u_bounds
        v_min, v_max = v_bounds

        u = np.linspace(u_min, u_max, n)
        v = np.linspace(v_min, v_max, n)
        U, V = np.meshgrid(u, v)

        X, Y, Z = param_surface(U, V)

        fig = plt.figure(figsize=(16, 12))

        # Main 3D surface view
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                               linewidth=0, antialiased=True)

        if show_normal_vectors:
            # Compute and draw normal vectors
            du = (u_max - u_min) / n
            dv = (v_max - v_min) / n
            for i in range(0, n, 5):
                for j in range(0, n, 5):
                    # Approximate partial derivatives
                    if i < n-1 and j < n-1:
                        r_u = np.array([X[j, i+1] - X[j, i], 
                                       Y[j, i+1] - Y[j, i],
                                       Z[j, i+1] - Z[j, i]]) / du
                        r_v = np.array([X[j+1, i] - X[j, i],
                                       Y[j+1, i] - Y[j, i],
                                       Z[j+1, i] - Z[j, i]]) / dv
                        normal = np.cross(r_u, r_v)
                        normal = normal / (np.linalg.norm(normal) + 1e-10) * 0.3
                        ax1.quiver(X[j, i], Y[j, i], Z[j, i],
                                  normal[0], normal[1], normal[2],
                                  color='red', alpha=0.6, arrow_length_ratio=0.3)

        ax1.set_title('Parametric Surface with Normals', fontsize=12, pad=10)
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_zlabel(r'$z$')

        # Parameter domain
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.fill_between([u_min, u_max], v_min, v_max, alpha=0.3, color='blue')
        ax2.set_xlim(u_min, u_max)
        ax2.set_ylim(v_min, v_max)
        ax2.set_title('Parameter Domain', fontsize=12)
        ax2.set_xlabel(r'$u$')
        ax2.set_ylabel(r'$v$')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        # XY Projection
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)
        z_offset = Z.min() - 0.5
        ax3.contourf(X, Y, Z, zdir='z', offset=z_offset, cmap='viridis', levels=20, alpha=0.6)
        ax3.set_title('XY Projection', fontsize=12, pad=10)
        ax3.set_xlabel(r'$x$')
        ax3.set_ylabel(r'$y$')
        ax3.set_zlabel(r'$z$')

        # Surface area element visualization
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)
        # Draw a few surface patches
        for i in range(0, n-1, 10):
            for j in range(0, n-1, 10):
                patch_x = [X[j, i], X[j, i+1], X[j+1, i+1], X[j+1, i]]
                patch_y = [Y[j, i], Y[j, i+1], Y[j+1, i+1], Y[j+1, i]]
                patch_z = [Z[j, i], Z[j, i+1], Z[j+1, i+1], Z[j+1, i]]
                verts = [list(zip(patch_x, patch_y, patch_z))]
                poly3d = Poly3DCollection(verts, alpha=0.5, facecolor='orange',
                                         edgecolor='black', linewidth=0.5)
                ax4.add_collection3d(poly3d)

        ax4.set_title(r'Surface Elements $dS = |\mathbf{r}_u \times \mathbf{r}_v|\,du\,dv$', 
                     fontsize=12, pad=10)
        ax4.set_xlabel(r'$x$')
        ax4.set_ylabel(r'$y$')
        ax4.set_zlabel(r'$z$')

        if title:
            fig.suptitle(title, fontsize=16, y=0.98)

        plt.tight_layout()
        return fig, [ax1, ax2, ax3, ax4]


# =============================================================================
# DEMONSTRATION / TEST FUNCTIONS
# =============================================================================

def demo_2d_integration():
    """Demonstrate 2D integration visualization."""
    viz = MathViz2D(figsize=(14, 8))

    # Example 1: Simple polynomial
    print("=" * 60)
    print("DEMO 1: Single Variable Integration")
    print("=" * 60)

    f = lambda x: x**2 * np.sin(x)
    fig, ax = viz.integrate_1d(f, 0, 4, method='riemann', n=30,
                               title=r'$\int_0^4 x^2 \sin(x) \,dx$')
    plt.savefig('/mnt/agents/output/demo_2d_riemann.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: demo_2d_riemann.png")

    # Example 2: Filled area
    fig, ax = viz.integrate_1d(f, 0, 4, method='filled',
                               title=r'$\int_0^4 x^2 \sin(x) \,dx$ (Filled)')
    plt.savefig('/mnt/agents/output/demo_2d_filled.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: demo_2d_filled.png")

    # Example 3: Double integral over region
    print("\n" + "=" * 60)
    print("DEMO 2: Double Integration over Region")
    print("=" * 60)

    f2d = lambda x, y: np.sin(np.sqrt(x**2 + y**2))
    fig, axes = viz.integrate_2d_region(
        f2d, 
        x_bounds=(-3, 3),
        y_bounds=(lambda x: -np.sqrt(9-x**2), lambda x: np.sqrt(9-x**2)),
        title=r'$\iint_D \sin(\sqrt{x^2+y^2}) \,dA$, $D: x^2+y^2 \leq 9$',
        n=80
    )
    plt.savefig('/mnt/agents/output/demo_2d_double_integral.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: demo_2d_double_integral.png")

    # Example 4: Method comparison
    print("\n" + "=" * 60)
    print("DEMO 3: Integration Methods Comparison")
    print("=" * 60)

    fig, axes = viz.compare_methods(lambda x: np.exp(-x**2), -2, 2, 
                                    n_values=[10, 50, 100])
    plt.savefig('/mnt/agents/output/demo_2d_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: demo_2d_comparison.png")


def demo_3d_integration():
    """Demonstrate 3D integration visualization."""
    viz = MathViz3D(figsize=(14, 10))

    print("\n" + "=" * 60)
    print("DEMO 4: Volume Integration (3D)")
    print("=" * 60)

    # Volume under a Gaussian surface
    f3d = lambda x, y: 3 * np.exp(-(x**2 + y**2)/2)
    fig, axes = viz.volume_integral(
        f3d,
        x_bounds=(-3, 3),
        y_bounds=(-3, 3),
        z_bounds=(0, 3.5),
        title=r'$\iiint_V e^{-(x^2+y^2)/2} \,dV$',
        n=40
    )
    plt.savefig('/mnt/agents/output/demo_3d_volume.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: demo_3d_volume.png")

    # Spherical coordinates
    print("\n" + "=" * 60)
    print("DEMO 5: Spherical Coordinates Integration")
    print("=" * 60)

    fig, axes = viz.spherical_integral(
        None,  # Just visualize the coordinate system
        r_bounds=(0.5, 2),
        theta_bounds=(0, 2*np.pi),
        phi_bounds=(0, np.pi/2),
        title='Spherical Coordinate System',
        n=25
    )
    plt.savefig('/mnt/agents/output/demo_3d_spherical.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: demo_3d_spherical.png")


def demo_advanced():
    """Demonstrate advanced features."""
    viz = MathVizAdvanced(figsize=(14, 10))

    print("\n" + "=" * 60)
    print("DEMO 6: Line Integral (Vector Field)")
    print("=" * 60)

    # Vector field F(x,y) = (-y, x) - rotational field
    F = lambda x, y: (-y, x)
    # Path: circle
    path = lambda t: (2*np.cos(t), 2*np.sin(t))

    fig, axes = viz.vector_field_integral(
        F, path, (0, 2*np.pi),
        title=r'$\oint_C \mathbf{F} \cdot d\mathbf{r}$, $\mathbf{F}=(-y, x)$, $C: x^2+y^2=4$'
    )
    plt.savefig('/mnt/agents/output/demo_line_integral.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: demo_line_integral.png")

    print("\n" + "=" * 60)
    print("DEMO 7: Surface Integral")
    print("=" * 60)

    # Parametric surface: sphere
    sphere = lambda u, v: (
        2*np.sin(u)*np.cos(v),
        2*np.sin(u)*np.sin(v),
        2*np.cos(u)
    )

    fig, axes = viz.surface_integral(
        sphere,
        u_bounds=(0, np.pi),
        v_bounds=(0, 2*np.pi),
        title=r'Surface Integral over Sphere $x^2+y^2+z^2=4$',
        n=30
    )
    plt.savefig('/mnt/agents/output/demo_surface_integral.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: demo_surface_integral.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("MathViz - Mathematical Integration Visualization Library")
    print("=" * 60)
    print("\nRunning comprehensive demonstrations...\n")

    demo_2d_integration()
    demo_3d_integration()
    demo_advanced()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 60)
