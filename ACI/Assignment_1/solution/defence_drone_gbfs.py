"""
================================================================================
Defence Drone - Greedy Best-First Search (GBFS) & A* Pathfinding Agent
Assignment 1 - PS4
Course: AIMLCZG557/AECLZG557 (S2_2025-2026)
Group: G116
================================================================================
Description:
    An autonomous UAV agent navigates an 8x8 grid from Start (0,0) to Goal (6,7)
    using GBFS with two heuristics and A* for comparison. The environment contains
    No-Fly Zones (impassable) and Weather Hazards (penalty zones).
================================================================================
"""

import heapq
import math
import time
import sys
import os
import io
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict
from tabulate import tabulate

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

# Cell types
CELL_START = 'S'
CELL_GOAL = 'E'
CELL_PASSABLE = '.'
CELL_WEATHER = 'W'
CELL_NOFLY = 'N'

# Transition costs (used in h2 bounding box and path cost)
COST_MAP = {
    CELL_START: 1,
    CELL_GOAL: 1,
    CELL_PASSABLE: 1,
    CELL_WEATHER: 4,
    CELL_NOFLY: 8
}

# Directions in priority order: North, East, South, West
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = ['North', 'East', 'South', 'West']


# ============================================================
# GRID CLASS
# ============================================================

class Grid:
    """Represents the 8x8 airspace grid environment."""

    def __init__(self, grid_chars, start, goal):
        self.grid = grid_chars
        self.rows = len(grid_chars)
        self.cols = len(grid_chars[0])
        self.start = start
        self.goal = goal

    @classmethod
    def from_file(cls, filepath):
        """Load grid from input file."""
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')

        rows, cols = map(int, lines[0].split())
        grid_chars = []
        for i in range(1, rows + 1):
            row = lines[i].split()
            grid_chars.append(row)

        start_coords = tuple(map(int, lines[rows + 1].split()))
        goal_coords = tuple(map(int, lines[rows + 2].split()))

        return cls(grid_chars, start_coords, goal_coords)

    def get_cell(self, row, col):
        """Get cell type at position."""
        return self.grid[row][col]

    def get_cost(self, row, col):
        """Get transition cost for a cell."""
        return COST_MAP[self.grid[row][col]]

    def is_valid(self, row, col):
        """Check if cell is within bounds and not a No-Fly Zone."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col] != CELL_NOFLY
        return False

    def get_neighbors(self, row, col):
        """Get valid neighbors in priority order: North, East, South, West."""
        neighbors = []
        for i, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = row + dr, col + dc
            if self.is_valid(nr, nc):
                neighbors.append((nr, nc, DIR_NAMES[i]))
        return neighbors

    def count_obstacles_on_path(self, path):
        """Count No-Fly zones and Weather Hazards crossed on a path."""
        nofly_count = 0
        weather_count = 0
        weather_penalty = 0
        for r, c in path:
            cell = self.get_cell(r, c)
            if cell == CELL_WEATHER:
                weather_count += 1
                weather_penalty += COST_MAP[CELL_WEATHER]
            elif cell == CELL_NOFLY:
                nofly_count += 1
        return nofly_count, weather_count, weather_penalty

    def display(self):
        """Print the grid in a formatted way."""
        print("\n" + "=" * 50)
        print("ENVIRONMENT GRID (8x8)")
        print("=" * 50)
        header = "     " + "   ".join([str(i) for i in range(self.cols)])
        print(header)
        print("   " + "-" * (self.cols * 4 + 1))
        for r in range(self.rows):
            row_str = f" {r} | "
            for c in range(self.cols):
                cell = self.grid[r][c]
                row_str += f" {cell}  "
            print(row_str)
        print()


# ============================================================
# HEURISTIC FUNCTIONS
# ============================================================

def h1_euclidean(node, goal):
    """
    Heuristic h1: Euclidean Distance.
    h1(n) = sqrt((x_n - x_goal)^2 + (y_n - y_goal)^2)
    """
    return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)


def h2_bounding_box(node, goal, grid):
    """
    Heuristic h2: Bounding-Box Risk-Weighted Heuristic (future-aware).
    h2(n) = Manhattan_distance(n, goal) * (1/K * sum(c(cell) for cell in BoundingBox))
    """
    manhattan_dist = abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    if manhattan_dist == 0:
        return 0.0

    # Define bounding box from current node to goal
    min_row = min(node[0], goal[0])
    max_row = max(node[0], goal[0])
    min_col = min(node[1], goal[1])
    max_col = max(node[1], goal[1])

    # Calculate K (number of cells in bounding box)
    K = (max_row - min_row + 1) * (max_col - min_col + 1)

    # Sum of all cell costs in the bounding box
    total_cost = 0
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            total_cost += grid.get_cost(r, c)

    avg_cost = total_cost / K
    return manhattan_dist * avg_cost


# ============================================================
# GREEDY BEST-FIRST SEARCH (GBFS)
# ============================================================

def gbfs(grid, heuristic_func, heuristic_name="h1"):
    """
    Greedy Best-First Search implementation.

    Parameters:
        grid: Grid object
        heuristic_func: function(node, goal) -> float (or function(node, goal, grid))
        heuristic_name: string identifier for the heuristic

    Returns:
        Dictionary containing path, metrics, and iteration details
    """
    start_time = time.time()

    # Priority queue: (h_value, counter, node)
    open_list = []
    counter = 0  # Tie-breaking: preserves direction priority (N, E, S, W)

    # Compute initial heuristic
    if "h2" in heuristic_name:
        h_start = heuristic_func(grid.start, grid.goal, grid)
    else:
        h_start = heuristic_func(grid.start, grid.goal)

    heapq.heappush(open_list, (h_start, counter, grid.start))
    counter += 1

    came_from = {grid.start: None}
    explored = set()
    open_set = {grid.start}

    # Tracking for output
    frontier_history = []
    iteration_details = []
    h_values_along_path = []

    while open_list:
        # Record current frontier
        current_frontier = [(h, node) for h, _, node in open_list]
        frontier_history.append(current_frontier[:])

        # Pop node with lowest h value
        h_val, _, current = heapq.heappop(open_list)
        open_set.discard(current)

        # Iteration detail
        iteration_info = {
            'iteration': len(iteration_details) + 1,
            'current_node': current,
            'h_value': round(h_val, 4),
            'frontier_size': len(open_list) + 1,
            'explored_count': len(explored),
            'frontier_nodes': [(node, round(h, 4)) for h, node in current_frontier],
        }

        # Check if goal reached
        if current == grid.goal:
            end_time = time.time()
            path = reconstruct_path(came_from, current)
            nofly, weather, w_penalty = grid.count_obstacles_on_path(path)

            h_values_along_path.append((current, h_val))

            metrics = {
                'nodes_expanded': len(explored) + 1,
                'runtime_ms': round((end_time - start_time) * 1000, 4),
                'memory_usage': len(open_set) + len(explored) + 1,
                'total_path_cost': sum(grid.get_cost(r, c) for r, c in path),
                'path_length': len(path) - 1,
                'heuristic': heuristic_name,
                'nofly_crossed': nofly,
                'weather_crossed': weather,
                'weather_penalty': w_penalty,
            }

            iteration_info['selected_next'] = 'GOAL REACHED'
            iteration_details.append(iteration_info)

            return {
                'path': path,
                'explored': explored,
                'metrics': metrics,
                'frontier_history': frontier_history,
                'iteration_details': iteration_details,
                'h_values_along_path': h_values_along_path,
            }

        explored.add(current)
        h_values_along_path.append((current, h_val))

        # Generate neighbors in priority order: N, E, S, W
        neighbors = grid.get_neighbors(current[0], current[1])
        neighbor_info = []

        for nr, nc, direction in neighbors:
            neighbor = (nr, nc)
            if neighbor in explored:
                continue
            if neighbor not in open_set:
                came_from[neighbor] = current
                if "h2" in heuristic_name:
                    h_n = heuristic_func(neighbor, grid.goal, grid)
                else:
                    h_n = heuristic_func(neighbor, grid.goal)
                heapq.heappush(open_list, (h_n, counter, neighbor))
                counter += 1
                open_set.add(neighbor)
                neighbor_info.append((neighbor, round(h_n, 4), direction))

        iteration_info['neighbors_evaluated'] = neighbor_info

        # Determine selected next node
        if open_list:
            next_h, _, next_node = open_list[0]
            iteration_info['selected_next'] = next_node
        else:
            iteration_info['selected_next'] = None

        iteration_details.append(iteration_info)

    # No path found
    end_time = time.time()
    return {
        'path': None,
        'explored': explored,
        'metrics': {
            'nodes_expanded': len(explored),
            'runtime_ms': round((end_time - start_time) * 1000, 4),
            'memory_usage': len(open_set) + len(explored),
            'total_path_cost': float('inf'),
            'path_length': 0,
            'heuristic': heuristic_name,
            'nofly_crossed': 0,
            'weather_crossed': 0,
            'weather_penalty': 0,
        },
        'frontier_history': frontier_history,
        'iteration_details': iteration_details,
        'h_values_along_path': h_values_along_path,
    }


# ============================================================
# A* SEARCH
# ============================================================

def astar(grid, heuristic_func, heuristic_name="h1"):
    """
    A* Search implementation using heuristic h1 (Euclidean).

    Returns:
        Dictionary containing path, metrics, and iteration details
    """
    start_time = time.time()

    open_list = []
    counter = 0

    g_score = {grid.start: grid.get_cost(grid.start[0], grid.start[1])}
    h_start = heuristic_func(grid.start, grid.goal)
    f_start = g_score[grid.start] + h_start

    heapq.heappush(open_list, (f_start, counter, grid.start))
    counter += 1

    came_from = {grid.start: None}
    explored = set()
    open_set = {grid.start}
    frontier_history = []
    iteration_details = []

    while open_list:
        current_frontier = [(f, node) for f, _, node in open_list]
        frontier_history.append(current_frontier[:])

        f_val, _, current = heapq.heappop(open_list)
        open_set.discard(current)

        iteration_info = {
            'iteration': len(iteration_details) + 1,
            'current_node': current,
            'f_value': round(f_val, 4),
            'g_value': round(g_score[current], 4),
            'h_value': round(f_val - g_score[current], 4),
            'frontier_size': len(open_list) + 1,
            'explored_count': len(explored),
        }

        if current == grid.goal:
            end_time = time.time()
            path = reconstruct_path(came_from, current)
            nofly, weather, w_penalty = grid.count_obstacles_on_path(path)

            metrics = {
                'nodes_expanded': len(explored) + 1,
                'runtime_ms': round((end_time - start_time) * 1000, 4),
                'memory_usage': len(open_set) + len(explored) + 1,
                'total_path_cost': g_score[current],
                'path_length': len(path) - 1,
                'heuristic': heuristic_name,
                'nofly_crossed': nofly,
                'weather_crossed': weather,
                'weather_penalty': w_penalty,
            }

            iteration_info['selected_next'] = 'GOAL REACHED'
            iteration_details.append(iteration_info)

            return {
                'path': path,
                'explored': explored,
                'metrics': metrics,
                'frontier_history': frontier_history,
                'iteration_details': iteration_details,
            }

        explored.add(current)

        neighbors = grid.get_neighbors(current[0], current[1])

        for nr, nc, direction in neighbors:
            neighbor = (nr, nc)
            if neighbor in explored:
                continue

            tentative_g = g_score[current] + grid.get_cost(nr, nc)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h_n = heuristic_func(neighbor, grid.goal)
                f_n = tentative_g + h_n
                heapq.heappush(open_list, (f_n, counter, neighbor))
                counter += 1
                open_set.add(neighbor)

        iteration_details.append(iteration_info)

    end_time = time.time()
    return {
        'path': None,
        'explored': explored,
        'metrics': {
            'nodes_expanded': len(explored),
            'runtime_ms': round((end_time - start_time) * 1000, 4),
            'memory_usage': len(open_set) + len(explored),
            'total_path_cost': float('inf'),
            'path_length': 0,
            'heuristic': heuristic_name,
            'nofly_crossed': 0,
            'weather_crossed': 0,
            'weather_penalty': 0,
        },
        'frontier_history': frontier_history,
        'iteration_details': iteration_details,
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def reconstruct_path(came_from, current):
    """Reconstruct path from came_from dictionary."""
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def identify_traps(iteration_details, grid):
    """
    Identify trap situations where the agent backtracks or hits dead-ends.
    A trap is when the selected next node is not adjacent to the current node,
    indicating the agent had to backtrack from the frontier.
    """
    traps = []
    for i in range(len(iteration_details) - 1):
        current = iteration_details[i]['current_node']
        next_node = iteration_details[i]['selected_next']

        if next_node is None or next_node == 'GOAL REACHED':
            continue

        # Check if next node is adjacent to current node
        row_diff = abs(next_node[0] - current[0])
        col_diff = abs(next_node[1] - current[1])

        if row_diff + col_diff > 1:
            # Agent is backtracking - this is a trap/dead-end situation
            trap_info = {
                'heuristic': iteration_details[i].get('heuristic', ''),
                'h_value': iteration_details[i]['h_value'],
                'trapped_at_node': current,
                'iteration': iteration_details[i]['iteration'],
                'escaped_to': next_node,
                'escape_iteration': iteration_details[i + 1]['iteration'] if i + 1 < len(iteration_details) else 'N/A',
            }
            traps.append(trap_info)

    return traps


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def visualize_grid(grid, path=None, explored=None, title="Grid Environment",
                   save_path=None, show_heuristic=None):
    """Visualize the grid with optional path and explored nodes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Color mapping
    color_map = {
        CELL_START: '#00FF00',     # Green
        CELL_GOAL: '#FF0000',      # Red
        CELL_PASSABLE: '#FFFFFF',  # White
        CELL_WEATHER: '#FFD700',   # Gold
        CELL_NOFLY: '#404040',     # Dark Gray
    }

    # Draw grid cells
    for r in range(grid.rows):
        for c in range(grid.cols):
            cell = grid.get_cell(r, c)
            color = color_map[cell]

            # Highlight explored nodes
            if explored and (r, c) in explored and cell not in [CELL_START, CELL_GOAL]:
                color = '#ADD8E6'  # Light blue for explored

            # Highlight path
            if path and (r, c) in path and cell not in [CELL_START, CELL_GOAL]:
                color = '#90EE90'  # Light green for path

            rect = plt.Rectangle((c, grid.rows - 1 - r), 1, 1,
                                 facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

            # Cell labels
            icon_map = {
                CELL_START: 'S',
                CELL_GOAL: 'E',
                CELL_WEATHER: 'W',
                CELL_NOFLY: 'N',
                CELL_PASSABLE: '.',
            }
            icon = icon_map[cell]
            fontcolor = 'black'
            fontsize = 14

            if cell == CELL_NOFLY:
                fontcolor = 'white'
                fontsize = 16
            elif cell == CELL_START:
                fontcolor = 'darkgreen'
                fontsize = 16
            elif cell == CELL_GOAL:
                fontcolor = 'darkred'
                fontsize = 16
            elif cell == CELL_WEATHER:
                fontcolor = 'darkorange'
                fontsize = 16

            if path and (r, c) in path and cell == CELL_PASSABLE:
                icon = '*'
                fontcolor = 'blue'

            ax.text(c + 0.5, grid.rows - 1 - r + 0.5, icon,
                    ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', color=fontcolor)

            # Show heuristic values
            if show_heuristic and (r, c) in show_heuristic:
                ax.text(c + 0.5, grid.rows - 1 - r + 0.15,
                        f"{show_heuristic[(r, c)]:.2f}",
                        ha='center', va='center', fontsize=7, color='blue')

    # Draw path arrows
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            dx = (c2 - c1) * 0.3
            dy = (r1 - r2) * 0.3
            ax.annotate('', xy=(c2 + 0.5, grid.rows - 1 - r2 + 0.5),
                        xytext=(c1 + 0.5, grid.rows - 1 - r1 + 0.5),
                        arrowprops=dict(arrowstyle='->', color='blue',
                                        lw=2, connectionstyle='arc3,rad=0'))

    ax.set_xlim(0, grid.cols)
    ax.set_ylim(0, grid.rows)
    ax.set_xticks(np.arange(0.5, grid.cols, 1))
    ax.set_yticks(np.arange(0.5, grid.rows, 1))
    ax.set_xticklabels(range(grid.cols))
    ax.set_yticklabels(range(grid.rows - 1, -1, -1))
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#00FF00', edgecolor='black', label='Start (S)'),
        mpatches.Patch(facecolor='#FF0000', edgecolor='black', label='Goal (E)'),
        mpatches.Patch(facecolor='#FFFFFF', edgecolor='black', label='Passable (.)'),
        mpatches.Patch(facecolor='#FFD700', edgecolor='black', label='Weather (W)'),
        mpatches.Patch(facecolor='#404040', edgecolor='black', label='No-Fly (N)'),
        mpatches.Patch(facecolor='#ADD8E6', edgecolor='black', label='Explored'),
        mpatches.Patch(facecolor='#90EE90', edgecolor='black', label='Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_comparison(metrics_list, title="Algorithm Comparison"):
    """Create comparison bar charts for different algorithms/heuristics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    labels = [f"{m['heuristic']}" for m in metrics_list]
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    # Nodes Expanded
    values = [m['nodes_expanded'] for m in metrics_list]
    axes[0, 0].bar(labels, values, color=colors[:len(labels)])
    axes[0, 0].set_title('Nodes Expanded')
    axes[0, 0].set_ylabel('Count')

    # Runtime
    values = [m['runtime_ms'] for m in metrics_list]
    axes[0, 1].bar(labels, values, color=colors[:len(labels)])
    axes[0, 1].set_title('Runtime (ms)')
    axes[0, 1].set_ylabel('Time (ms)')

    # Memory Usage
    values = [m['memory_usage'] for m in metrics_list]
    axes[0, 2].bar(labels, values, color=colors[:len(labels)])
    axes[0, 2].set_title('Memory Usage (OPEN + CLOSED)')
    axes[0, 2].set_ylabel('Nodes')

    # Total Path Cost
    values = [m['total_path_cost'] for m in metrics_list]
    axes[1, 0].bar(labels, values, color=colors[:len(labels)])
    axes[1, 0].set_title('Total Path Cost')
    axes[1, 0].set_ylabel('Cost')

    # Path Length
    values = [m['path_length'] for m in metrics_list]
    axes[1, 1].bar(labels, values, color=colors[:len(labels)])
    axes[1, 1].set_title('Path Length (moves)')
    axes[1, 1].set_ylabel('Moves')

    # Weather Zones Crossed
    values = [m['weather_crossed'] for m in metrics_list]
    axes[1, 2].bar(labels, values, color=colors[:len(labels)])
    axes[1, 2].set_title('Weather Zones Crossed')
    axes[1, 2].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('comparison_chart.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_heuristic_progression(h_values_h1, h_values_h2, title="Heuristic Values Along Path"):
    """Visualize heuristic values at each step for both heuristics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # h1 progression
    steps_h1 = range(len(h_values_h1))
    vals_h1 = [v for _, v in h_values_h1]
    ax1.plot(steps_h1, vals_h1, 'b-o', markersize=4, label='h1 (Euclidean)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Heuristic Value')
    ax1.set_title('h1: Euclidean Distance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # h2 progression
    steps_h2 = range(len(h_values_h2))
    vals_h2 = [v for _, v in h_values_h2]
    ax2.plot(steps_h2, vals_h2, 'r-o', markersize=4, label='h2 (Bounding-Box)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Heuristic Value')
    ax2.set_title('h2: Bounding-Box Risk-Weighted')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('heuristic_progression.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_heuristic_vs_time(h_values_h1, h_values_h2, runtime_h1, runtime_h2):
    """Visualize heuristic values vs estimated time progression."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Approximate time per step
    n_h1 = len(h_values_h1)
    n_h2 = len(h_values_h2)

    time_steps_h1 = np.linspace(0, runtime_h1, n_h1)
    time_steps_h2 = np.linspace(0, runtime_h2, n_h2)

    vals_h1 = [v for _, v in h_values_h1]
    vals_h2 = [v for _, v in h_values_h2]

    ax.plot(time_steps_h1, vals_h1, 'b-o', markersize=3, label='GBFS h1 (Euclidean)')
    ax.plot(time_steps_h2, vals_h2, 'r-s', markersize=3, label='GBFS h2 (Bounding-Box)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Heuristic Value')
    ax.set_title('Heuristic Values vs Time to Reach Target')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heuristic_vs_time.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# OUTPUT FUNCTIONS
# ============================================================

def print_separator(char='=', length=70):
    """Print a separator line."""
    print(char * length)


def print_section(title):
    """Print a section header."""
    print_separator()
    print(f"  {title}")
    print_separator()


def print_iteration_details(iteration_details, algorithm_name):
    """Print detailed iteration information."""
    print(f"\n{'─' * 70}")
    print(f"  {algorithm_name} - Iteration Details")
    print(f"{'─' * 70}")

    for detail in iteration_details:
        print(f"\n  Iteration {detail['iteration']}:")
        print(f"    Current Node: {detail['current_node']}")
        print(f"    Heuristic Value: {detail['h_value']}")
        print(f"    Explored Nodes Count: {detail['explored_count']}")
        print(f"    Frontier Size: {detail['frontier_size']}")

        if 'neighbors_evaluated' in detail:
            print(f"    Neighbors Evaluated:")
            for neighbor, h_val, direction in detail['neighbors_evaluated']:
                print(f"      → {direction}: {neighbor}, h={h_val}")

        if 'frontier_nodes' in detail and len(detail['frontier_nodes']) <= 10:
            print(f"    Frontier Nodes (top 10): {detail['frontier_nodes'][:10]}")

        print(f"    Selected Next Node: {detail['selected_next']}")


def print_path(path, grid):
    """Print the path with direction annotations."""
    if not path:
        print("  No path found!")
        return

    print(f"\n  Path ({len(path)} nodes, {len(path)-1} moves):")
    print(f"  {'─' * 40}")

    for i, (r, c) in enumerate(path):
        cell = grid.get_cell(r, c)
        cost = grid.get_cost(r, c)
        direction = ""
        if i > 0:
            dr = r - path[i - 1][0]
            dc = c - path[i - 1][1]
            dir_idx = DIRECTIONS.index((dr, dc))
            direction = f" <- {DIR_NAMES[dir_idx]}"
        cell_type = {'S': 'Start', 'E': 'Goal', '.': 'Passable',
                     'W': 'Weather[!]', 'N': 'No-Fly[X]'}
        print(f"    Step {i:2d}: ({r},{c}) [{cell_type[cell]}] cost={cost}{direction}")


def print_metrics_table(metrics_list):
    """Print comparison metrics table."""
    headers = ['Metric', ] + [m['heuristic'] for m in metrics_list]
    rows = [
        ['Nodes Expanded'] + [m['nodes_expanded'] for m in metrics_list],
        ['Runtime (ms)'] + [f"{m['runtime_ms']:.4f}" for m in metrics_list],
        ['Memory Usage'] + [m['memory_usage'] for m in metrics_list],
        ['Total Path Cost'] + [m['total_path_cost'] for m in metrics_list],
        ['Path Length'] + [m['path_length'] for m in metrics_list],
        ['Heuristic'] + [m['heuristic'] for m in metrics_list],
        ['Weather Zones Crossed'] + [m['weather_crossed'] for m in metrics_list],
        ['Weather Penalty'] + [m.get('weather_penalty', 0) for m in metrics_list],
    ]
    print(tabulate(rows, headers=headers, tablefmt='grid'))


def print_trap_table(traps_h1, traps_h2):
    """Print trap identification table."""
    all_traps = []
    for t in traps_h1:
        all_traps.append(['h1', t['h_value'], t['trapped_at_node'],
                          t['iteration'], t['escaped_to'], t['escape_iteration']])
    for t in traps_h2:
        all_traps.append(['h2', t['h_value'], t['trapped_at_node'],
                          t['iteration'], t['escaped_to'], t['escape_iteration']])

    if all_traps:
        headers = ['Heuristic', 'H-Value', 'Trapped at Node',
                   'Iteration', 'Escaped To', 'Escape Iteration']
        print(tabulate(all_traps, headers=headers, tablefmt='grid'))
    else:
        print("  No trap situations detected.")


def print_peas():
    """Print PEAS components for the Defence Drone agent."""
    print_section("PEAS COMPONENTS")
    print("""
  ┌─────────────────┬────────────────────────────────────────────────────────┐
  │ Component       │ Description                                            │
  ├─────────────────┼────────────────────────────────────────────────────────┤
  │ Performance     │ • Minimize path cost (transition costs)                │
  │ Measure         │ • Reach the goal (External Signal Source)              │
  │                 │ • Minimize nodes expanded                              │
  │                 │ • Avoid Weather Hazards (reduce penalties)             │
  │                 │ • Never enter No-Fly Zones                             │
  ├─────────────────┼────────────────────────────────────────────────────────┤
  │ Environment     │ • 8×8 discrete grid (airspace)                        │
  │                 │ • Static obstacles: No-Fly Zones (impassable)          │
  │                 │ • Hazards: Weather regions (passable, penalty +4)      │
  │                 │ • Fully observable (grid is known)                     │
  │                 │ • Deterministic (actions have predictable outcomes)    │
  │                 │ • Sequential (decisions depend on previous state)      │
  │                 │ • Single agent                                         │
  ├─────────────────┼────────────────────────────────────────────────────────┤
  │ Actuators       │ • Movement controller (North, South, East, West)      │
  │                 │ • Navigation system for path execution                 │
  │                 │ • Signal interception mechanism at goal                │
  ├─────────────────┼────────────────────────────────────────────────────────┤
  │ Sensors         │ • GPS/Position sensor (current coordinates)           │
  │                 │ • Grid map sensor (full environment knowledge)         │
  │                 │ • Obstacle detector (No-Fly Zone identification)       │
  │                 │ • Weather sensor (Hazard zone detection)               │
  │                 │ • Goal detector (External Signal Source locator)       │
  └─────────────────┴────────────────────────────────────────────────────────┘
  
  Agent Type: Goal-based, informed search agent
  Environment Type: Fully Observable, Deterministic, Sequential, Static, 
                    Discrete, Single-agent
""")


def print_complexity_analysis():
    """Print theoretical complexity analysis for GBFS and A*."""
    print_section("COMPLEXITY ANALYSIS")
    print("""
  ┌─────────────────────┬───────────────────────┬───────────────────────────┐
  │ Aspect              │ GBFS                  │ A*                        │
  ├─────────────────────┼───────────────────────┼───────────────────────────┤
  │ Time Complexity     │ O(b^m) worst case     │ O(b^d) with admissible h  │
  │                     │ O(b*d) best case      │                           │
  ├─────────────────────┼───────────────────────┼───────────────────────────┤
  │ Space Complexity    │ O(b^m) worst case     │ O(b^d)                    │
  │                     │ O(b*d) best case      │                           │
  ├─────────────────────┼───────────────────────┼───────────────────────────┤
  │ Completeness        │ No (can get stuck     │ Yes (with finite costs    │
  │                     │ in loops w/o closed   │ and admissible heuristic) │
  │                     │ set; Yes with closed) │                           │
  ├─────────────────────┼───────────────────────┼───────────────────────────┤
  │ Optimality          │ No (does not consider │ Yes (with admissible and  │
  │                     │ path cost g(n))       │ consistent heuristic)     │
  └─────────────────────┴───────────────────────┴───────────────────────────┘

  Where: b = branching factor (max 4 for orthogonal movement)
         m = maximum depth of search tree
         d = depth of optimal solution

  For this 8×8 grid:
  • Maximum possible nodes: 64 (8×8)
  • Effective nodes (excluding No-Fly): 64 - 8 = 56
  • Branching factor: ≤ 4 (orthogonal moves)
  • GBFS with closed set is complete for finite graphs
""")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main function - executes the complete assignment solution."""

    # Determine input file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'inputPS04.txt')

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found at: {input_file}")
        sys.exit(1)

    # Load grid from input file
    grid = Grid.from_file(input_file)

    # Open output file for writing
    output_file = os.path.join(script_dir, 'outputPS04.txt')
    original_stdout = sys.stdout

    # Write to both console and file
    class DualOutput:
        def __init__(self, file, console):
            self.file = file
            self.console = console

        def write(self, text):
            try:
                self.console.write(text)
            except UnicodeEncodeError:
                self.console.write(text.encode('ascii', 'replace').decode('ascii'))
            self.file.write(text)

        def flush(self):
            self.console.flush()
            self.file.flush()

    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = DualOutput(f, original_stdout)

        # ====================================================
        # SECTION 1: Grid Display
        # ====================================================
        print_section("DEFENCE DRONE - GBFS PATHFINDING AGENT")
        print("  Group: G116")
        print("  Course: AIMLCZG557/AECLZG557 (S2_2025-2026)")
        print(f"  Start Node: {grid.start}")
        print(f"  Goal Node: {grid.goal}")
        print(f"  Grid Size: {grid.rows} x {grid.cols}")
        grid.display()

        # Count obstacles
        nofly_count = sum(1 for r in range(grid.rows) for c in range(grid.cols)
                         if grid.get_cell(r, c) == CELL_NOFLY)
        weather_count = sum(1 for r in range(grid.rows) for c in range(grid.cols)
                           if grid.get_cell(r, c) == CELL_WEATHER)
        print(f"  No-Fly Zones: {nofly_count}")
        print(f"  Weather Hazards: {weather_count}")

        # ====================================================
        # SECTION 2: GBFS with Heuristic h1 (Euclidean)
        # ====================================================
        print_section("GBFS WITH HEURISTIC h1 (EUCLIDEAN DISTANCE)")

        result_h1 = gbfs(grid, h1_euclidean, "GBFS-h1")

        if result_h1['path']:
            print(f"\n  ✓ Path FOUND using h1 (Euclidean Distance)")
            print_path(result_h1['path'], grid)
            print(f"\n  Penalties Incurred:")
            print(f"    Weather Zones Crossed: {result_h1['metrics']['weather_crossed']}")
            print(f"    Weather Penalty Total: {result_h1['metrics']['weather_penalty']}")
            print(f"    No-Fly Zones Crossed: {result_h1['metrics']['nofly_crossed']}")
        else:
            print("  ✗ No path found using h1!")

        print_iteration_details(result_h1['iteration_details'], "GBFS h1")

        # ====================================================
        # SECTION 3: GBFS with Heuristic h2 (Bounding-Box)
        # ====================================================
        print_section("GBFS WITH HEURISTIC h2 (BOUNDING-BOX RISK-WEIGHTED)")

        result_h2 = gbfs(grid, h2_bounding_box, "GBFS-h2")

        if result_h2['path']:
            print(f"\n  ✓ Path FOUND using h2 (Bounding-Box Risk-Weighted)")
            print_path(result_h2['path'], grid)
            print(f"\n  Penalties Incurred:")
            print(f"    Weather Zones Crossed: {result_h2['metrics']['weather_crossed']}")
            print(f"    Weather Penalty Total: {result_h2['metrics']['weather_penalty']}")
            print(f"    No-Fly Zones Crossed: {result_h2['metrics']['nofly_crossed']}")
        else:
            print("  ✗ No path found using h2!")

        print_iteration_details(result_h2['iteration_details'], "GBFS h2")

        # ====================================================
        # SECTION 4: A* with Heuristic h1
        # ====================================================
        print_section("A* SEARCH WITH HEURISTIC h1 (EUCLIDEAN DISTANCE)")

        result_astar = astar(grid, h1_euclidean, "A*-h1")

        if result_astar['path']:
            print(f"\n  ✓ Path FOUND using A* with h1")
            print_path(result_astar['path'], grid)
            print(f"\n  Penalties Incurred:")
            print(f"    Weather Zones Crossed: {result_astar['metrics']['weather_crossed']}")
            print(f"    Weather Penalty Total: {result_astar['metrics']['weather_penalty']}")
            print(f"    No-Fly Zones Crossed: {result_astar['metrics']['nofly_crossed']}")
        else:
            print("  ✗ No path found using A*!")

        # ====================================================
        # SECTION 5: Metrics Comparison Table
        # ====================================================
        print_section("METRICS COMPARISON TABLE")

        all_metrics = []
        if result_h1['metrics']:
            all_metrics.append(result_h1['metrics'])
        if result_h2['metrics']:
            all_metrics.append(result_h2['metrics'])
        if result_astar['metrics']:
            all_metrics.append(result_astar['metrics'])

        print_metrics_table(all_metrics)

        # ====================================================
        # SECTION 6: Trap Identification
        # ====================================================
        print_section("TRAP IDENTIFICATION")
        print("""
  A trap is identified when the agent must backtrack - the next node
  selected from the frontier is NOT adjacent to the current node,
  indicating the agent explored a dead-end or inefficient region.
""")

        traps_h1 = identify_traps(result_h1['iteration_details'], grid)
        traps_h2 = identify_traps(result_h2['iteration_details'], grid)

        # Add heuristic labels
        for t in traps_h1:
            t['heuristic'] = 'h1'
        for t in traps_h2:
            t['heuristic'] = 'h2'

        print_trap_table(traps_h1, traps_h2)

        # ====================================================
        # SECTION 7: GBFS vs A* Comparison
        # ====================================================
        print_section("GBFS vs A* COMPARISON (Heuristic h1)")
        print("""
  ┌──────────────────────┬────────────────────┬────────────────────┐
  │ Criterion            │ GBFS (h1)          │ A* (h1)            │""")

        gbfs_m = result_h1['metrics']
        astar_m = result_astar['metrics']

        print(f"""  ├──────────────────────┼────────────────────┼────────────────────┤
  │ Complete?            │ Yes (closed set)   │ Yes                │
  ├──────────────────────┼────────────────────┼────────────────────┤
  │ Optimal?             │ No                 │ Yes                │
  ├──────────────────────┼────────────────────┼────────────────────┤
  │ Nodes Expanded       │ {gbfs_m['nodes_expanded']:<18} │ {astar_m['nodes_expanded']:<18} │
  ├──────────────────────┼────────────────────┼────────────────────┤
  │ Runtime (ms)         │ {gbfs_m['runtime_ms']:<18.4f} │ {astar_m['runtime_ms']:<18.4f} │
  ├──────────────────────┼────────────────────┼────────────────────┤
  │ Memory Usage         │ {gbfs_m['memory_usage']:<18} │ {astar_m['memory_usage']:<18} │
  ├──────────────────────┼────────────────────┼────────────────────┤
  │ Total Path Cost      │ {gbfs_m['total_path_cost']:<18} │ {astar_m['total_path_cost']:<18} │
  ├──────────────────────┼────────────────────┼────────────────────┤
  │ Path Length          │ {gbfs_m['path_length']:<18} │ {astar_m['path_length']:<18} │
  └──────────────────────┴────────────────────┴────────────────────┘""")

        print(f"""
  Analysis:
  • A* is COMPLETE: Guaranteed to find a path if one exists.
  • A* is OPTIMAL: Finds the lowest-cost path with admissible heuristic.
  • GBFS is COMPLETE with closed set (for finite graphs) but NOT optimal.
  • GBFS is generally FASTER as it only uses h(n) for expansion ordering.
  • A* explores more nodes but guarantees optimality.
  • GBFS may find a sub-optimal path faster due to greedy expansion.
""")

        # ====================================================
        # SECTION 8: PEAS Components
        # ====================================================
        print_peas()

        # ====================================================
        # SECTION 9: Complexity Analysis
        # ====================================================
        print_complexity_analysis()

        # ====================================================
        # SECTION 10: Heuristic Function Analysis
        # ====================================================
        print_section("HEURISTIC FUNCTION ANALYSIS")

        h1_cost = result_h1['metrics']['total_path_cost'] if result_h1['path'] else float('inf')
        h2_cost = result_h2['metrics']['total_path_cost'] if result_h2['path'] else float('inf')
        h1_len = result_h1['metrics']['path_length'] if result_h1['path'] else 0
        h2_len = result_h2['metrics']['path_length'] if result_h2['path'] else 0

        print(f"""
  Comparison of Heuristic Functions:
  ─────────────────────────────────────────────────────────────────
  
  h1 (Euclidean Distance):
    • Path Cost: {h1_cost}
    • Path Length: {h1_len} moves
    • Nodes Expanded: {result_h1['metrics']['nodes_expanded']}
    • Characteristics: Simple geometric distance, does not account
      for obstacles or terrain costs in the path.
    
  h2 (Bounding-Box Risk-Weighted):
    • Path Cost: {h2_cost}
    • Path Length: {h2_len} moves
    • Nodes Expanded: {result_h2['metrics']['nodes_expanded']}
    • Characteristics: Future-aware heuristic that considers the
      density of obstacles between current position and goal.
      Multiplies Manhattan distance by average cell cost in the
      bounding box.
  
  Conclusion:
  """)
        if h1_cost <= h2_cost:
            print("    h1 (Euclidean) achieves a lower or equal path cost.")
        else:
            print("    h2 (Bounding-Box) achieves a lower path cost.")

        if result_h1['metrics']['nodes_expanded'] <= result_h2['metrics']['nodes_expanded']:
            print("    h1 expands fewer or equal nodes (more efficient search).")
        else:
            print("    h2 expands fewer nodes (more efficient search).")

        print("""
    h2 is generally better for environments with clustered obstacles
    as it accounts for risk density. h1 is simpler but may lead the
    agent through hazardous regions it cannot foresee.
""")

        # ====================================================
        # SECTION 11: Conclusion
        # ====================================================
        print_section("CONCLUSION")
        print("""
  GBFS Algorithm Performance Analysis:
  ─────────────────────────────────────────────────────────────────
  
  Optimality:
  • GBFS is NOT optimal. It only considers h(n) for node selection,
    ignoring the actual cost g(n) accumulated so far. This means it
    may find a path that appears promising heuristically but has a
    higher total transition cost than the optimal path.
  • In this environment, GBFS may route through Weather Hazards if
    they appear closer to the goal heuristically.
  
  Completeness:
  • GBFS with a closed set (explored set) IS complete for finite
    graphs. Since our 8×8 grid is finite, GBFS will find a path if
    one exists.
  • Without a closed set, GBFS can get stuck in infinite loops.
  
  Practical Performance:
  • GBFS is faster than A* in practice for this problem as it
    expands fewer nodes by greedily pursuing the most promising
    direction.
  • The trade-off is that the path found may not be optimal.
  • For real-time drone navigation, GBFS may be preferred when
    speed is critical, while A* is preferred when fuel/cost
    optimization is paramount.
  
  Recommendation:
  • Use A* when optimality is required (mission-critical paths).
  • Use GBFS when fast response time is needed and sub-optimality
    is acceptable.
  • h2 provides better obstacle awareness than h1, making it
    more suitable for environments with clustered hazards.
""")

    sys.stdout = original_stdout

    # ====================================================
    # VISUALIZATIONS (matplotlib - separate from text output)
    # ====================================================
    print("\nGenerating visualizations...")

    # 1. Initial Grid
    visualize_grid(grid, title="Initial Environment Grid - Defence Drone",
                   save_path=os.path.join(script_dir, 'grid_initial.png'))

    # 2. GBFS h1 path
    if result_h1['path']:
        visualize_grid(grid, path=result_h1['path'],
                       explored=result_h1['explored'],
                       title="GBFS Path - Heuristic h1 (Euclidean)",
                       save_path=os.path.join(script_dir, 'grid_gbfs_h1.png'))

    # 3. GBFS h2 path
    if result_h2['path']:
        visualize_grid(grid, path=result_h2['path'],
                       explored=result_h2['explored'],
                       title="GBFS Path - Heuristic h2 (Bounding-Box)",
                       save_path=os.path.join(script_dir, 'grid_gbfs_h2.png'))

    # 4. A* path
    if result_astar['path']:
        visualize_grid(grid, path=result_astar['path'],
                       explored=result_astar['explored'],
                       title="A* Path - Heuristic h1 (Euclidean)",
                       save_path=os.path.join(script_dir, 'grid_astar_h1.png'))

    # 5. Heuristic progression
    if result_h1['h_values_along_path'] and result_h2['h_values_along_path']:
        visualize_heuristic_progression(
            result_h1['h_values_along_path'],
            result_h2['h_values_along_path'],
            title="Heuristic Values Along Search Path"
        )

    # 6. Heuristic vs Time
    if result_h1['h_values_along_path'] and result_h2['h_values_along_path']:
        visualize_heuristic_vs_time(
            result_h1['h_values_along_path'],
            result_h2['h_values_along_path'],
            result_h1['metrics']['runtime_ms'],
            result_h2['metrics']['runtime_ms']
        )

    # 7. Comparison charts
    visualize_comparison(all_metrics, "Algorithm & Heuristic Comparison")

    # 8. Heuristic value heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Heuristic Value Heatmaps", fontsize=14, fontweight='bold')

    # h1 heatmap
    h1_grid = np.zeros((grid.rows, grid.cols))
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.get_cell(r, c) == CELL_NOFLY:
                h1_grid[r][c] = -1  # Mark as blocked
            else:
                h1_grid[r][c] = h1_euclidean((r, c), grid.goal)

    masked_h1 = np.ma.masked_where(h1_grid == -1, h1_grid)
    im1 = ax1.imshow(masked_h1, cmap='YlOrRd_r', interpolation='nearest')
    ax1.set_title('h1: Euclidean Distance Heatmap')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im1, ax=ax1)
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.get_cell(r, c) == CELL_NOFLY:
                ax1.text(c, r, 'N', ha='center', va='center', fontsize=8, color='white')
            else:
                ax1.text(c, r, f'{h1_grid[r][c]:.1f}', ha='center', va='center', fontsize=7)

    # h2 heatmap
    h2_grid = np.zeros((grid.rows, grid.cols))
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.get_cell(r, c) == CELL_NOFLY:
                h2_grid[r][c] = -1
            else:
                h2_grid[r][c] = h2_bounding_box((r, c), grid.goal, grid)

    masked_h2 = np.ma.masked_where(h2_grid == -1, h2_grid)
    im2 = ax2.imshow(masked_h2, cmap='YlOrRd_r', interpolation='nearest')
    ax2.set_title('h2: Bounding-Box Risk-Weighted Heatmap')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    plt.colorbar(im2, ax=ax2)
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.get_cell(r, c) == CELL_NOFLY:
                ax2.text(c, r, 'N', ha='center', va='center', fontsize=8, color='white')
            else:
                ax2.text(c, r, f'{h2_grid[r][c]:.1f}', ha='center', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'heuristic_heatmaps.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 9. Nodes explored comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    algorithms = ['GBFS h1', 'GBFS h2', 'A* h1']
    explored_counts = [
        result_h1['metrics']['nodes_expanded'],
        result_h2['metrics']['nodes_expanded'],
        result_astar['metrics']['nodes_expanded'],
    ]
    bars = ax.bar(algorithms, explored_counts, color=['#2196F3', '#FF9800', '#4CAF50'])
    ax.set_title('Nodes Expanded Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Nodes Expanded')
    ax.set_xlabel('Algorithm')
    for bar, val in zip(bars, explored_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'nodes_explored.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Output written to: {output_file}")
    print("✓ Visualizations saved as PNG files in the script directory.")
    print("\nExecution complete.")


if __name__ == "__main__":
    main()
