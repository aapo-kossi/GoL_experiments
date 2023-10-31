import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba_progress import ProgressBar
import sys
from numba import njit, prange

NEIGHBORS = np.array(
    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)),
    dtype=np.int64,
)


def get_colors(grid, neighbor_counts):
    grid_size = grid.shape[0]
    # Define colors based on the specified scheme
    colors = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)  # RGB format
    unsatisfied_dead_color = [0, 255, 0]  # Green
    satisfied_alive_color = [255, 255, 255]  # White
    unsatisfied_alive_color = [255, 0, 0]  # Bright Red

    unsatisfied_alive_mask, unsatisfied_dead_mask = get_unsatisfied(
        grid, neighbor_counts
    )

    # Apply colors to the grid based on the cell state
    colors[unsatisfied_dead_mask.astype(bool)] = unsatisfied_dead_color
    colors[unsatisfied_alive_mask.astype(bool)] = unsatisfied_alive_color
    colors[(grid & ~unsatisfied_alive_mask).astype(bool)] = satisfied_alive_color
    return colors


@njit
def calculate_nn(grid):
    rows, cols = grid.shape
    alive_neighbors = np.zeros_like(grid, dtype=np.uint8)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            shifted_grid = np.zeros_like(grid, dtype=np.uint8)
            for row in range(rows):
                for col in range(cols):
                    shifted_row = (row + i) % rows
                    shifted_col = (col + j) % cols
                    shifted_grid[shifted_row, shifted_col] = grid[row, col]
            alive_neighbors += shifted_grid
    return alive_neighbors


@njit(parallel=True)
def update_neighbors(
    neighbor_counts,
    x_alive,
    x_dead,
):
    L = neighbor_counts.shape[0]
    for i in prange(8):
        alive_n = (NEIGHBORS[i] + x_alive) % L
        dead_n = (NEIGHBORS[i] + x_dead) % L
        neighbor_counts[alive_n[0], alive_n[1]] -= 1
        neighbor_counts[dead_n[0], dead_n[1]] += 1


@njit(parallel=True)
def get_unsatisfied(grid, neighbor_counts):
    return grid & ((neighbor_counts < 2) | (neighbor_counts > 3)), ~grid & (
        neighbor_counts == 3
    )


@njit
def conservative_game_of_life(
    grid, steps, max_animation_steps, progress_proxy, animation_freq=1
):
    neighbor_counts_anim = np.zeros(
        (max_animation_steps,) + tuple(grid.shape), dtype=np.uint8
    )
    grids = np.full((max_animation_steps,) + tuple(grid.shape), False, dtype=np.bool_)
    neighbor_counts = calculate_nn(grid)
    neighbor_counts_anim[0] = neighbor_counts
    grids[0] = grid

    dt = 0
    animation_step = 1
    t = 0
    for _ in range(steps):
        progress_proxy.update(1)
        unsatisfied_alive_mask, unsatisfied_dead_mask = get_unsatisfied(
            grid, neighbor_counts
        )

        if not (np.any(unsatisfied_alive_mask) and np.any(unsatisfied_dead_mask)):
            # If there are no unsatisfied cells, the game has reached a final state.
            # The final state should be used as the figure, and last frame of the animation
            grids[animation_step - 1] = grid
            neighbor_counts_anim[animation_step - 1] = neighbor_counts
            break

        # sample single random cells from unsatisfied alive and dead cells
        # note: funnily enough, these calls to nonzero() constitute most of the algorithm runtime :))
        ua_i = unsatisfied_alive_mask.nonzero()
        ud_i = unsatisfied_dead_mask.nonzero()
        alive_unsatisfied = np.column_stack(ua_i)
        dead_unsatisfied = np.column_stack(ud_i)

        n_unsatisfied = alive_unsatisfied.shape[0] + dead_unsatisfied.shape[0]
        alive_cell = alive_unsatisfied[np.random.randint(alive_unsatisfied.shape[0])]
        dead_cell = dead_unsatisfied[np.random.randint(dead_unsatisfied.shape[0])]

        # swap the chosen alive and dead cell
        grid[alive_cell[0], alive_cell[1]] = 0
        grid[dead_cell[0], dead_cell[1]] = 1

        # Update unsatisfied cells around the modified cells
        update_neighbors(neighbor_counts, alive_cell, dead_cell)

        # animation stuff
        dt = 1 / n_unsatisfied
        if (animation_step < max_animation_steps) and (
            int((dt + t) * animation_freq) > int(t * animation_freq)
        ):
            grids[animation_step] = grid
            neighbor_counts_anim[animation_step] = neighbor_counts
            animation_step += 1

        t += dt
    return grids[:animation_step], neighbor_counts_anim[:animation_step], t


# Example usage
grid_size, p_dead, steps, ani_steps, ani_freq = sys.argv[1:]
grid_size = int(grid_size)
p_dead = float(p_dead)
steps = int(steps)
ani_steps = int(ani_steps)
ani_freq = float(ani_freq)
initial_grid = np.random.choice(
    [0, 1],
    size=(grid_size, grid_size),
    p=[p_dead, 1 - p_dead],
).astype(bool)

with ProgressBar(total=steps) as progress:
    animation_grid, animation_nn, t = conservative_game_of_life(
        initial_grid, steps, ani_steps, progress, animation_freq=ani_freq
    )
print(f"finished in {t} steps!")
print(f"Generated {len(animation_grid)} frames.")

# Create a figure and axis
fig1, ax1 = plt.subplots(figsize=(8, 8))
final_colors = get_colors(animation_grid[-1], animation_nn[-1])
ax1.imshow(final_colors)
ax1.set_axis_off()  # Hide the axis
fig1.tight_layout()

# Create animation window
fig, ax = plt.subplots(figsize=(8, 8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
initial_colors = get_colors(animation_grid[0], animation_nn[0])
im = ax.imshow(initial_colors, animated=True)
ax.set_axis_off()
fps = 20


def update(frame):
    colors = get_colors(animation_grid[frame], animation_nn[frame])
    im.set_array(colors)
    return [im]


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(animation_grid),
    interval=1000 / fps,
    blit=False,
    repeat=True,
)
writer = animation.FFMpegWriter(
        fps=fps, bitrate=320,
        extra_args=['-crf', '0', '-preset', 'veryslow', '-c:a', 'libmp3lame']
)
ani.save(
    f"videos/GoL_animation_L{grid_size}_pd{p_dead}_steps{steps}_frames{ani_steps}_framedt{ani_freq}.mp4",
    dpi=max(128, grid_size / 4),  # ensure that a pixel size is atleast two cells
    writer=writer,
)


plt.show()
