from google import genai
import pygame
import numpy as np
import time
import serial
from collections import deque
import csv
import os
import re
import sys

# Allow importing text_parsing from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_parsing import parse_command

# ================== Config ==================
SERIAL_PORTS = ['/dev/cu.usbmodem1020BA0ABA902']

# Polarity control: direction = 1 -> negative (red), -1 -> positive (blue)
direction = -1

HERD_PULSE_DT = 1       # seconds between band pulses
HERD_OVERLAP = True
HERD_OVERLAP_HOLD = 10  # seconds to hold overlap
SERIAL_BAUD = 115200

SCREEN_W, SCREEN_H = 800, 650
FPS = 30
N_ROWS, N_COLS = 16, 16

BG_COLOR       = (30, 30, 30)
GRID_COLOR     = (100, 100, 100)
POS_COLOR      = (0, 0, 255)
NEG_COLOR      = (255, 0, 0)
BUTTON_BG      = (60, 60, 60)
BUTTON_BG_HOVER= (90, 90, 90)
TEXT_COLOR     = (255, 255, 255)
TABLE_BG       = (15, 15, 15)
TABLE_GRID     = (70, 70, 70)
# ================== Config ==================


# ================== Gemini ==================
client = genai.Client(api_key="AIzaSyBVK0w9mb8KY-nZBJ5c33BcpxklctZg3UA")

_OPPOSITE_EDGE = {
    "right":    "left",
    "left":     "right",
    "up":       "bottom",
    "forward":  "bottom",
    "down":     "top",
    "backward": "top",
}

def get_shape_cells(shape, for_movement=None):
    """
    Ask Gemini for a binary shape bitmap.

    The shape is always small (4-6 cells wide) so movement is visible on the grid.
    - for_movement=None  → place the shape centered in the grid.
    - for_movement=<dir> → place the shape near the opposite edge so it has
                           room to travel; offset_cells_to_start enforces this
                           afterwards as a hard guarantee.
    """
    if for_movement:
        edge = _OPPOSITE_EDGE.get(for_movement, "left")
        placement = (
            f"positioned near the {edge} edge of the grid, "
            f"leaving most of the grid empty for movement"
        )
    else:
        placement = "centered in the grid"

    # If the shape is a single letter (e.g. "e") or "letter X", default to
    # the uppercase version unless the user explicitly said "lowercase".
    shape_label = shape
    single = re.fullmatch(r'[a-zA-Z]', shape)
    letter_word = re.fullmatch(r'letter\s+([a-zA-Z])', shape, re.IGNORECASE)
    if single or letter_word:
        letter = (letter_word.group(1) if letter_word else shape).upper()
        shape_label = f"uppercase letter {letter}"

    prompt = (
        f"Generate a {N_ROWS}x{N_COLS} matrix of 1s and 0s representing a small, "
        f"compact {shape_label} {placement}. "
        f"The shape must be approximately 4 to 6 cells wide — do NOT fill the whole grid. "
        f"0s are background and 1s are the {shape_label}. "
        f"Output only the matrix, one row per line, values separated by spaces. Nothing else."
    )
    print("Sending request to Gemini...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt]
    )
    print(f"Gemini response:\n{response.text}")
    return parse_matrix(response.text)

def parse_matrix(text):
    """Parse Gemini text output into a list of (row, col) cells where value == 1."""
    cells = []
    for r, line in enumerate(text.strip().splitlines()):
        for c, val in enumerate(line.split()):
            if val.strip('[](),') == '1':
                cells.append((r, c))
    return cells
# ================== Gemini ==================


# ================== Grid helpers ==================
def create_grid(n, m):
    return np.zeros((n, m, 3), dtype=float)

def get_dynamic_tile_size(n, m):
    tile_size = min((SCREEN_W - 20) // m, (SCREEN_H - 180) // n)
    return max(4, tile_size)

def draw_text(surface, text, pos, size=28, color=TEXT_COLOR, center=False):
    font = pygame.font.SysFont(None, size)
    img = font.render(str(text), True, color)
    rect = img.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(img, rect)
    return rect

def draw_grid(surface, grid):
    n, m, _ = grid.shape
    tile = get_dynamic_tile_size(n, m)
    grid_w, grid_h = m * tile, n * tile
    x0 = (SCREEN_W - grid_w) // 2
    y0 = 20

    for i in range(n):
        for j in range(m):
            # even rows are offset by half a tile (matching gui_manual_control layout)
            if i % 2 == 0:
                x = x0 + (j + 0.5) * tile
            else:
                x = x0 + j * tile
            y = y0 + i * tile
            pos_val, neg_val, _ = grid[i, j]
            if pos_val > 0:
                alpha = int(np.clip(25.5 * pos_val, 25, 255))
                surf = pygame.Surface((tile - 2, tile - 2), pygame.SRCALPHA)
                surf.fill((*POS_COLOR, alpha))
                surface.blit(surf, (x, y))
            elif neg_val > 0:
                alpha = int(np.clip(25.5 * neg_val, 25, 255))
                surf = pygame.Surface((tile - 2, tile - 2), pygame.SRCALPHA)
                surf.fill((*NEG_COLOR, alpha))
                surface.blit(surf, (x, y))
            else:
                pygame.draw.rect(surface, GRID_COLOR, (x, y, tile - 2, tile - 2))

    return (x0, y0 + grid_h, grid_w, tile, y0)

def draw_table(surface, grid, pos_x, pos_y, width):
    n, m, _ = grid.shape
    table_top = pos_y + 10
    cell_h = 28
    cell_w = max(width // max(m, 1), 28)
    pygame.draw.rect(surface, TABLE_BG, (pos_x, table_top, m * cell_w, n * cell_h))
    for i in range(n):
        draw_text(surface, i, (pos_x - 20, table_top + i * cell_h + 3), size=16, color=(200, 220, 180))
        for j in range(m):
            cx = pos_x + j * cell_w + cell_w // 2
            ry = table_top + i * cell_h + cell_h // 2
            pos_val = int(round(grid[i, j][0]))
            neg_val = int(round(grid[i, j][1]))
            draw_text(surface, f"{pos_val}/{neg_val}", (cx, ry), size=16, center=True)
    for i in range(n + 1):
        y = table_top + i * cell_h
        pygame.draw.line(surface, TABLE_GRID, (pos_x, y), (pos_x + m * cell_w, y))
    for j in range(m + 1):
        x = pos_x + j * cell_w
        pygame.draw.line(surface, TABLE_GRID, (x, table_top), (x, table_top + n * cell_h))

def get_output_matrix(grid):
    arr = np.round(grid[:, :, :2]).astype(int).reshape(-1, 2)
    flat = arr.flatten()
    group = 16
    num_rows = (len(flat) + group - 1) // group
    A = np.zeros((num_rows, group), dtype=int)
    for idx, val in enumerate(flat):
        A[idx // group, idx % group] = val
    return A

def try_open_serial():
    for port in SERIAL_PORTS:
        try:
            ser = serial.Serial(port, SERIAL_BAUD, timeout=0)
            time.sleep(1.5)
            print(f"Serial opened: {port}")
            return ser
        except Exception:
            continue
    print("Serial not found; running without serial output.")
    return None

def send_matrix_over_serial(A, ser):
    if ser is None:
        return
    try:
        data = ",".join(str(int(v)) for v in A.flatten()) + "\n"
        ser.write(data.encode('utf-8'))
    except Exception:
        pass

def clear_all_pwm(grid, ser):
    grid[:, :, :2] = 0
    A = get_output_matrix(grid)
    send_matrix_over_serial(A, ser)
# ================== Grid helpers ==================


# ================== Distance transform ==================
def build_target_mask(n, m, cells):
    mask = np.zeros((n, m), dtype=bool)
    for (i, j) in cells:
        if 0 <= i < n and 0 <= j < m:
            mask[i, j] = True
    return mask

def manhattan_distance_to_targets(n, m, target_mask):
    D = np.full((n, m), np.inf, dtype=float)
    q = deque()
    ti, tj = np.where(target_mask)
    for i, j in zip(ti, tj):
        D[i, j] = 0
        q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < n and 0 <= nj < m and D[ni, nj] > D[i, j] + 1:
                D[ni, nj] = D[i, j] + 1
                q.append((ni, nj))
    D[np.isinf(D)] = 0
    return D.astype(int)

def activate_band(grid, D, k, direction):
    grid[:, :, :2] = 0
    sel = (D == k)
    if direction == -1:
        grid[sel, 0] = 10.0
    else:
        grid[sel, 1] = 10.0

def activate_band_with_overlap(grid, D, k, direction):
    activate_band(grid, D, k, direction)
    if k > 0:
        inner = (D == (k - 1))
        if direction == -1:
            grid[inner, 0] = 10.0
        else:
            grid[inner, 1] = 10.0

def activate_targets_only(grid, target_mask, direction):
    grid[:, :, :2] = 0
    if direction == -1:
        grid[target_mask, 0] = 10.0
    else:
        grid[target_mask, 1] = 10.0
# ================== Distance transform ==================


# ================== Movement ==================
# Direction strings come from text_parsing.parse_command()["direction"].
# Grid coordinates: row 0 is top, so "up" decrements the row index.
_DIRECTION_VECTORS = {
    "up":               (-1,  0),
    "forward":          (-1,  0),
    "down":             ( 1,  0),
    "backward":         ( 1,  0),
    "left":             ( 0, -1),
    "right":            ( 0,  1),
}

def direction_to_vector(direction_str):
    """Return a (dr, dc) unit grid vector for a direction string from text_parsing."""
    return _DIRECTION_VECTORS.get(direction_str, (0, 0))


def is_movement_command(params):
    """Return True when text_parsing params describe a swarm translation command."""
    motion_words = {"move", "translate", "shift"}
    return (
        params.get("motion") in motion_words
        and params.get("direction") in _DIRECTION_VECTORS
    )


def shift_target_mask(target_mask, dr, dc):
    """
    Shift every True cell in target_mask by (dr, dc).
    Cells that would land outside the grid are discarded so the
    shape simply stops moving when it reaches a boundary.
    Returns the new mask (same shape as input).
    """
    n, m = target_mask.shape
    new_mask = np.zeros((n, m), dtype=bool)
    rows, cols = np.where(target_mask)
    for r, c in zip(rows, cols):
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < m:
            new_mask[nr, nc] = True
    return new_mask


def steps_to_far_edge(mask, move_vec, n_rows, n_cols, end_margin=2):
    """
    Return the number of single-cell steps needed to bring the shape's
    leading edge to `end_margin` cells from the far grid boundary.

      right  (dc=+1): travel until max_col  == n_cols-1-end_margin
      left   (dc=-1): travel until min_col  == end_margin
      down   (dr=+1): travel until max_row  == n_rows-1-end_margin
      up     (dr=-1): travel until min_row  == end_margin
    """
    dr, dc = move_vec
    rows, cols = np.where(mask)
    if not rows.size:
        return 0
    if dc == 1:
        steps = (n_cols - 1 - end_margin) - int(cols.max())
    elif dc == -1:
        steps = int(cols.min()) - end_margin
    elif dr == 1:
        steps = (n_rows - 1 - end_margin) - int(rows.max())
    elif dr == -1:
        steps = int(rows.min()) - end_margin
    else:
        steps = 0
    return max(0, steps)


def offset_cells_to_start(cells, direction, n_rows, n_cols, margin=1):
    """
    Translate the cell list so the shape starts near the edge *opposite* to
    the direction of travel, leaving `margin` empty cells at that edge.

      direction right   → left edge   (min col  = margin)
      direction left    → right edge  (max col  = n_cols-1-margin)
      direction up/fwd  → bottom edge (max row  = n_rows-1-margin)
      direction down/bk → top edge    (min row  = margin)

    Cells that would fall outside the grid after translation are dropped.
    """
    if not cells:
        return cells

    rows = [r for r, c in cells]
    cols = [c for r, c in cells]

    dr, dc = 0, 0
    if direction == "right":
        dc = margin - min(cols)
    elif direction == "left":
        dc = (n_cols - 1 - margin) - max(cols)
    elif direction in ("up", "forward"):
        dr = (n_rows - 1 - margin) - max(rows)
    elif direction in ("down", "backward"):
        dr = margin - min(rows)

    return [
        (r + dr, c + dc)
        for r, c in cells
        if 0 <= r + dr < n_rows and 0 <= c + dc < n_cols
    ]


def move_swarm(grid, target_mask, move_vec, num_steps, step_delay, ser):
    """
    Hold the swarm in its current shape and translate it one grid cell at a
    time in move_vec = (dr, dc).

    For each step:
      1. Compute the shifted target mask.
      2. Run a mini-herd (distance-transform bands) toward the new positions
         so the robots follow the advancing magnetic pattern.
      3. Activate only the new target cells to lock the shape in place.
      4. Pause for step_delay seconds before the next step.

    Parameters
    ----------
    grid        : ndarray  current PWM grid
    target_mask : ndarray  boolean mask of the formed shape
    move_vec    : (dr, dc) unit step direction
    num_steps   : int      how many grid cells to travel
    step_delay  : float    seconds to hold each intermediate position
    ser         : serial   serial port (may be None)

    Returns
    -------
    Updated target_mask after all steps completed.
    """
    dr, dc = move_vec
    current_mask = target_mask.copy()

    for step in range(num_steps):
        next_mask = shift_target_mask(current_mask, dr, dc)

        # If the entire shape has moved off-grid, stop early.
        if not np.any(next_mask):
            print(f"[move_swarm] Swarm reached grid boundary at step {step}. Stopping.")
            break

        # Mini-herd: sweep distance-transform bands from outermost inward
        # so the magnetic wave pulls robots into the new position.
        D_next = manhattan_distance_to_targets(N_ROWS, N_COLS, next_mask)
        Dmax_next = int(D_next.max())
        for k in range(Dmax_next, -1, -1):
            activate_band_with_overlap(grid, D_next, k, direction)
            A = get_output_matrix(grid)
            send_matrix_over_serial(A, ser)
            time.sleep(HERD_PULSE_DT)

        # Lock the shape at its new position.
        activate_targets_only(grid, next_mask, direction)
        A = get_output_matrix(grid)
        send_matrix_over_serial(A, ser)
        time.sleep(step_delay)

        current_mask = next_mask
        print(f"[move_swarm] Step {step + 1}/{num_steps} complete.")

    return current_mask
# ================== Movement ==================


# ================== Main ==================
def main(cells, D, target_mask, shape, move_params=None):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"Swarm Control - {shape}")
    clock = pygame.time.Clock()

    grid = create_grid(N_ROWS, N_COLS)

    csv_folder = f"{shape.replace(' ', '_')}_frames"
    os.makedirs(csv_folder, exist_ok=True)

    frame_idx = 0

    btn_w, btn_h = 160, 48
    start_rect = pygame.Rect(0, 0, btn_w, btn_h)
    start_rect.center = (SCREEN_W // 2 - 100, SCREEN_H - 40)
    stop_rect  = pygame.Rect(0, 0, btn_w, btn_h)
    stop_rect.center  = (SCREEN_W // 2 + 100, SCREEN_H - 40)

    running = True
    started = False
    ser = try_open_serial()
    last_send = 0.0

    Dmax = int(D.max())
    state = "idle"
    band_k = None
    band_last_t = 0.0
    overlap_until = 0.0

    # Non-blocking move sub-state (driven frame-by-frame in the main loop)
    move_step_idx     = 0      # steps completed so far
    move_current_mask = None   # shape mask at the current resting position
    move_last_step_t  = 0.0    # wall-clock time of the last step

    # Resolve movement parameters from text_parsing output (if provided).
    move_vec = (0, 0)
    move_steps = 0
    move_step_delay = 1.0
    if move_params is not None and is_movement_command(move_params):
        move_vec = direction_to_vector(move_params["direction"])
        # Derive number of steps from speed: slow->1 step, fast->5, default->3
        spd = move_params.get("speed") or 10  # __main__ always sets a default
        if spd <= 5:
            move_steps = 1
        elif spd <= 15:
            move_steps = 3
        elif spd < 50:
            move_steps = 5
        else:
            move_steps = 8
        # Pause between steps: 10/speed gives 2.0 s at slow, 1.0 s at default, 0.2 s at fast
        move_step_delay = max(0.2, 10.0 / spd)
        print(f"[main] Movement queued: direction={move_params['direction']} "
              f"vec={move_vec} steps={move_steps} delay={move_step_delay:.2f}s")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    started = True
                    state = "herd"
                    band_k = Dmax
                    band_last_t = time.time() - HERD_PULSE_DT
                    overlap_until = 0.0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if start_rect.collidepoint(mx, my):
                    started = True
                    state = "herd"
                    band_k = Dmax
                    band_last_t = time.time() - HERD_PULSE_DT
                    overlap_until = 0.0
                elif stop_rect.collidepoint(mx, my):
                    clear_all_pwm(grid, ser)
                    running = False

        now = time.time()

        if started:
            if state == "herd":
                if (now - band_last_t) >= HERD_PULSE_DT:
                    band_last_t = now
                    if HERD_OVERLAP:
                        activate_band_with_overlap(grid, D, band_k, direction)
                        overlap_until = now + HERD_OVERLAP_HOLD
                    else:
                        activate_band(grid, D, band_k, direction)
                    band_k -= 1
                    if band_k < 0:
                        state = "form"
                        activate_targets_only(grid, target_mask, direction)
                        band_last_t = now
                else:
                    if HERD_OVERLAP and now < overlap_until and band_k is not None and (band_k + 1) >= 0:
                        activate_band_with_overlap(grid, D, band_k + 1, direction)

            elif state == "form":
                activate_targets_only(grid, target_mask, direction)
                # Once the shape is formed, kick off any queued movement.
                if move_steps > 0:
                    move_step_idx     = 0
                    move_current_mask = target_mask.copy()
                    move_last_step_t  = now  # first step fires after one delay
                    # Override step count: travel exactly to 2 cells from far edge
                    move_steps = steps_to_far_edge(
                        target_mask, move_vec, N_ROWS, N_COLS, end_margin=2
                    )
                    print(f"[move] Steps to edge: {move_steps}")
                    if move_steps > 0:
                        state = "move"

            elif state == "move":
                # Each step: shift every active cell one position in move_vec,
                # then deactivate the cells that were left behind.
                if (now - move_last_step_t) >= move_step_delay:
                    move_last_step_t = now
                    next_mask = shift_target_mask(move_current_mask, move_vec[0], move_vec[1])
                    if not np.any(next_mask):
                        print("[move] Boundary reached. Stopping.")
                        move_steps = 0
                        state = "form"
                    else:
                        activate_targets_only(grid, next_mask, direction)
                        move_current_mask = next_mask
                        target_mask       = next_mask   # keep in sync for CSV
                        move_step_idx    += 1
                        print(f"[move] Step {move_step_idx}/{move_steps} complete.")
                        if move_step_idx >= move_steps:
                            move_steps = 0
                            state = "form"

            if (now - last_send) >= 0.5:
                A = get_output_matrix(grid)
                send_matrix_over_serial(A, ser)
                last_send = now
                # Save to Frame Sheet

                binary_grid = ((grid[:, :, 0] > 0) | (grid[:, :, 1] > 0)).astype(int) * 7

                # Rotate 90 degrees clockwise to correct display orientation
                binary_grid = np.rot90(binary_grid, k=-1)

                # Embed 16x16 working grid into bottom-left of a 32x32 output grid
                full_grid = np.zeros((32, 32), dtype=int)
                full_grid[0:16, 0:16] = binary_grid

                csv_path = os.path.join(csv_folder, f"frame_{frame_idx:04d}.csv")
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(full_grid.tolist())
                frame_idx += 1

        screen.fill(BG_COLOR)
        if state == "move" and move_params:
            move_info = f" | {move_params['direction']} step {move_step_idx + 1}/{move_steps}"
        elif move_params and is_movement_command(move_params) and move_steps == 0 and move_step_idx > 0:
            move_info = f" | {move_params['direction']} done ({move_step_idx} steps)"
        else:
            move_info = ""
        draw_text(screen, f"Shape: {shape} | dir={direction} | state: {state}{move_info}", (20, 8), size=22, color=(200, 220, 200))
        x0, bot_y, grid_w, tile, _ = draw_grid(screen, grid)
        draw_table(screen, grid, x0, bot_y, grid_w)

        for rect, label in [(start_rect, "Start / Space"), (stop_rect, "Stop")]:
            hover = rect.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(screen, BUTTON_BG_HOVER if hover else BUTTON_BG, rect, border_radius=10)
            draw_text(screen, label, rect.center, size=26, center=True)

        pygame.display.flip()
        clock.tick(FPS)

    if ser is not None:
        try:
            clear_all_pwm(grid, ser)
            ser.close()
        except Exception:
            pass

    print(f"Saved {frame_idx} frames to folder: {csv_folder}")

    pygame.quit()


if __name__ == "__main__":
    # Single unified command: "make a circle and move it to the right"
    # or just a shape: "triangle"
    user_input = input("Enter command: ").strip()

    params = parse_command(user_input)

    # --- Shape name ---
    shape = params.get("shape")
    if not shape:
        shape = input("Shape not detected. Enter shape name: ").strip()

    # --- Movement params ---
    # Apply defaults: speed=None → 10 mm/s (3 steps, 0.5 s hold)
    if params.get("speed") is None:
        params["speed"] = 10

    move_params = params if is_movement_command(params) else None

    if move_params:
        print(f"Shape: {shape} | direction: {move_params['direction']} | "
              f"speed: {move_params['speed']} mm/s | motion: {move_params['motion']}")
    else:
        print(f"Shape: {shape} | no movement command — converge only")

    try:
        cells = get_shape_cells(
            shape,
            for_movement=move_params["direction"] if move_params else None
        )
        if not cells:
            print("No cells parsed from Gemini response. Check the output above.")
        else:
            # Hard-enforce starting position via offset in case Gemini didn't
            # follow the placement instruction precisely.
            if move_params:
                cells = offset_cells_to_start(
                    cells, move_params["direction"], N_ROWS, N_COLS
                )
                print(f"Offset to starting edge. {len(cells)} cells after clipping.")

            print(f"Parsed {len(cells)} target cells.")
            target_mask = build_target_mask(N_ROWS, N_COLS, cells)
            D = manhattan_distance_to_targets(N_ROWS, N_COLS, target_mask)
            main(cells, D, target_mask, shape, move_params)
    except Exception as e:
        print(f"Failed: {e}")
