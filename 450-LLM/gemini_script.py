from google import genai
import pygame
import numpy as np
import time
import serial
from collections import deque
import csv
import os

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
client = genai.Client(api_key="API Key")

def get_shape_cells(shape):
    prompt = (
        f"Generate a {N_ROWS}x{N_COLS} matrix of 1s and 0s representing a {shape}. "
        f"0s are background and 1s are the {shape}. "
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


# ================== Main ==================
def main(cells, D, target_mask, shape):
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
        draw_text(screen, f"Shape: {shape} | dir={direction} (-1: pos/blue, 1: neg/red) | state: {state}", (20, 8), size=22, color=(200, 220, 200))
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
    shape = input("Enter shape name: ").strip()
    try:
        cells = get_shape_cells(shape)
        if not cells:
            print("No cells parsed from Gemini response. Check the output above.")
        else:
            print(f"Parsed {len(cells)} target cells: {cells}")
            target_mask = build_target_mask(N_ROWS, N_COLS, cells)
            D = manhattan_distance_to_targets(N_ROWS, N_COLS, target_mask)
            main(cells, D, target_mask, shape)
    except Exception as e:
        print(f"Failed: {e}")
