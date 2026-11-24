"""Turns a picture into terminal characters, and only sends the ones that changed.

The picture always has 2x4 square pixels per character cell (cells are twice as tall as wide); each mode spends them
differently.
"""

import numpy as np
from numba import njit, prange

MODES = ("quad", "half", "braille", "ascii")

# Quadrant glyphs by bitmask: 1 upper left, 2 upper right, 4 lower left, 8 lower right.
QUADS = np.array([ord(c) for c in " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"], np.uint32)
ASCII = np.array([ord(c) for c in " .`':,;-~+=<?c7oLxneaS3hkdDRNGHB8W#@"], np.uint32)
BRAILLE_BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], np.uint32)
DITHER = np.array([[0.1, 0.6], [0.85, 0.35], [0.2, 0.7], [0.95, 0.45]], np.float32)


@njit(cache=True, inline="always")
def _pack(r, g, b):
    return (np.uint32(min(max(r, 0.0), 255.0)) << 16) | (np.uint32(min(max(g, 0.0), 255.0)) << 8) | np.uint32(min(max(b, 0.0), 255.0))


@njit(cache=True, fastmath=True, parallel=True)
def to_cells(rgb, mode, char, fg, bg):
    """Fill the char, fg and bg grids from the picture."""
    rows, cols = char.shape
    for row in prange(rows):
        q = np.zeros((4, 3), np.float32)
        for col in range(cols):
            y, x = row * 4, col * 2
            if mode == 0:  # quad: four pixels, two colours; try every split and keep the closest
                for k in range(4):
                    for c in range(3):
                        q[k, c] = 0.5 * (rgb[y + (k >> 1) * 2, x + (k & 1), c] + rgb[y + (k >> 1) * 2 + 1, x + (k & 1), c])
                best, best_err = 15, 1e30
                fr = fgc = fb = br = bgc = bb = 0.0
                for m in range(15, 7, -1):  # the other eight are the same splits with the colours swapped
                    n1 = (m & 1) + (m >> 1 & 1) + (m >> 2 & 1) + (m >> 3 & 1)
                    n0 = 4 - n1
                    err = 0.0
                    s1r = s1g = s1b = s0r = s0g = s0b = 0.0
                    for k in range(4):
                        if m >> k & 1:
                            s1r += q[k, 0]; s1g += q[k, 1]; s1b += q[k, 2]
                        else:
                            s0r += q[k, 0]; s0g += q[k, 1]; s0b += q[k, 2]
                    s1r /= n1; s1g /= n1; s1b /= n1
                    if n0:
                        s0r /= n0; s0g /= n0; s0b /= n0
                    for k in range(4):
                        if m >> k & 1:
                            err += (q[k, 0] - s1r) ** 2 + (q[k, 1] - s1g) ** 2 + (q[k, 2] - s1b) ** 2
                        else:
                            err += (q[k, 0] - s0r) ** 2 + (q[k, 1] - s0g) ** 2 + (q[k, 2] - s0b) ** 2
                    if err < best_err - 12:  # not worth a glyph for a difference nobody can see
                        best, best_err = m, err
                        fr, fgc, fb, br, bgc, bb = s1r, s1g, s1b, s0r, s0g, s0b
                if best == 15:
                    char[row, col], fg[row, col], bg[row, col] = 32, 0, _pack(fr, fgc, fb)
                else:
                    char[row, col], fg[row, col], bg[row, col] = QUADS[best], _pack(fr, fgc, fb), _pack(br, bgc, bb)
                continue
            t0 = t1 = t2 = b0 = b1 = b2 = 0.0
            for i in range(2):
                for j in range(2):
                    t0 += rgb[y + i, x + j, 0]; t1 += rgb[y + i, x + j, 1]; t2 += rgb[y + i, x + j, 2]
                    b0 += rgb[y + 2 + i, x + j, 0]; b1 += rgb[y + 2 + i, x + j, 1]; b2 += rgb[y + 2 + i, x + j, 2]
            if mode == 1:  # half
                char[row, col], fg[row, col], bg[row, col] = 0x2580, _pack(t0 / 4, t1 / 4, t2 / 4), _pack(b0 / 4, b1 / 4, b2 / 4)
                continue
            r, g, b = (t0 + b0) / 8, (t1 + b1) / 8, (t2 + b2) / 8
            top = max(max(r, g), max(b, 1.0))
            lift = min(255.0 / top, 1.8)  # the gaps between dots and letters darken everything, so brighten the ink
            if mode == 2:  # braille
                bits = np.uint32(0)
                for i in range(4):
                    for j in range(2):
                        light = 0.3 * rgb[y + i, x + j, 0] + 0.59 * rgb[y + i, x + j, 1] + 0.11 * rgb[y + i, x + j, 2]
                        if light > 8 and (light / 255) ** 0.6 > DITHER[i, j] * 0.95:
                            bits |= BRAILLE_BITS[i, j]
                char[row, col] = 0x2800 + bits
            else:
                light = 0.3 * r + 0.59 * g + 0.11 * b
                char[row, col] = ASCII[min(int((light / 255) ** 0.75 * len(ASCII)), len(ASCII) - 1)]
            fg[row, col], bg[row, col] = _pack(r * lift, g * lift, b * lift), 0


@njit(cache=True)
def _digits(out, n, v):
    if v >= 100:
        out[n] = 48 + v // 100; n += 1
    if v >= 10:
        out[n] = 48 + v // 10 % 10; n += 1
    out[n] = 48 + v % 10
    return n + 1


@njit(cache=True)
def _encode(char, fg, bg, pchar, pfg, pbg, mask, out):
    """ANSI for every changed cell. `mask` drops the lowest colour bits, which nobody sees but the terminal must parse."""
    rows, cols = char.shape
    n = 0
    cur_fg = cur_bg = np.int64(-1)
    at_row = at_col = -1
    for row in range(rows):
        for col in range(cols):
            c, f, b = char[row, col], fg[row, col] & mask, bg[row, col] & mask
            if c == 32:
                f = np.uint32(0)
            if c == pchar[row, col] and f == pfg[row, col] and b == pbg[row, col]:
                continue
            pchar[row, col], pfg[row, col], pbg[row, col] = c, f, b
            if at_row != row or at_col != col:
                out[n] = 27; out[n + 1] = 91; n += 2
                n = _digits(out, n, row + 1)
                out[n] = 59; n += 1
                n = _digits(out, n, col + 1)
                out[n] = 72; n += 1
            for which in range(2):
                v = np.int64(f) if which == 0 else np.int64(b)
                if which == 0:
                    if v == cur_fg or c == 32:
                        continue
                    cur_fg = v
                else:
                    if v == cur_bg:
                        continue
                    cur_bg = v
                out[n] = 27; out[n + 1] = 91; out[n + 2] = 51 if which == 0 else 52; out[n + 3] = 56
                out[n + 4] = 59; out[n + 5] = 50; out[n + 6] = 59; n += 7
                n = _digits(out, n, v >> 16 & 255)
                out[n] = 59; n += 1
                n = _digits(out, n, v >> 8 & 255)
                out[n] = 59; n += 1
                n = _digits(out, n, v & 255)
                out[n] = 109; n += 1
            if c < 0x80:
                out[n] = c; n += 1
            elif c < 0x800:
                out[n] = 0xC0 | c >> 6; out[n + 1] = 0x80 | c & 63; n += 2
            else:
                out[n] = 0xE0 | c >> 12; out[n + 1] = 0x80 | c >> 6 & 63; out[n + 2] = 0x80 | c & 63; n += 3
            at_row, at_col = row, col + 1
    return n


class Screen:
    """A grid of cells to draw on, and the memory of what the terminal already shows."""

    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        shape = (rows, cols)
        self.char, self.fg, self.bg = (np.zeros(shape, np.uint32) for _ in range(3))
        self._pchar = np.full(shape, 0xFFFFFFFF, np.uint32)
        self._pfg, self._pbg = np.zeros(shape, np.uint32), np.zeros(shape, np.uint32)
        self._out = np.zeros(rows * cols * 56 + 64, np.uint8)

    def picture(self, rgb, mode: int, rows: int):
        to_cells(rgb, mode, self.char[:rows], self.fg[:rows], self.bg[:rows])

    def text(self, row: int, col: int, text: str, fg: int, bg: int | None = None):
        """Without `bg` the picture shows through, darkened."""
        if not 0 <= row < self.rows:
            return
        # shortcut: one character per cell and nothing beyond U+FFFF, so emoji and CJK show as "?".
        codes = [ord(c) if 32 <= ord(c) < 0x1100 or 0x2000 <= ord(c) < 0x2E80 else 63 for c in text]
        if col < 0:
            codes, col = codes[-col:], 0
        codes = codes[: self.cols - col]
        if not codes:
            return
        part = np.s_[row, col : col + len(codes)]
        if bg is None:
            self.bg[part] = (self.bg[part] >> 2) & 0x3F3F3F
        else:
            self.bg[part] = bg
        self.char[part] = codes
        self.fg[part] = fg

    def flush(self) -> bytes:
        n = _encode(self.char, self.fg, self.bg, self._pchar, self._pfg, self._pbg, np.uint32(0xFCFCFC), self._out)
        # Synchronised update: the terminal shows the whole frame at once.
        return b"\x1b[?2026h" + self._out[:n].tobytes() + b"\x1b[0m\x1b[?2026l" if n else b""
