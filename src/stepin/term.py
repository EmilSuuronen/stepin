"""Raw terminal in, key and mouse events out."""

import os
import re
import select
import sys
import termios
import tty
from dataclasses import dataclass

_CSI = re.compile(rb"\x1b\[([0-?]*)[ -/]*([@-~])")
_KEYS = {b"A": "up", b"B": "down", b"C": "right", b"D": "left", b"H": "home", b"F": "end", b"Z": "shift-tab"}
_TILDE = {b"1": "home", b"3": "delete", b"4": "end", b"5": "pageup", b"6": "pagedown", b"7": "home", b"8": "end"}
_CONTROL = {9: "tab", 10: "enter", 13: "enter", 8: "backspace", 127: "backspace"}


@dataclass
class Key:
    name: str  # "a", "up", "enter", "ctrl-c", "paste"...
    text: str = ""  # what to insert when typing


@dataclass
class Mouse:
    kind: str  # press, release, drag, wheel-up, wheel-down
    col: int
    row: int


class Parser:
    """Bytes may arrive cut anywhere; whatever is incomplete waits for the next feed."""

    def __init__(self):
        self._buffer = b""

    @property
    def pending(self) -> bool:
        return bool(self._buffer)

    def feed(self, data: bytes, final=False) -> list:
        """`final`: no more is coming soon, so a lone ESC really is the Escape key."""
        self._buffer += data
        events = []
        while self._buffer:
            event, used = self._next(final)
            if not used:
                break
            self._buffer = self._buffer[used:]
            if event:
                events.append(event)
        return events

    def _next(self, final):
        b = self._buffer
        if b[0] == 27:
            if b.startswith(b"\x1b[200~"):
                end = b.find(b"\x1b[201~")
                if end < 0:
                    return None, 0
                text = re.sub(r"[\x00-\x1f\x7f]+", " ", b[6:end].decode(errors="replace"))
                return Key("paste", text), end + 6
            if b[1:2] == b"O" and len(b) >= 3:
                return (Key(_KEYS[b[2:3]]) if b[2:3] in _KEYS else None), 3
            match = _CSI.match(b)
            if match:
                return self._csi(*match.groups()), match.end()
            if b[1:2] in (b"[", b"O") or len(b) == 1:
                return (Key("escape"), 1) if final and len(b) == 1 else (None, len(b) if final else 0)
            return Key("escape"), 1
        if b[0] < 32 or b[0] == 127:
            return Key(_CONTROL.get(b[0]) or f"ctrl-{chr(b[0] + 96)}"), 1
        size = 1 if b[0] < 0x80 else 2 if b[0] < 0xE0 else 3 if b[0] < 0xF0 else 4
        if len(b) < size and not final:
            return None, 0
        char = b[:size].decode(errors="replace")
        return Key(char, char), min(size, len(b))

    def _csi(self, params: bytes, end: bytes):
        if params.startswith(b"<") and end in b"Mm":
            try:
                button, col, row = (int(p) for p in params[1:].split(b";"))
            except ValueError:
                return None
            if button & 64:
                kind = "wheel-down" if button & 1 else "wheel-up"
            elif button & 32:
                kind = "drag"
            else:
                kind = "press" if end == b"M" else "release"
            return Mouse(kind, col - 1, row - 1)
        if end == b"~":
            name = _TILDE.get(params.split(b";")[0])
        else:
            name = _KEYS.get(end)
        return Key(name) if name else None


class Terminal:
    ENTER = b"\x1b[?1049h\x1b[?25l\x1b[?1002h\x1b[?1006h\x1b[?2004h\x1b[2J"
    LEAVE = b"\x1b[?2004l\x1b[?1006l\x1b[?1002l\x1b[?25h\x1b[0m\x1b[?1049l"

    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        self._parser = Parser()
        tty.setraw(self._fd)
        self.write(self.ENTER)
        return self

    def __exit__(self, *_):
        self.write(self.LEAVE)
        termios.tcsetattr(self._fd, termios.TCSAFLUSH, self._saved)

    def size(self) -> tuple[int, int]:
        cols, rows = os.get_terminal_size(sys.stdout.fileno())
        return rows, cols

    def write(self, data: bytes):
        view = memoryview(data)
        while view:
            view = view[os.write(sys.stdout.fileno(), view) :]

    def events(self, timeout: float) -> list:
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return self._parser.feed(b"", final=True) if self._parser.pending else []
        data = os.read(self._fd, 65536)
        if not data:
            raise EOFError
        events = self._parser.feed(data)
        if self._parser.pending and not select.select([self._fd], [], [], 0.03)[0]:
            events += self._parser.feed(b"", final=True)
        return events
