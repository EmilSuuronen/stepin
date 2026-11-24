import numpy as np
import pyte

from stepin.cells import MODES, Screen
from stepin.term import Key, Mouse, Parser


def names(events):
    return [e.name for e in events]


def test_keys_mouse_and_paste():
    p = Parser()
    assert names(p.feed(b"a\x1b[A\x1bOB\r\x7f\x03\x1b[3~")) == ["a", "up", "down", "enter", "backspace", "ctrl-c", "delete"]
    assert p.feed(b"\x1b[<0;10;5M\x1b[<32;11;5M\x1b[<64;1;1M") == [Mouse("press", 9, 4), Mouse("drag", 10, 4), Mouse("wheel-up", 0, 0)]
    assert p.feed(b"\x1b[200~rm -r box\nls\x1b[201~") == [Key("paste", "rm -r box ls")]


def test_input_cut_anywhere_gives_the_same_events():
    data = "å\x1b[A\x1b[<0;10;5Mz\x1b[200~hi\x1b[201~".encode()
    whole = Parser().feed(data)
    for cut in range(1, len(data)):
        p = Parser()
        assert p.feed(data[:cut]) + p.feed(data[cut:]) == whole, cut


def test_escape_alone():
    p = Parser()
    assert p.feed(b"\x1b") == []
    assert names(p.feed(b"", final=True)) == ["escape"]


def test_garbage_never_hangs():
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = Parser()
        p.feed(bytes(rng.integers(0, 256, 40, np.uint8)))
        p.feed(b"", final=True)
        p.feed(b"\x1b[201~", final=True)


def test_what_the_terminal_shows_is_what_was_drawn():
    """Through a real terminal emulator, over several frames, so the only-what-changed logic is covered."""
    rows, cols = 12, 40
    emulator = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(emulator)
    screen = Screen(rows, cols)
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0, 255, (rows * 4, cols * 2, 3)).astype(np.float32)
    for frame in range(8):
        rgb[rng.integers(0, rows * 4, 5)] = rng.uniform(0, 255, 3)
        screen.picture(rgb, frame % len(MODES), rows)
        screen.text(3, 5, "hello", 0xFFFFFF)
        stream.feed(screen.flush().replace(b"\x1b[?2026h", b"").replace(b"\x1b[?2026l", b""))
        for row in range(rows):
            for col in range(cols):
                cell = emulator.buffer[row][col]
                assert ord(cell.data) == screen.char[row, col]
                assert int(cell.bg, 16) == screen.bg[row, col] & 0xFCFCFC
                if cell.data != " ":
                    assert int(cell.fg, 16) == screen.fg[row, col] & 0xFCFCFC


def test_nothing_changed_sends_nothing():
    screen = Screen(5, 10)
    screen.picture(np.ones((20, 20, 3), np.float32), 0, 5)
    assert screen.flush()
    assert screen.flush() == b""


def test_text_is_clipped_and_safe():
    screen = Screen(3, 10)
    screen.text(0, 7, "abcdef", 1)
    screen.text(1, -2, "abcdef", 1)
    screen.text(9, 0, "x", 1)
    screen.text(2, 0, "a\x1b[2J", 1)
    assert [chr(c) for c in screen.char[0, 7:]] == list("abc")
    assert chr(screen.char[1, 0]) == "c"
    assert 27 not in screen.char
