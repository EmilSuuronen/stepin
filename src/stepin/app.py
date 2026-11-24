"""The window: camera, animations, prompt, and the loop that ties them together."""

import math
import time
from pathlib import Path

import numpy as np

from .cells import MODES, Screen
from .paths import PHOTOS, builtin, scene_path, title_of
from .render import STYLES, Camera, Renderer
from .scene import Build, Scene
from .shell import Shell, ShellError
from .term import Key, Mouse, Terminal

AMBER, WHITE, DIM, RED, GREEN, BAR = 0xFFAE61, 0xF6F1E8, 0x9A948A, 0xFF6B5E, 0x8FE0A0, 0x14110D
FRAME = 1 / 30
UNDO_LIMIT = 10
FLY_HINT = "WASD move · R/F up/down · arrows or drag look · click pick · Enter shell · N next photo · M mode · T style · Q quit"


def overlap(scene, obj, mask) -> float:
    whole = np.any([o.mask for o in scene.family(obj) if o.mask is not None], axis=0)
    return max((m & mask).sum() / max((m | mask).sum(), 1) for m in (obj.mask, whole))


class App:
    def __init__(self, scene: Scene, worker, title: str, mode: int = 0):
        self.worker = worker
        self.renderer, self.shell = Renderer(), Shell(self)
        self._enter(scene, title)
        self.mode, self.style, self.labels = mode, 0, False
        self.log: list[tuple[str, int]] = []
        self.running, self.typing = True, False
        self.line, self.cursor = "", 0
        self.history: list[str] = []
        self._recall = 0
        self._scroll = 0
        self._pending = None  # what to do with the AI's answer
        self._after_warm = None  # one request may wait for the models to load
        self._drag = None
        self._log_until = 0.0
        self._start = time.monotonic()
        self._dirty = True
        self._screen: Screen | None = None

    def _enter(self, scene: Scene, title: str):
        self.scene, self.title, self.camera = scene, title, Camera()
        self.shell.cwd = None
        self._velocity, self._spin = np.zeros(3), np.zeros(2)  # right/down/forward and yaw/pitch, per second
        self._flight = self._tags = None
        self._tweens: list[tuple] = []
        self._flashes: dict[int, float] = {}
        self._label_until = 0.0
        self._labelled: list = []
        self._anchors: dict = {}
        self._intro = time.monotonic()

    # verbs the shell uses -------------------------------------------------------

    def say(self, text: str, colour: int = WHITE):
        for line in text.splitlines() or [""]:
            self.log.append((line, colour))
        del self.log[:-500]
        self._scroll = 0
        self._log_until = time.monotonic() + 6
        self._dirty = True

    def error(self, text: str):
        self.say(text, RED)

    def columns(self, words: list[str]):
        width = max(map(len, words)) + 2
        per_row = max(1, ((self._screen.cols if self._screen else 80) - 4) // width)
        for i in range(0, len(words), per_row):
            self.say("".join(w.ljust(width) for w in words[i : i + per_row]).rstrip(), AMBER)

    def flash(self, objs):
        now = time.monotonic()
        for o in objs:
            for member in self.scene.family(o):
                self._flashes[member.tag] = now
        self._label_until = now + 4
        self._labelled = list(objs)

    def _anchor(self, obj):
        """The middle of an object, remembered until something about it changes."""
        key = (id(self.scene.tag), self.scene.offset[obj.tag].tobytes())
        if self._anchors.get(obj, (None,))[0] != key:
            box = self.scene.bounds(obj)
            self._anchors[obj] = (key, None if box is None else (box[0] + box[1]) / 2)
        return self._anchors[obj][1]

    def fly_to(self, obj):
        if obj is None:
            return self._fly(np.zeros(3), 0.0, 0.0)
        box = self.scene.bounds(obj)
        if box is None:
            raise ShellError(f"{obj.name}: too few points to fly to")
        centre, size = (box[0] + box[1]) / 2, box[1] - box[0]
        # Stay on the side the photo was taken from; the far side was never seen.
        out = centre / max(np.linalg.norm(centre), 1e-6)
        w, h = self._view_size()
        f = self.camera.focal(self.scene, w, h)
        back = max(size[0] * f / (0.7 * w), size[1] * f / (0.7 * h)) + size[2] / 4
        eye = centre - out * min(max(back, 0.25 * np.linalg.norm(centre)), 0.95 * np.linalg.norm(centre))
        probe = Camera()
        probe.eye = eye
        self._fly(eye, *probe.aim(centre))
        self.flash([obj])

    def _fly(self, eye, yaw, pitch):
        c = self.camera
        turn = (yaw - c.yaw + math.pi) % (2 * math.pi) - math.pi
        self._velocity[:] = self._spin[:] = 0
        self._flight = (time.monotonic(), c.eye.copy(), c.yaw, c.pitch, np.asarray(eye, float), c.yaw + turn, pitch)

    def _view_size(self):
        rows, cols = (self._screen.rows, self._screen.cols) if self._screen else (24, 80)
        return cols * 2, (rows - 1) * 4

    def _ask(self, op, args, then):
        """Hand work to the AI process; `then(result)` runs when it answers."""
        if self._after_warm or (self.worker.busy and self.worker.busy != "warm"):
            raise ShellError("still thinking about the last one")
        if not self.worker.alive:
            raise ShellError("the AI process stopped; restart stepin")
        if self.worker.busy == "warm":
            self._after_warm = (op, args, then)
        else:
            self._pending = then
            self.worker.request(op, *args)

    def idle(self) -> bool:
        return self._after_warm is None and self.worker.busy in (None, "warm")

    def _need_idle(self):
        if not self.idle():
            raise ShellError("still thinking about the last one")

    def _remember(self):
        self.scene.undo_stack.append(self.scene.snapshot())
        del self.scene.undo_stack[:-UNDO_LIMIT]

    def _tween(self, array, index, end, seconds, delay=0.0):
        self._tweens.append((array, index, np.copy(array[index]), end, time.monotonic() + delay, seconds))

    def _finish_tweens(self):
        for array, index, _, end, _, _ in self._tweens:
            array[index] = end
        self._tweens.clear()

    def _vacate(self, objs, on_failure):
        """Ask for what is behind these objects and fade it in."""
        mask = self.scene.fill_mask(objs)
        if not mask.any():
            return
        scene = self.scene

        def arrived(result):
            tag = scene.lift(objs, mask, result)
            scene.show[tag] = 0
            self._tween(scene.show, tag, 1.0, 0.8)
            # The outline and the rough guess stay until the better one is there.
            for o in objs:
                for hidden in (o.halo, o.plate):
                    self._tween(scene.show, hidden, 0.0, 0.3, delay=0.6)
            self.say("painted in what might be behind it", DIM)

        self._ask("fill", (scene.image, scene.points, scene.valid, mask, scene.focal), (arrived, on_failure))

    def remove(self, objs):
        self._need_idle()
        self._finish_tweens()
        self._remember()
        try:
            self._vacate(objs, lambda: None)
        except ShellError:
            self.scene.undo_stack.pop()
            raise
        for o in objs:
            o.alive = False
            self._tween(self.scene.show, o.tag, 0.0, 1.6)
        self.say(f"removed {', '.join(o.name for o in objs)}", DIM)

    def move(self, objs, step):
        self._need_idle()
        self._finish_tweens()
        self._remember()
        try:
            self._vacate([o for o in objs if not o.lifted], lambda: None)
        except ShellError:
            self.scene.undo_stack.pop()
            raise
        for o in objs:
            self._tween(self.scene.offset, o.tag, self.scene.offset[o.tag] + step, 0.9)
        self.flash(objs[:1])

    def copy(self, obj):
        self._need_idle()
        self._finish_tweens()
        self._remember()
        new = self.scene.copy_object(obj)
        self._tween(self.scene.show, new.tag, 1.0, 1.2)
        self.say(f"made {new.name}", DIM)
        self.flash([new])

    def undo(self):
        self._need_idle()
        if not self.scene.undo_stack:
            raise ShellError("nothing to undo")
        self._tweens.clear()
        self.scene.restore(self.scene.undo_stack.pop())
        if self.shell.cwd is not None and (not self.shell.cwd.alive or self.shell.cwd.name not in self.scene.objects):
            self.shell.cwd = None
        self.say("undone", DIM)

    def find(self, text):
        def arrived(found):
            scene, hits, snap = self.scene, [], None
            for score, mask in found:
                known = [o for o in scene.objects.values() if o.alive and o.mask is not None and not o.lifted]
                # A thing's outline includes what is in it: a lamp with its shade is still that lamp.
                same = next((o for o in known if overlap(scene, o, mask) > 0.6), None)
                if same is None:
                    if snap is None:
                        self._finish_tweens()
                        snap = scene.snapshot()
                    # Deepest existing thing that holds most of it becomes its parent.
                    holders = [o for o in known if (o.mask & mask).sum() > 0.7 * mask.sum()]
                    same = scene.add_object(text, mask)
                    same.parent = min(holders, key=lambda o: o.mask.sum(), default=None)
                hits.append((score, same))
            if snap:
                scene.undo_stack.append(snap)
                del scene.undo_stack[:-UNDO_LIMIT]
            if not hits:
                return self.say(f"no {text} in sight")
            for score, obj in hits:
                self.say(f"/{obj.path}  ({score:.0%} sure)", AMBER)
            self.flash([o for _, o in hits])

        self._ask("find", (self.scene.image, text), (arrived, lambda: None))
        self.say(f"looking for {text}...", DIM)

    def open(self, name):
        """A built-in photo by name, or any image file. New files are looked at first, which takes a while."""
        self._need_idle()
        photo = Path(name).expanduser()
        if name in builtin():
            path, title = PHOTOS / f"{name}.npz", name
        elif not photo.is_file():
            raise ShellError(f"{name}: no such photo (see photos)")
        else:
            path, title = scene_path(photo), title_of(photo)
        try:
            return self._enter(Scene(Build.load(path)), title)
        except Exception:
            if name in builtin():
                raise ShellError(f"{name}: the scene file is damaged") from None
        # Not seen before, or saved by an older version.
        path.parent.mkdir(parents=True, exist_ok=True)

        def arrived(_):
            self._enter(Scene(Build.load(path)), title)

        self._ask("build", (str(photo.resolve()), path), (arrived, lambda: None))
        self.say(f"looking at {photo.name}...", DIM)

    def next_photo(self):
        names = builtin()
        if names:
            self.open(names[(names.index(self.title) + 1) % len(names)] if self.title in names else names[0])

    def snap(self, name):
        path = Path(name or f"{self.title}-{time.strftime('%H%M%S')}.png").expanduser()
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            raise ShellError(f"{path}: use a .png, .jpg or .webp name")
        if path.exists():
            raise ShellError(f"{path}: already exists")
        from PIL import Image

        rgb, _ = Renderer().draw(self.scene, self.camera, 1600, 1000, self.style, time.monotonic() - self._start)
        try:
            Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(path)
        except OSError as error:
            raise ShellError(f"{path}: {error.strerror or error}") from None
        self.say(f"saved {path}", GREEN)

    # input ----------------------------------------------------------------------

    def handle(self, event):
        self._dirty = True
        if isinstance(event, Mouse):
            return self._mouse(event)
        if event.name in ("ctrl-c", "ctrl-d") and not (self.typing and self.line):
            self.running = False
        elif self.typing:
            self._type(event)
        else:
            self._steer(event)

    def _mouse(self, event):
        if event.kind.startswith("wheel"):
            up = event.kind == "wheel-up"
            if self.typing:
                self._scroll = min(max(self._scroll + (3 if up else -3), 0), max(0, len(self.log) - 3))
            else:
                self._push(2, 1 if up else -1)
        elif event.kind == "press":
            self._drag = (event.col, event.row, False)
        elif event.kind == "drag" and self._drag:
            col, row, _ = self._drag
            w, h = self._view_size()
            f = self.camera.focal(self.scene, w, h)
            self._flight = None
            # Grab the picture: it follows the pointer.
            self.camera.yaw -= (event.col - col) * 2 / f
            self.camera.pitch = min(max(self.camera.pitch + (event.row - row) * 4 / f, -1.4), 1.4)
            self._drag = (event.col, event.row, True)
        elif event.kind == "release" and self._drag:
            if not self._drag[2]:
                self._pick(event.col, event.row)
            self._drag = None

    def _pick(self, col, row):
        if self._tags is None or not (0 <= row * 4 < self._tags.shape[0] and 0 <= col * 2 < self._tags.shape[1]):
            return
        cell = self._tags[row * 4 : row * 4 + 4, col * 2 : col * 2 + 2].ravel()
        found, count = np.unique(cell, return_counts=True)
        by_tag = {o.tag: o for o in self.scene.objects.values() if o.alive}
        obj = by_tag.get(int(found[count.argmax()]))
        if obj is None:
            return
        self.flash([obj])
        if self.typing and (not self.line or self.line.endswith(" ")):
            self._insert(obj.name + " ")
        else:
            try:
                self.shell.cmd_cat([obj.name])
            except ShellError as error:
                self.error(str(error))

    def _push(self, axis, sign):
        self._flight = None
        top = 0.35 * self.scene.scale
        self._velocity[axis] = min(max(self._velocity[axis] + sign * 0.2 * self.scene.scale, -top), top)

    def _turn(self, axis, sign):
        self._flight = None
        self._spin[axis] = min(max(self._spin[axis] + sign * 0.45, -0.9), 0.9)

    def _steer(self, key: Key):
        name = key.name.lower() if len(key.name) == 1 else key.name
        moves = {"w": (2, 1), "s": (2, -1), "d": (0, 1), "a": (0, -1), "r": (1, -1), "f": (1, 1)}
        turns = {"right": (0, 1), "left": (0, -1), "up": (1, 1), "down": (1, -1)}
        if name in moves:
            self._push(*moves[name])
        elif name in turns:
            self._turn(*turns[name])
        elif name in ("enter", ":", "/"):
            self.typing = True
        elif name == "m":
            self.mode = (self.mode + 1) % len(MODES)
        elif name == "t":
            self.style = (self.style + 1) % len(STYLES)
        elif name == "l":
            self.labels = not self.labels
        elif name == "n":
            try:
                self.next_photo()
            except ShellError as error:
                self.error(str(error))
        elif name in ("home", "z"):
            self.fly_to(None)
        elif name == "q":
            self.running = False

    def _insert(self, text):
        self.line = self.line[: self.cursor] + text + self.line[self.cursor :]
        self.cursor += len(text)

    def _type(self, key: Key):
        name = key.name
        if name == "enter":
            line, self.line, self.cursor = self.line.strip(), "", 0
            if not line:
                self.typing = False
                return
            if not self.history or self.history[-1] != line:
                self.history.append(line)
            self._recall = len(self.history)
            self.say(self.shell.prompt + line, GREEN)
            self.shell.run(line)
        elif name == "escape":
            self.typing = False
        elif name in ("ctrl-c", "ctrl-u"):
            self.line, self.cursor = "", 0
        elif name == "backspace" and self.cursor:
            self.line = self.line[: self.cursor - 1] + self.line[self.cursor :]
            self.cursor -= 1
        elif name == "delete" or name == "ctrl-d":
            self.line = self.line[: self.cursor] + self.line[self.cursor + 1 :]
        elif name == "ctrl-w":
            kept = self.line[: self.cursor].rstrip().rpartition(" ")
            head = kept[0] + kept[1]
            self.line, self.cursor = head + self.line[self.cursor :], len(head)
        elif name == "ctrl-l":
            self.log.clear()
        elif name in ("home", "ctrl-a"):
            self.cursor = 0
        elif name in ("end", "ctrl-e"):
            self.cursor = len(self.line)
        elif name == "left":
            self.cursor = max(0, self.cursor - 1)
        elif name == "right":
            self.cursor = min(len(self.line), self.cursor + 1)
        elif name in ("up", "down") and self.history:
            self._recall = min(max(self._recall + (-1 if name == "up" else 1), 0), len(self.history))
            self.line = self.history[self._recall] if self._recall < len(self.history) else ""
            self.cursor = len(self.line)
        elif name in ("pageup", "pagedown"):
            self._scroll = min(max(self._scroll + (5 if name == "pageup" else -5), 0), max(0, len(self.log) - 3))
        elif name == "tab":
            done, options = self.shell.complete(self.line[: self.cursor])
            self.line, self.cursor = done + self.line[self.cursor :], len(done)
            if options:
                self.columns(options)
        elif key.text:
            self._insert(key.text)

    # time -------------------------------------------------------------------------

    def step(self, now, dt):
        """Advance everything that moves. Returns whether anything did."""
        active = False
        c = self.camera
        if self._flight:
            start, eye0, yaw0, pitch0, eye1, yaw1, pitch1 = self._flight
            u = min((now - start) / 1.1, 1)
            s = u * u * (3 - 2 * u)
            c.eye, c.yaw, c.pitch = eye0 + (eye1 - eye0) * s, yaw0 + (yaw1 - yaw0) * s, pitch0 + (pitch1 - pitch0) * s
            if u >= 1:
                self._flight = None
            active = True
        if np.abs(self._velocity).max() > 0.002 * self.scene.scale or np.abs(self._spin).max() > 0.004:
            # Walk level whatever the pitch: forward is where you face, up is up.
            right = np.array([math.cos(c.yaw), 0, -math.sin(c.yaw)])
            ahead = np.array([math.sin(c.yaw), 0, math.cos(c.yaw)])
            c.eye = c.eye + (right * self._velocity[0] + np.array([0, 1, 0]) * self._velocity[1] + ahead * self._velocity[2]) * dt
            # Nobody should get lost in the void.
            reach = np.linalg.norm(c.eye)
            if reach > 4 * self.scene.scale:
                c.eye *= 4 * self.scene.scale / reach
            c.yaw += self._spin[0] * dt
            c.pitch = min(max(c.pitch + self._spin[1] * dt, -1.4), 1.4)
            active = True
        # Terminals report key presses but not releases, so motion is a push that fades.
        self._velocity *= math.exp(-3.5 * dt)
        self._spin *= math.exp(-5 * dt)

        running, self._tweens = self._tweens, []
        for tween in running:
            array, index, begin, end, start, seconds = tween
            u = min(max((now - start) / seconds, 0), 1)
            array[index] = begin + (end - begin) * (u * u * (3 - 2 * u))
            if u < 1:
                self._tweens.append(tween)
            active = True
        for tag, start in list(self._flashes.items()):
            u = (now - start) / 2.2
            self.scene.glow[tag] = max(0.0, (1 - u) * (0.75 + 0.25 * math.cos((now - start) * 9)))
            if u >= 1:
                del self._flashes[tag]
                self.scene.glow[tag] = 0
            active = True
        self._listen()
        return active

    def _listen(self):
        message = self.worker.poll()
        if message is None:
            return
        self._dirty = True
        kind, value = message
        if kind == "progress":
            return self.say(f"{value}...", DIM)
        pending, self._pending = self._pending, None
        if pending:
            arrived, failed = pending
            if kind == "done":
                try:
                    arrived(value)
                except ValueError as error:  # out of tags
                    self.error(str(error))
            else:
                failed()
                self.error(value)
        elif kind == "error":
            self.error(value)
        if self._after_warm:
            op, args, then = self._after_warm
            self._after_warm = None
            if self.worker.alive:
                self._pending = then
                self.worker.request(op, *args)
            else:
                then[1]()

    # drawing ------------------------------------------------------------------------

    def draw(self, screen: Screen, now):
        self._screen = screen
        rows = screen.rows - 1
        show = None
        u = (now - self._intro) / 2.2
        if u < 1:  # the scene assembles itself out of dust
            show = self.scene.show * min(max(u, 0.01), 0.99)
        rgb, self._tags = self.renderer.draw(self.scene, self.camera, screen.cols * 2, rows * 4, self.style, now - self._start, show)
        screen.picture(rgb, self.mode, rows)
        if self.labels or now < self._label_until:
            self._draw_labels(screen, rows)
        self._draw_log(screen, rows, now)
        self._draw_bar(screen, rows, now)

    def _draw_labels(self, screen, rows):
        objs = [o for o in self.scene.objects.values() if o.alive] if self.labels else [o for o in self._labelled if o.alive]
        taken = set()
        for o in objs:
            middle = self._anchor(o)
            at = None if middle is None else self.camera.to_screen(middle, self.scene, screen.cols * 2, rows * 4)
            if at is None:
                continue
            row, col = int(at[1] // 4), int(at[0] // 2) - len(o.name) // 2
            if not (0 <= row < rows and 0 <= col < screen.cols):
                continue
            while (row, col // 12) in taken and row < rows - 1:
                row += 1
            taken.add((row, col // 12))
            screen.text(row, col, f" {o.name} ", AMBER)

    def _draw_log(self, screen, rows, now):
        if not self.log or not (self.typing or now < self._log_until):
            return
        height = max(3, min(rows // 3, 16)) if self.typing else 4
        width = screen.cols - 2
        lines = []
        for text, colour in self.log[-200:]:
            lines += [(text[i : i + width - 2], colour) for i in range(0, max(len(text), 1), width - 2)]
        end = len(lines) - self._scroll
        shown = lines[max(0, end - height) : max(end, 1)]
        for i, (text, colour) in enumerate(shown):
            screen.text(rows - len(shown) + i, 1, f" {text} ".ljust(min(width, max(len(t) for t, _ in shown) + 2)), colour)

    def _draw_bar(self, screen, row, now):
        screen.text(row, 0, " " * screen.cols, WHITE, BAR)
        busy = self._after_warm or (self.worker.busy and self.worker.busy != "warm")
        status = f"{'⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'[int(now * 10) % 10]} thinking " if busy else "waking the AI " if self.worker.busy else ""
        status += f" {STYLES[self.style]} · {MODES[self.mode]} "
        room = screen.cols - len(status) - 1
        if self.typing:
            prompt = self.shell.prompt
            screen.text(row, 1, prompt[: screen.cols - 1], GREEN, BAR)
            at = 1 + len(prompt)
            # Keep the cursor in sight when the line is longer than the window.
            first = max(0, self.cursor - max(screen.cols - at - 2, 1))
            screen.text(row, at, self.line[first:], WHITE, BAR)
            under = self.line[self.cursor] if self.cursor < len(self.line) else " "
            screen.text(row, min(at + self.cursor - first, screen.cols - 1), under, BAR, AMBER)
            room = room if at + len(self.line) < room else 0
        else:
            screen.text(row, 1, FLY_HINT[: max(room - 1, 0)], DIM, BAR)
        if room > 0:
            screen.text(row, screen.cols - len(status), status, AMBER, BAR)

    def prepare(self):
        """Compile the renderer before the window opens; only slow the very first time."""
        screen = Screen(8, 40)
        self.draw(screen, self._start)
        screen.flush()
        self._start = self._intro = time.monotonic()

    def run(self):
        with Terminal() as terminal:
            screen, last = None, time.monotonic()
            drawn = 0.0
            while self.running:
                now = time.monotonic()
                dt, last = min(now - last, 0.1), now
                size = terminal.size()
                if screen is None or (screen.rows, screen.cols) != size:
                    screen = Screen(max(size[0], 4), max(size[1], 20))
                    terminal.write(b"\x1b[2J")
                    self._dirty = True
                active = self.step(now, dt) or now - self._intro < 2.3
                # Dust, scan lines and the spinner keep moving on their own, more slowly; a still picture costs nothing.
                restless = self.style == 1 or self.worker.busy or np.any(self.camera.eye) or now < max(self._log_until, self._label_until) + 0.2
                if active or self._dirty or (restless and now - drawn > 0.1):
                    self._dirty = False
                    self.draw(screen, now)
                    terminal.write(screen.flush())
                    drawn = now
                spent = time.monotonic() - now
                try:
                    events = terminal.events(max(FRAME - spent, 0.001))
                except EOFError:
                    break
                for event in events:
                    self.handle(event)
