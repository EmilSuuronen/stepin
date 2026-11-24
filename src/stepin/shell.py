"""The commands. They talk to the app through a few verbs (fly_to, fade, say...) and never draw anything."""

import fnmatch
import os
import shlex

import numpy as np
from scipy.spatial import cKDTree

from .cells import MODES
from .paths import builtin
from .render import STYLES
from .scene import Obj

AXES = {"left": (0, -1), "right": (0, 1), "up": (1, -1), "down": (1, 1), "back": (2, 1), "away": (2, 1), "forward": (2, -1), "closer": (2, -1)}

HELP = """\
ls [-l] [thing]      what is here
tree                 everything, nested
cd thing | .. | /    fly to it, or back
cat thing            what it is and how big
find words           look for anything: find red mug
rm [-r] thing...     take it away and see what was behind
mv thing dir metres  mv chair left 1.2
cp thing             another one
measure a [b]        size, or distance between two
undo                 take back the last change
style photo|hologram|depth|objects
mode quad|half|braille|ascii
photos               photos that come with stepin
open photo           step into another one, or any image file
labels               show or hide name tags
snap [file.png]      save this view
reset  clear  help  exit"""


class ShellError(Exception):
    pass


def metres(value: float) -> str:
    return f"{value * 100:.0f} cm" if value < 1 else f"{value:.2f} m"


class Shell:
    def __init__(self, app):
        self.app = app
        self.cwd: Obj | None = None
        self.commands = {name[4:]: getattr(self, name) for name in dir(self) if name.startswith("cmd_")}
        self.commands.update(quit=self.cmd_exit, grep=self.cmd_find, dir=self.cmd_ls, man=self.cmd_help)

    @property
    def scene(self):
        return self.app.scene

    @property
    def prompt(self) -> str:
        return f"{self.app.title}{'/' + self.cwd.path if self.cwd else ''}> "

    def run(self, line: str):
        try:
            words = shlex.split(line)
        except ValueError as error:
            return self.app.error(f"syntax: {error}".lower())
        if not words:
            return
        command = self.commands.get(words[0])
        if command is None:
            return self.app.error(f"{words[0]}: command not found (try help)")
        try:
            command(words[1:])
        except (ShellError, ValueError) as error:
            self.app.error(f"{words[0]}: {error}")

    def complete(self, line: str) -> tuple[str, list[str]]:
        """The line with the last word completed as far as it is certain, and the choices if it isn't."""
        head, _, word = line.rpartition(" ")
        if not head:
            options = sorted(self.commands)
        elif head.split()[0] == "style":
            options = STYLES
        elif head.split()[0] == "mode":
            options = MODES
        elif head.split()[0] == "open":
            options = builtin()
        else:
            base, _, _ = word.rpartition("/")
            prefix = base + "/" if base else ""
            try:
                parent = self.resolve(base) if base else self.cwd
            except ShellError:
                return line, []
            options = [prefix + o.name + ("/" if self.scene.children(o) else "") for o in self.scene.children(parent)]
        hits = [o for o in options if o.startswith(word)]
        if not hits:
            return line, []
        common = os.path.commonprefix(hits)
        if len(hits) == 1 and not common.endswith("/"):
            common += " "
        return (head + " " if head else "") + common, hits if len(hits) > 1 else []

    # names --------------------------------------------------------------------

    def resolve(self, path: str) -> Obj | None:
        """None is the photo itself."""
        at = None if path.startswith(("/", "~")) else self.cwd
        for part in path.lstrip("~").split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                at = at.parent if at else None
                continue
            at_children = {o.name: o for o in self.scene.children(at)}
            # Names are unique, so a nested thing can also be reached without its path.
            anywhere = self.scene.objects.get(part)
            if part in at_children:
                at = at_children[part]
            elif anywhere is not None and anywhere.alive:
                at = anywhere
            else:
                raise ShellError(f"{part}: no such thing here")
        return at

    def thing(self, path: str) -> Obj:
        obj = self.resolve(path)
        if obj is None:
            raise ShellError(f"{path}: that's the whole photo")
        return obj

    def things(self, paths: list[str]) -> list[Obj]:
        out: list[Obj] = []
        for path in paths:
            if any(c in path for c in "*?["):
                hits = [o for o in self.scene.children(self.cwd) if fnmatch.fnmatchcase(o.name, path)]
                if not hits:
                    raise ShellError(f"{path}: no matches")
            else:
                hits = [self.thing(path)]
            out += [o for o in hits if o not in out]
        return out

    def size(self, obj: Obj):
        box = self.scene.bounds(obj)
        if box is None:
            raise ShellError(f"{obj.name}: too few points to measure")
        return box[1] - box[0]

    # looking ------------------------------------------------------------------

    def cmd_ls(self, args):
        long = "-l" in args or "-la" in args
        names = [a for a in args if not a.startswith("-")]
        if any(a.startswith("-") and a not in ("-l", "-a", "-la") for a in args):
            raise ShellError("only -l is supported")
        at = self.resolve(names[0]) if names else self.cwd
        items = sorted(self.scene.children(at), key=lambda o: o.name)
        if not items and at is not None:
            items = [at]
        if not items:
            return self.app.say("(nothing here; try: find something)")
        if long:
            for o in items:
                box = self.scene.bounds(o)
                w, h, d = (box[1] - box[0]) if box else (0, 0, 0)
                away = np.linalg.norm((box[0] + box[1]) / 2 - self.app.camera.eye) if box else 0
                kids = len(self.scene.children(o))
                self.app.say(f"{'d' if kids else '-'}  {w:5.2f} x {h:4.2f} x {d:4.2f} m  {away:6.2f} m away  {o.name}{'/' if kids else ''}")
        else:
            self.app.columns([o.name + ("/" if self.scene.children(o) else "") for o in items])
        self.app.flash(items)

    def cmd_tree(self, args):
        def walk(parent, indent):
            items = sorted(self.scene.children(parent), key=lambda o: o.name)
            for i, o in enumerate(items):
                last = i == len(items) - 1
                self.app.say(f"{indent}{'└── ' if last else '├── '}{o.name}")
                walk(o, indent + ("    " if last else "│   "))

        self.app.say(self.app.title)
        walk(None, "")

    def cmd_pwd(self, args):
        self.app.say("/" + (self.cwd.path if self.cwd else ""))

    def cmd_cd(self, args):
        if len(args) > 1:
            raise ShellError("one place at a time")
        target = self.resolve(args[0]) if args else None
        self.app.fly_to(target)
        self.cwd = target

    def cmd_cat(self, args):
        if not args:
            return self.app.say(self.scene.caption)
        for obj in self.things(args):
            w, h, d = self.size(obj)
            centre = sum(self.scene.bounds(obj)) / 2
            self.app.say(f"{obj.name}: a {obj.label}, about {metres(w)} wide, {metres(h)} tall, {metres(d)} deep (visible part), "
                         f"{metres(np.linalg.norm(centre - self.app.camera.eye))} from you")
            self.app.flash([obj])

    def cmd_measure(self, args):
        if len(args) == 1:
            return self.cmd_cat(args)
        if len(args) != 2:
            raise ShellError("usage: measure thing [other]")
        a, b = (self.thing(x) for x in args)
        pa, pb = (np.concatenate([self.scene.object_points(o) for o in self.scene.family(x)]) for x in (a, b))
        if len(pa) < 4 or len(pb) < 4:
            raise ShellError("too few points to measure")
        # The 1st percentile rather than the minimum: a single stray point must not decide it.
        gap = np.percentile(cKDTree(pb).query(pa[:: max(1, len(pa) // 20000)])[0], 1)
        between = np.linalg.norm(np.median(pa, 0) - np.median(pb, 0))
        self.app.say(f"{a.name} to {b.name}: about {metres(gap)} apart, {metres(between)} between centres")
        self.app.flash([a, b])

    def cmd_find(self, args):
        text = " ".join(a for a in args if not a.startswith("-")).strip()
        if not text:
            raise ShellError("usage: find anything you can describe")
        self.app.find(text.lower())

    # changing -----------------------------------------------------------------

    def cmd_rm(self, args):
        recursive = any(a.startswith("-") and "r" in a for a in args)
        names = [a for a in args if not a.startswith("-")]
        if not names:
            raise ShellError("usage: rm [-r] thing...")
        if any(n.strip("/~.") == "" for n in names):
            raise ShellError("refusing to remove the room you are standing in")
        objs = self.things(names)
        for o in objs:
            if self.scene.children(o) and not recursive:
                raise ShellError(f"{o.name}: has things in it (use rm -r)")
        family = []
        for o in objs:
            family += [f for f in self.scene.family(o) if f not in family]
        self.app.remove(family)
        if self.cwd in family:
            self.cwd = None

    def cmd_mv(self, args):
        if len(args) not in (2, 3) or args[1] not in AXES:
            raise ShellError(f"usage: mv thing {'|'.join(AXES)} [metres]")
        obj = self.thing(args[0])
        try:
            distance = float(args[2].removesuffix("m")) if len(args) == 3 else 0.5
        except ValueError:
            raise ShellError(f"{args[2]}: not a distance") from None
        if not abs(distance) <= 1000:  # also catches nan
            raise ShellError("that's too far")
        axis, sign = AXES[args[1]]
        step = np.zeros(3, np.float32)
        step[axis] = sign * distance
        self.app.move(self.scene.family(obj), step)

    def cmd_cp(self, args):
        if len(args) != 1:
            raise ShellError("usage: cp thing")
        self.app.copy(self.thing(args[0]))

    def cmd_undo(self, args):
        self.app.undo()

    # the window ---------------------------------------------------------------

    def _choice(self, args, options, what):
        if len(args) != 1 or args[0] not in options:
            raise ShellError(f"usage: {what} {'|'.join(options)}")
        return options.index(args[0])

    def cmd_style(self, args):
        self.app.style = self._choice(args, STYLES, "style")

    def cmd_mode(self, args):
        self.app.mode = self._choice(args, MODES, "mode")

    def cmd_photos(self, args):
        self.app.columns([name + ("*" if name == self.app.title else "") for name in builtin()])

    def cmd_open(self, args):
        if len(args) != 1:
            raise ShellError("usage: open photo")
        self.app.open(args[0])

    def cmd_labels(self, args):
        self.app.labels = not self.app.labels

    def cmd_snap(self, args):
        if len(args) > 1:
            raise ShellError("usage: snap [file.png]")
        self.app.snap(args[0] if args else None)

    def cmd_reset(self, args):
        self.app.fly_to(None)
        self.cwd = None

    def cmd_clear(self, args):
        self.app.log.clear()

    def cmd_help(self, args):
        for line in HELP.splitlines():
            self.app.say(line)

    def cmd_exit(self, args):
        self.app.running = False
