import time

import numpy as np
import pytest

from stepin.term import Key, Mouse


def text(app):
    return "\n".join(line for line, _ in app.log)


def settle(app):
    """Run the clock until every animation has finished."""
    now = time.monotonic()
    for i in range(60):
        app.step(now + i * 0.1, 0.1)


def run(app, *lines):
    for line in lines:
        app.shell.run(line)
        settle(app)


def test_ls_and_cd_walk_the_tree(app):
    run(app, "ls")
    assert "box/" in text(app) and "button" not in text(app)
    run(app, "cd box", "ls")
    assert "button" in text(app)
    assert app.shell.prompt == "room/box> "
    assert np.linalg.norm(app.camera.eye) > 0.1
    run(app, "cd ..")
    assert np.allclose(app.camera.eye, 0) and app.shell.prompt == "room> "


def test_rm_wants_r_for_things_with_things_in_them(app):
    run(app, "rm box")
    assert "use rm -r" in text(app) and app.scene.objects["box"].alive


def test_rm_dissolves_fills_and_undo_restores(app):
    before = {k: getattr(app.scene, k).copy() for k in ("image", "points", "tag", "show", "front")}
    run(app, "rm -r box")
    box = app.scene.objects["box"]
    assert not box.alive and app.scene.show[box.tag] == 0 and app.scene.show[box.plate] == 0
    assert (app.scene.image[40, 50] == 90).all()
    app.draw_tags = app.renderer.draw(app.scene, app.camera, 128, 96)[1]
    assert (app.draw_tags[40, 50] >= 0) and app.draw_tags[40, 50] != box.tag
    run(app, "undo")
    assert box.alive
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(app.scene, key), value)
    run(app, "undo")
    assert "nothing to undo" in text(app)


def test_rm_refuses_the_room(app):
    for victim in ("/", "..", ".", "~"):
        run(app, f"rm -rf {victim}")
    assert text(app).count("refusing") == 4 and app.scene.objects["box"].alive


def test_rm_of_the_place_you_are_in_moves_you_out(app):
    run(app, "cd box", "rm -r box")
    assert app.shell.prompt == "room> "


def test_mv_moves_the_family_and_leaves_background(app):
    run(app, "mv box left 0.5")
    assert app.scene.offset[app.scene.objects["box"].tag][0] == pytest.approx(-0.5)
    assert app.scene.offset[app.scene.objects["button"].tag][0] == pytest.approx(-0.5)
    count = len(app.scene.pos)
    run(app, "mv box left 0.5")  # its old place is already filled
    assert len(app.scene.pos) == count


@pytest.mark.parametrize("line", ["mv box sideways 1", "mv box left nan", "mv box left x", "mv box left 1e9", "cd nowhere",
                                  "rm", "cp", "style x", "find", "measure a b c", "'", "frobnicate", "rm zz*", "ls -q"])
def test_mistakes_are_explained_not_fatal(app, line):
    run(app, line)
    assert app.log and app.running


def test_cp_then_rm_the_copy(app):
    run(app, "cp button", "ls box")
    assert "button2" in text(app)
    run(app, "rm button2")
    assert app.scene.objects["button"].alive and not app.scene.objects["button2"].alive


def test_find_adds_new_things_once(app):
    mask = np.zeros(app.scene.valid.shape, bool)
    mask[5:15, 5:15] = True
    app.worker.found = [(0.8, mask)]
    run(app, "find light switch")
    assert app.scene.objects["light-switch"].alive
    run(app, "find light switch")
    assert "light-switch2" not in app.scene.objects
    run(app, "undo")
    assert "light-switch" not in app.scene.objects and (app.scene.tag[app.scene.front[mask]] == 0).all()


def test_measure(app):
    run(app, "measure box")
    assert "cm wide" in text(app)


def test_failed_ai_is_reported_and_not_stuck(app):
    app.worker.request = lambda op, *a: (setattr(app.worker, "busy", op), setattr(app.worker, "_answer", ("error", "no memory")))
    run(app, "rm -r box")
    assert "no memory" in text(app) and app.idle()


def test_one_request_waits_for_the_models_to_load(app):
    app.worker.busy = "warm"
    app.shell.run("rm -r box")
    app.shell.run("find lamp")
    assert "still thinking" in text(app)
    app.worker._answer = ("done", None)
    settle(app)
    assert app.scene.objects["box"].lifted


def test_completion(app):
    assert app.shell.complete("c")[1] == ["cat", "cd", "clear", "cp"]
    assert app.shell.complete("cd bo")[0] == "cd box/"
    assert app.shell.complete("cd box/bu")[0] == "cd box/button "
    assert app.shell.complete("style ho")[0] == "style hologram "


def test_typing_and_clicking(app):
    from stepin.cells import Screen

    app.draw(Screen(25, 64), time.monotonic() + 5)
    app.handle(Key("enter"))
    for c in "cat ":
        app.handle(Key(c, c))
    app.handle(Mouse("press", 30, 12))
    app.handle(Mouse("release", 30, 12))
    assert app.line == "cat button "
    app.handle(Key("enter"))
    assert "a button, about" in text(app)
    app.handle(Key("up"))
    assert app.line == "cat button"


def test_snap_never_overwrites(app, tmp_path):
    out = tmp_path / "a.png"
    run(app, f"snap {out}", f"snap {out}")
    assert out.exists() and "already exists" in text(app)
