import numpy as np

from stepin.build import pick_objects
from stepin.render import Camera, Renderer
from stepin.scene import Build, Scene, depth_edges, rays, slug


def test_save_and_load_keep_everything(build, tmp_path):
    path = tmp_path / "a.npz"
    build.save(path)
    again = Build.load(path)
    assert again.labels == build.labels and again.caption == build.caption
    np.testing.assert_array_equal(again.masks, build.masks)
    np.testing.assert_allclose(again.points, build.points, atol=1e-5)


def test_objects_own_their_points_and_nest(scene):
    button, box = scene.objects["button"], scene.objects["box"]
    assert button.parent is box and box.parent is None
    assert button.path == "box/button"
    assert (scene.tag == button.tag).sum() == 100
    assert (scene.tag == box.tag).sum() == 1600 - 100


def test_sizes_are_metric(scene):
    low, high = scene.bounds(scene.objects["box"])
    assert 0.7 < high[0] - low[0] < 0.8  # 40 px at 2 m and f=100 is 0.8 m; percentiles trim the rim


def test_edges_are_breaks_not_slopes():
    ray = rays((60, 60), 60.0)
    v = np.arange(60)[:, None]
    floor = 1 / (0.1 + 0.02 * v) * np.ones((60, 60))  # a plane seen at a grazing angle
    assert not depth_edges(ray * floor[..., None], np.ones((60, 60), bool)).any()
    step = np.where(np.arange(60) < 30, 2.0, 4.0) * np.ones((60, 60))
    edges = depth_edges(ray * step[..., None], np.ones((60, 60), bool))
    assert edges[30, 28:32].any() and not edges[:, :27].any() and not edges[:, 33:].any()


def test_home_view_matches_the_photo(scene, build):
    rgb, tags = Renderer().draw(scene, Camera(), 128, 96)
    assert np.abs(rgb - build.image).mean() < 3
    assert (tags[48, 60], tags[35, 45]) == (scene.objects["button"].tag, scene.objects["box"].tag)


def test_stepping_aside_shows_the_hidden_layer(scene):
    camera = Camera()
    camera.eye = np.array([0.25, 0, 0])
    _, tags = Renderer().draw(scene, camera, 128, 96)
    assert (tags == scene.objects["box"].plate).any()


def test_empty_scene_does_not_crash(build):
    build.masks, build.labels = build.masks[:0], []
    Renderer().draw(Scene(build), Camera(), 64, 48)


def test_pick_objects_drops_duplicates_and_specks():
    a = np.zeros((100, 100), bool)
    a[10:40, 10:40] = True
    b = a.copy()
    b[10:12] = False
    speck = np.zeros_like(a)
    speck[90, 90] = True
    assert pick_objects(["x", "y", "z"], np.stack([a, b, speck]), np.ones(3), a.size) == [0]


def test_slug():
    assert slug("Kitchen & Dining table") == "kitchen-dining-table"
    assert slug("../..") == "thing"
