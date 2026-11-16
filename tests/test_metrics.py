import cv2
import numpy as np

from culler.config import Config
from culler.features import Frame, region_motion, region_sharpness
from culler.scoring import score_moment, sharp01

from conftest import scene


def raw(img):
    f = Frame(img)
    return region_sharpness(f, (0, 0, f.w, f.h))["raw"]


def noisy(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def test_sharpness_decreases_with_blur():
    img = scene(3)
    vals = [raw(img if s == 0 else cv2.GaussianBlur(img, (0, 0), s)) for s in (0, 1, 2, 4)]
    assert vals == sorted(vals, reverse=True)
    assert vals[0] > 2 * vals[-1]


def test_noise_does_not_fake_sharpness():
    """The property variance-of-Laplacian lacks: a blurred high-ISO frame must
    not outscore a sharp clean one, and noise must never *raise* the score."""
    img = scene(4)
    blurred = cv2.GaussianBlur(img, (0, 0), 2.5)
    assert raw(noisy(blurred, 20)) < raw(img)
    assert raw(noisy(blurred, 20)) <= raw(blurred) * 1.05
    # Laplacian variance gets this wrong
    lap = lambda a: cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    assert lap(noisy(blurred, 20)) > lap(blurred) * 5


def test_ranking_survives_equal_noise():
    img = scene(5)
    for sigma in (5, 12, 25):
        assert raw(noisy(img, sigma)) > raw(noisy(cv2.GaussianBlur(img, (0, 0), 2), sigma, seed=1))


def test_pure_noise_scores_near_zero():
    flat = np.full((400, 600, 3), 90, np.uint8)
    assert sharp01(raw(noisy(flat, 15)), Config()) < 0.15


def test_motion_blur_is_directional():
    img = scene(6)
    k = np.zeros((21, 21), np.float32)
    k[10, :] = 1 / 21
    f0, f1 = Frame(img), Frame(cv2.filter2D(img, -1, k))
    b = (0, 0, f0.w, f0.h)
    assert region_motion(f1, b)["coherence"] > region_motion(f0, b)["coherence"] + 0.2
    ang = region_motion(f1, b)["blur_angle"]
    assert min(ang, 180 - ang) < 15          # horizontal


def ft(sharp_raw, blink=None, src="face", coh=0.2):
    d = {"sharp_raw": sharp_raw, "sharp_conf": 1.0, "coherence": coh,
         "subject": {"src": src, "box": [0.4, 0.3, 0.2, 0.2]}, "faces": [],
         "subject_exposure": {"clip": 0, "crush": 0}}
    if src == "face":
        d["faces"] = [{"box": [0.4, 0.3, 0.2, 0.2], "yaw": 0.0, **({"blink": blink} if blink is not None else {})}]
        d["subject_face"] = 0
    return d


def test_pick_sharpest_and_reject_soft():
    res = score_moment([ft(0.05), ft(0.30), ft(0.28)], Config())
    assert [r["auto_flag"] for r in res] == ["reject", "pick", None]
    assert res[1]["auto_rank"] == 1


def test_closed_eyes_break_the_tie():
    res = score_moment([ft(0.30, blink=0.9), ft(0.29, blink=0.05)], Config())
    assert res[1]["auto_flag"] == "pick"
    assert res[0]["auto_flag"] == "reject" and "eyes closed" in res[0]["reasons"]


def test_closed_eyes_alone_do_not_reject():
    # nobody in the burst has open eyes (singer with eyes shut): keep the best
    res = score_moment([ft(0.30, blink=0.9), ft(0.25, blink=0.9)], Config())
    assert res[0]["auto_flag"] == "pick" and res[1]["auto_flag"] is None


def test_untrusted_subject_is_never_rejected_on_absolute_score():
    res = score_moment([ft(0.021, src="frame")], Config())
    assert res[0]["auto_flag"] == "pick"


def test_second_pick_only_for_big_moments():
    cfg = Config()
    small = score_moment([ft(0.30), ft(0.30), ft(0.30)], cfg)
    big = score_moment([ft(0.30)] * 6, cfg)
    assert sum(r["auto_flag"] == "pick" for r in small) == 1
    assert sum(r["auto_flag"] == "pick" for r in big) == 2
