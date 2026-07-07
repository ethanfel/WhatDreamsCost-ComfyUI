import torch

import ltx_extend_loop


def _latent(frames=6):
    return {"samples": torch.zeros((1, 128, frames, 2, 3), dtype=torch.float32)}


def test_fold_latent_preserves_anchor_fields():
    anchor_image = torch.ones((1, 64, 96, 3), dtype=torch.float32)
    anchor_latent = _latent(frames=4)
    state = {
        "latent": _latent(frames=6),
        "frame_rate": 24.0,
        "guide_overlap_seconds": 0.5,
        "abs_pos_px": 0,
        "anchor_image": anchor_image,
        "anchor_latent": anchor_latent,
        "anchor_mode": "auto",
        "anchor_strength": 0.25,
        "anchor_every_n_steps": 2,
    }
    passed_latent = _latent(frames=5)

    folded = ltx_extend_loop._fold_latent(state, passed_latent)

    assert folded is not state
    assert state["latent"] is not passed_latent
    assert state["abs_pos_px"] == 0
    assert folded["latent"] is passed_latent
    assert folded["anchor_image"] is anchor_image
    assert folded["anchor_latent"] is anchor_latent
    assert folded["anchor_mode"] == "auto"
    assert folded["anchor_strength"] == 0.25
    assert folded["anchor_every_n_steps"] == 2
    assert folded["abs_pos_px"] == 32
