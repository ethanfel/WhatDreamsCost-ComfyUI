# LTX Autoextend Source Anchor Design

Date: 2026-07-07

## Goal

Improve long-run quality and identity stability for the LTX 2.3 autoextend flow without weakening the existing extension behavior.

The flow should keep using the previous pass tail as the primary continuity context, while optionally carrying an original/source keyframe as a weak recurring anchor. The anchor should reduce drift over multiple extension passes without forcing the model to reset pose, camera, lighting, or motion back to the first frame.

## Context

The current autoextend path builds one extension pass in `_build_extend_pass()`:

- It slices the previous latent tail from the incoming latent.
- It creates a new empty latent for `overlap + extension`.
- It emits Director-shaped `guide_data`.
- It passes the tail latent through `guide_latents[0]`, avoiding VAE re-encode.

`LTXDirectorGuide` already accepts multiple guide entries. Each entry can use either a precomputed guide latent or an image that the Guide node encodes. It appends the guides through ComfyUI's `LTXVAddGuide.append_keyframe()` path, and `LTXDirectorCropGuides` later crops the appended guide latents back out of the sampled latent.

Upstream LTX 2.3 guidance supports this direction:

- Official extension is context-window based: the model uses context frames from the input to produce a seamless continuation.
- LTX 2.3 image-to-video preserves the visual identity of a source image.
- LTX 2.3 adds first-to-last-frame control for image-to-video, which confirms source-frame anchoring is useful, but does not replace temporal context for extension.

Sources:

- https://docs.ltx.video/api-documentation/api-reference/video-generation/extend
- https://docs.ltx.video/api-documentation/api-reference/video-generation/image-to-video
- https://docs.ltx.video/models
- https://github.com/Lightricks/LTX-Video

## Chosen Approach

Use a two-anchor extension pass:

1. Tail latent anchor at frame 0, high strength.
2. Original/source keyframe anchor at frame 0, low strength.

The tail latent remains the hard continuity source. The source keyframe becomes a weak identity and texture reference. This preserves the existing extension contract and avoids replacing the current proven continuation mechanism.

## Alternatives Considered

### Tail Latent Only

This is the current behavior. It is safest for motion continuity and transition smoothness, but it can accumulate visual drift over many passes because every pass only remembers the last generated result.

### Source Keyframe Only

This is not recommended. A strong source keyframe can improve identity on a single pass, but it competes with the current tail pose, camera angle, and scene state. In long autoregressive extension, that can cause resets or visible pulls back toward the starting frame.

### Hard Source Frame Replacement

Directly pasting the original keyframe into the generated latent window would be more invasive than needed. It risks visible discontinuity and changes the meaning of the extension pass. The source keyframe should guide attention and denoising, not overwrite the continuation window.

## Data Flow

### Manual Chain

`LTXAutoExtend` gains optional anchor inputs:

- `anchor_image`: optional `IMAGE`, normally from `LTXKeyframeOut.original` or `LTXKeyframeOut.resized`.
- `anchor_latent`: optional `LATENT`, used when available to avoid VAE re-encode.
- `anchor_mode`: `off`, `image`, `latent`, `auto`.
- `anchor_strength`: default `0.25`.

`auto` mode resolves in this order:

1. Use `anchor_latent` when connected and spatial/channel compatible.
2. Use `anchor_image` when connected.
3. Disable the source anchor.

### Loop Chain

`LTXExtendInit` stores the same anchor fields in `LTX_EXTEND_STATE`:

- `anchor_image`
- `anchor_latent`
- `anchor_mode`
- `anchor_strength`
- `anchor_every_n_steps`

`LTXExtendStep` reads the anchor state and decides whether to emit the source anchor for the current loop index.

`anchor_every_n_steps` defaults to `1`. Setting it to `2` or higher can reduce conditioning pressure and token cost for very long runs.

## Guide Construction

`_build_extend_pass()` should continue to emit the current tail guide first:

- `images[0]`: dummy image
- `insert_frames[0]`: `0`
- `strengths[0]`: `guide_strength`
- `guide_latents[0]`: tail latent

When the source anchor is active, append a second guide:

- `images[1]`: anchor image or dummy image when `anchor_latent` is used
- `original_images[1]`: pristine source image when available
- `insert_frames[1]`: `0`
- `strengths[1]`: `anchor_strength`
- `segment_numbers[1]`: `-1`
- `guide_latents[1]`: anchor latent or `None`

The tail guide must always remain first. This keeps logs, debugging, and compatibility predictable.

## Frame Placement

Place the source anchor at frame `0` of each extension window.

This intentionally aligns the identity reference with the same local coordinate system as the tail context. It should not be placed at the far end of the new extension by default, because that would behave like first-to-last-frame interpolation and could fight the desired motion.

Future work may add an `anchor_frame_mode` option, but this design keeps the first implementation narrow.

## Strength Defaults

Recommended defaults:

- `guide_strength`: existing default, normally `1.0`.
- `anchor_strength`: `0.25`.
- Useful anchor range: `0.15` to `0.35`.

Values above `0.5` should be treated as experimental. A strong original-frame anchor may improve identity but can pull the generation away from the current tail state.

## Compatibility Rules

The implementation must not change:

- Output count or output types for existing nodes unless optional inputs are appended.
- The meaning of `extension_seconds`.
- The overlap math.
- The audio window math.
- The crop behavior after appended guides.

For latent anchors:

- Channel mismatch means fallback to image mode or disable the source anchor.
- Spatial mismatch can use the same latent resizing path already used by `LTXDirectorGuide`.
- Temporal length should be cropped to one latent frame for an identity anchor unless explicitly expanded later.

For image anchors:

- The pristine image is preferred as the source input.
- The Guide node should resize to the active latent resolution using existing behavior.
- If both pristine and resized images are available, pristine should be stored in `original_images`, while the active image entry can be the resized tensor when that avoids redundant resize work.

## Error Handling

Missing anchor inputs are not an error. The extension pass should run exactly as it does today.

Invalid latent anchors should log a warning and fall back to `anchor_image` when available. If no valid image exists, skip the source anchor.

`anchor_strength <= 0` disables the source anchor.

`anchor_every_n_steps <= 0` should be treated as `1`.

## Testing

Add focused tests around `_build_extend_pass()` and loop state handling:

- Existing behavior without anchor remains byte/shape compatible where practical.
- Active image anchor adds a second guide entry with strength `anchor_strength`.
- Active latent anchor adds a second `guide_latents` entry.
- `anchor_every_n_steps` emits anchors only on expected loop indices.
- Invalid or missing anchor inputs fall back cleanly.
- The number of real latent frames before crop remains the expected extension window length.

Manual validation in ComfyUI:

- One-pass extension still matches current behavior when anchor is off.
- Multi-pass extension with source anchor preserves identity better than tail-only.
- Strong anchor values visibly pull toward the source, confirming the control works and default strength should remain low.

## Risks

The main risk is over-conditioning. If the source anchor is too strong or too frequent, the model may resist natural scene evolution and drift back toward the first frame. This is why the source anchor defaults low and stays optional.

The second risk is token cost. Each appended guide increases conditioning tokens. The mitigation is `anchor_every_n_steps`.

The third risk is confusing pristine image quality with latent quality. Pristine pixels are the best semantic identity reference, but a latent avoids VAE encode loss. The design supports both and prefers latent only when it is compatible.

## Default Behavior

`anchor_mode` defaults to `off` for backwards compatibility. Existing workflows must behave exactly as they do today unless the user connects or enables an anchor.

For new long autoextend workflows, document `auto` as the recommended setting. This gives users the best long-run stability path without silently changing saved workflows.
