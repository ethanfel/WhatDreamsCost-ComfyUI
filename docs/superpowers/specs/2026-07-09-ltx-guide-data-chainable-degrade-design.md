# LTX Guide Data Chainable Degrade Design

## Goal

Add small, chainable `GUIDE_DATA` transform nodes for two-pass LTX workflows.

The nodes should let pass 1 loosen image keyframes for motion, while pass 2 can use cleaner/stronger guide data for detail and identity.

## Design

Keep the existing `LTX Guide Data Image Compression` node as the CRF/H.264 artifact option.

Add three focused nodes:

1. `LTX Guide Data Image Resample Degrade`
   - Downscale each image keyframe, then upscale it back to its original dimensions.
   - Inputs: `guide_data`, `scale`, `method`.
   - Output: copied `guide_data` with only `images` changed.

2. `LTX Guide Data Image Noise`
   - Add deterministic image-space noise to each image keyframe.
   - Inputs: `guide_data`, `amount`, `seed`.
   - Output: copied `guide_data` with only `images` changed.

3. `LTX Guide Data Strength Override`
   - Override or multiply image/keyframe guide strengths.
   - Inputs: `guide_data`, `strengths`, `mode`.
   - Output: copied `guide_data` with only `strengths` changed.

## Data Rules

Image degrade nodes modify only:

- `guide_data["images"]`

They do not modify:

- `guide_data["guide_latents"]`
- `guide_data["original_images"]`
- `insert_frames`
- `strengths`
- `segment_numbers`
- prompt, timeline, audio, or motion guide data

Strength override modifies only:

- `guide_data["strengths"]`

It does not modify images, guide latents, or metadata.

## Strength Override Semantics

The `strengths` input is a comma-separated list mapped by image-keyframe order:

```text
0.75, 1.0, 0.6
```

It maps to:

```python
guide_data["images"][0]
guide_data["images"][1]
guide_data["images"][2]
```

It does not count text-only timeline segments.

Modes:

- `replace`: values replace the corresponding existing strengths.
- `multiply`: values multiply the corresponding existing strengths.

If fewer values are provided than image keyframes, only those positions are changed and the rest keep their existing/default strengths.

## Guide Latent Interaction

Image degradation does not apply to `guide_latents`.

If a keyframe has a `guide_latents[i]`, the existing Guide node may use that latent instead of VAE-encoding the transformed image. In that case, image degradation for that entry is bypassed by existing guide logic.

Strength override still applies to latent-backed entries because strength is passed into `append_keyframe` regardless of whether the guide came from an image encode or an existing latent.

## Example Workflows

Motion-oriented pass:

```text
LTX Director img_compression=0
  -> LTX Guide Data Image Resample Degrade scale=0.5
  -> LTX Guide Data Image Noise amount=0.02 seed=123
  -> LTX Guide Data Strength Override mode=replace strengths="0.75,0.75,0.75"
  -> LTX Director Guide pass 1
```

Detail-oriented pass:

```text
LTX Director img_compression=0
  -> LTX Guide Data Strength Override mode=replace strengths="1.0,1.0,1.0"
  -> LTX Director Guide pass 2
```

## Testing

Add focused tests for:

- Resample node changes image tensors but preserves shape and metadata.
- Noise node is deterministic for the same seed and changes only `images`.
- Strength override `replace` modifies listed strengths only.
- Strength override `multiply` multiplies listed strengths only.
- All new legacy node metadata is JSON serializable for `/object_info`.

