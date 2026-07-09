# LTX Guide Data Image Compression Design

## Goal

Add a small node that changes image-keyframe compression per generation pass without changing the upstream `LTX Director` timeline.

This supports two-pass workflows such as:

```text
LTX Director img_compression=0
        |
        +-> LTX Guide Data Image Compression CRF=36 -> Guide pass 1
        |
        +-> LTX Guide Data Image Compression CRF=0  -> Guide pass 2
```

The intended use is motion experimentation: pass 1 can receive compressed keyframe images, while pass 2 can receive clean keyframe images.

## Scope

The node applies only to image keyframes carried in `guide_data["images"]`.

It does not modify:

- `guide_data["guide_latents"]`
- `guide_data["original_images"]`
- `insert_frames`
- `strengths`
- `segment_numbers`
- prompt, timeline, audio, or motion guide data

`guide_latents` are already latent tensors. Pixel compression such as H.264 CRF does not conceptually apply to them without a VAE decode -> image compression -> VAE encode round trip. That round trip is intentionally out of scope.

## Node

Name: `LTX Guide Data Image Compression`

Inputs:

- `guide_data`: Director guide data
- `img_compression`: integer CRF `0-100`

Output:

- `guide_data`: copied guide data with updated image keyframes

Behavior:

- Copy the incoming `guide_data` dictionary and list fields before mutation.
- If `img_compression <= 0`, return clean image keyframes unchanged.
- If `img_compression > 0`, apply the existing `_compress_image()` helper to each tensor in `guide_data["images"]`.
- Preserve all metadata and guide latent entries unchanged.

## Important Workflow Rule

To make pass-specific compression meaningful, upstream `LTX Director.img_compression` should be `0`.

If the Director already baked compression into `guide_data["images"]`, this node cannot restore clean keyframes later. It can only pass through or further compress the already-compressed pixels.

## Guide Latent Interaction

If a guide entry has `guide_latents[i]`, `LTX Director Guide` may use that latent instead of VAE-encoding `images[i]`. In that case, image compression for that specific entry is bypassed by the existing guide logic.

This is expected. The compression node should not remove or alter `guide_latents`; it is strictly an image-keyframe transform.

## Testing

Add focused tests for:

- CRF `0` returns image entries with unchanged tensor values.
- CRF `>0` calls the compression path for `guide_data["images"]`.
- Metadata lists are preserved.
- `guide_latents` are preserved by object identity.

