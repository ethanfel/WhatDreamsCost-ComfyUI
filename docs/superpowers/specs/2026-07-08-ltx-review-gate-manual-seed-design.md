# LTX Review Gate Manual Seed Design

## Goal

Let `LTX Review Gate` work outside the extend retry loop as a preview and seed-control tool.

The gate should still support its current blocking loop workflow, but it should also be useful in passthrough/manual workflows where there is no loop `attempt` value to rotate the seed.

## User Behavior

- Review preview audio/video plays only while the mouse is over the preview area.
- Leaving the preview pauses media playback.
- The existing `Reroll seed` gate button increments a controlled seed by `+1`.
- `+1` reroll is deterministic and reproducible.
- Existing loop workflows continue to work through `decision="reroll"` and `attempt -> seed_offset`.
- In passthrough/out-of-loop preview mode, `Reroll seed` remains enabled for seed control while `Pass` and `Reload prompt` remain disabled because no backend review is waiting.

## Components

### LTX Review Gate Frontend

`js/ltx_review.js` owns the embedded gate UI. It should:

- Stop autoplaying preview media when a preview arrives.
- Start video and any fallback audio on `mouseenter`.
- Pause video and any fallback audio on `mouseleave`.
- Keep the existing native video controls.
- On `Reroll seed`, increment compatible `LTX Review Seed` nodes before submitting the existing backend decision.

Browser audio autoplay rules may block sound until the page has user activation. The hover handler should attempt playback and fail quietly if the browser blocks it.

### LTX Review Seed Node

Add a small Python node with a normal integer seed widget:

- Input: `seed` integer widget.
- Output: `seed` integer.
- Optional input: `gate_id` string, default empty.

The node itself is deliberately simple and stateless. The frontend mutates its `seed` widget by `+1`, then ComfyUI uses the updated widget value during the next queue/run.

### Gate-To-Seed Matching

The frontend should increment:

- Prefer seed nodes whose `gate_id` widget matches the Review Gate node id.
- If there is exactly one `LTX Review Seed` node in the graph, increment it.
- If there are multiple seed nodes and no clear match, do not guess. Show a short gate status message asking for a matching `gate_id`.

This avoids accidentally changing the wrong seed in larger graphs.

## Data Flow

### Existing Loop Flow

`LTX Review Gate -> decision="reroll" -> LTX Extend Loop Close -> attempt + 1 -> LTX Extend Step.seed_offset`

This remains unchanged.

### Manual/Out-Of-Loop Flow

`LTX Review Seed.seed -> sampler seed`

When the gate's `Reroll seed` button is pressed, the frontend increments the seed widget on the matching `LTX Review Seed` node. The user can then queue again, or an existing workflow can trigger a requeue if already configured.

## Error Handling

- Media playback failures are ignored because browser policy may block hover audio before user activation.
- Seed increment wraps to the ComfyUI 64-bit unsigned seed range.
- If no controlled seed is found, the gate still sends its normal backend decision so loop workflows keep working.
- If multiple possible seed nodes are found without a match, no seed is changed.

## Testing

- Unit test the Python seed node output and 64-bit wrap behavior.
- Manually inspect the frontend behavior:
  - Preview does not autoplay on arrival.
  - Hover plays preview audio/video.
- Mouse leave pauses preview audio/video.
- `Reroll seed` increments a single controlled seed node by `+1`.
- In passthrough preview mode, only `Reroll seed` is enabled.
- Existing blocking loop reroll still posts `/ltx_review_decide` with `action="reroll"`.
