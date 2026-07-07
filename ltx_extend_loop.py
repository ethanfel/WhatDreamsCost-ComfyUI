"""Native, self-contained loop for the LTX extend chain — no external loop pack needed.

LTX Extend Loop Open / Close use ComfyUI's execution-inversion (GraphBuilder) to re-run the loop
body, carrying ONE 'state' signal. Close folds each passed latent back into the state and advances
the audio position + step index; on a Review Gate 'reroll'/'reload' it re-runs the SAME step (reroll
bumps the seed via state.attempt). The expansion mechanics mirror the well-known ComfyUI while-loop.

Wiring:
    Init.state -> LoopOpen.state
    LoopOpen -> (flow, state, index, attempt)
        state -> Step.state ; index -> Step.index ; attempt -> Step.seed_offset
        Step -> Guide -> sampler1 -> sampler2 -> (latent, VAEDecode->images)
        [optional] images/latent/attempt -> LTX Review Gate -> (decision, latent)
    LoopClose: flow<-LoopOpen.flow, state<-LoopOpen.state, latent<-(gate or sampler2),
               decision<-gate.decision (omit for headless auto-pass)
    LoopClose -> (state, final_latent, steps_done)
"""

import logging

try:
    import torch
except Exception:
    torch = None

try:
    from comfy_execution.graph import ExecutionBlocker
    from comfy_execution.graph_utils import GraphBuilder, is_link
except Exception:  # allow import/syntax checks outside ComfyUI
    ExecutionBlocker = None
    GraphBuilder = None

    def is_link(value):
        return isinstance(value, list) and len(value) == 2

try:
    from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
except Exception:
    ALL_NODE_CLASS_MAPPINGS = {}

log = logging.getLogger(__name__)

_TSF = 8  # LTX temporal downscale (8n+1)


class _LoopAny(str):
    """Permissive type so the carried 'state' wires anywhere."""
    def __ne__(self, _other):
        return False


_LOOP_ANY = _LoopAny("*")


def _execution_blocker():
    return ExecutionBlocker(None) if ExecutionBlocker is not None else None


def _fold_latent(state, latent):
    """Collect: fold a PASSED step's latent into the state and advance the master-audio position by
    this pass's new-content length (= (T_old - n_overlap)*8 px), matching what Step used to cut audio."""
    st = dict(state or {})
    fps = float(st.get("frame_rate", 24.0)) or 24.0
    overlap_px = max(1, int(round(float(st.get("guide_overlap_seconds", 3.0)) * fps)))
    old = st.get("latent")
    rel_off = 0
    if torch is not None and isinstance(old, dict) and isinstance(old.get("samples"), torch.Tensor):
        t_old = int(old["samples"].shape[2])
        n_ov = max(1, min(t_old, (overlap_px + _TSF - 1) // _TSF))
        rel_off = (t_old - n_ov) * _TSF
    st["abs_pos_px"] = int(st.get("abs_pos_px", 0)) + rel_off
    st["latent"] = latent
    return st


class LTXExtendLoopOpen:
    """Entry of the native extend loop. Injects the step index / retry counter / total into the state
    and emits them each iteration. Resume by setting start_index (and feeding Init the latent you
    stopped at + its resume_from_seconds)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (_LOOP_ANY, {"tooltip": "State bundle from LTX Extend Init."}),
                "total": ("INT", {"default": 14, "min": 1, "max": 100000, "step": 1, "tooltip": "How many extension steps to run."}),
            },
            "optional": {
                "start_index": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "tooltip": "1-based step to start at (resume a dead run here)."}),
            },
            "hidden": {"initial_state": (_LOOP_ANY,)},
        }

    RETURN_TYPES = ("FLOW_CONTROL", _LOOP_ANY, "INT", "INT")
    RETURN_NAMES = ("flow", "state", "index", "attempt")
    FUNCTION = "open"
    CATEGORY = "WhatDreamsCost"

    def open(self, state, total, start_index=1, initial_state=None):
        if initial_state is not None:
            st = initial_state  # re-entry: carried state already has index/attempt/total
        else:
            st = dict(state or {})
            st["index"] = int(start_index)
            st["attempt"] = 0
            st["total"] = int(total)
        idx = int(st.get("index", 1))
        tot = int(st.get("total", 1))
        if idx <= tot:
            log.info("[LTXExtendLoopOpen] step %d/%d (attempt %d)", idx, tot, int(st.get("attempt", 0)))
            return ("stub", st, idx, int(st.get("attempt", 0)))
        blk = _execution_blocker()  # start_index past the end -> run nothing
        return ("stub", blk, blk, blk)


class LTXExtendLoopClose:
    """Exit / iteration driver of the native extend loop. On decision='pass' (default when no Review
    Gate is wired) it folds this step's latent into the state and advances index + audio position; on
    'reroll' it bumps the seed (state.attempt+1) and redoes the step; on 'reload' it redoes the step
    (Step re-pulls its live prompts). Loops until the last step passes, then outputs the final state,
    the last latent, and the number of steps completed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "state": (_LOOP_ANY,),
                "latent": ("LATENT", {"tooltip": "This step's final (second-sampler) latent, or the Review Gate's latent."}),
            },
            "optional": {
                "decision": ("STRING", {"default": "pass", "forceInput": True, "tooltip": "From LTX Review Gate: pass / reroll / reload. Omit for headless auto-pass."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (_LOOP_ANY, "LATENT", "INT")
    RETURN_NAMES = ("state", "final_latent", "steps_done")
    FUNCTION = "close"
    CATEGORY = "WhatDreamsCost"

    # --- graph exploration (ported from the ComfyUI execution-inversion while-loop) ---
    def _explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for value in node_info["inputs"].values():
            if not is_link(value):
                continue
            parent_id = value[0]
            display_id = dynprompt.get_display_node_id(parent_id)
            display_node = dynprompt.get_node(display_id)
            class_type = display_node["class_type"]
            if class_type != "LTXExtendLoopClose":
                parent_ids.append(display_id)
            if parent_id not in upstream:
                upstream[parent_id] = []
                self._explore_dependencies(parent_id, dynprompt, upstream, parent_ids)
            upstream[parent_id].append(node_id)

    def _collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id in contained:
                continue
            contained[child_id] = True
            self._collect_contained(child_id, upstream, contained)

    def close(self, flow, state, latent, decision="pass", dynprompt=None, unique_id=None):
        st = dict(state or {})
        idx = int(st.get("index", 1))
        total = int(st.get("total", 1))
        dec = str(decision or "pass").strip().lower()

        if dec == "reroll":
            st["attempt"] = int(st.get("attempt", 0)) + 1
            cont = True
            log.info("[LTXExtendLoopClose] step %d REROLL -> attempt %d (new seed)", idx, st["attempt"])
        elif dec == "reload":
            cont = True
            log.info("[LTXExtendLoopClose] step %d RELOAD -> redo (re-pull prompt)", idx)
        else:  # pass
            st = _fold_latent(st, latent)
            st["index"] = idx + 1
            st["attempt"] = 0
            cont = (idx + 1) <= total
            log.info("[LTXExtendLoopClose] step %d PASS -> abs_pos %d px; %s",
                     idx, int(st.get("abs_pos_px", 0)), "next step" if cont else "DONE")

        if not cont:
            return (st, st.get("latent", latent), idx)

        # --- re-run the loop body with the updated state (execution-inversion expansion) ---
        if GraphBuilder is None:
            raise RuntimeError("LTX Extend Loop requires ComfyUI's comfy_execution GraphBuilder.")
        upstream = {}
        parent_ids = []
        self._explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))

        graph = GraphBuilder()
        contained = {}
        open_node = flow[0]
        self._collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for key, value in original_node["inputs"].items():
                if is_link(value) and value[0] in contained:
                    parent = graph.lookup_node(value[0])
                    node.set_input(key, parent.out(value[1]))
                else:
                    node.set_input(key, value)

        new_open = graph.lookup_node(open_node)
        new_open.set_input("initial_state", st)  # carry the advanced state into the next iteration
        my_clone = graph.lookup_node("Recurse")
        return {
            "result": (my_clone.out(0), my_clone.out(1), my_clone.out(2)),
            "expand": graph.finalize(),
        }


LOOP_NODE_CLASS_MAPPINGS = {
    "LTXExtendLoopOpen": LTXExtendLoopOpen,
    "LTXExtendLoopClose": LTXExtendLoopClose,
}

LOOP_NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXExtendLoopOpen": "LTX Extend Loop Open",
    "LTXExtendLoopClose": "LTX Extend Loop Close",
}
