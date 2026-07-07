from .ltx_keyframer import LTXKeyframer
from .multi_image_loader import MultiImageLoader
from .ltx_sequencer import LTXSequencer
from .speech_length_calculator import SpeechLengthCalculator
from .load_audio_ui import LoadAudioUI
from .load_video_ui import LoadVideoUI
from .ltx_director import LTXDirector, LTXKeyframeOut, LTXAutoExtend, LTXExtendInit, LTXExtendStep, LTXExtendCollect, LTXReviewGate
from .ltx_director_guide import LTXDirectorGuide, LTXDirectorCropGuides, LTXICLoraSelector
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTXDirector,
            LTXDirectorGuide,
            LTXKeyframeOut,
            LTXAutoExtend
        ]

async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()
    
NODE_CLASS_MAPPINGS = {
    "LTXKeyframer": LTXKeyframer,
    "MultiImageLoader": MultiImageLoader,
    "LTXSequencer": LTXSequencer,
    "SpeechLengthCalculator": SpeechLengthCalculator,
    "LoadAudioUI": LoadAudioUI,
    "LoadVideoUI": LoadVideoUI,
    "LTXDirector": LTXDirector,
    "LTXKeyframeOut": LTXKeyframeOut,
    "LTXAutoExtend": LTXAutoExtend,
    "LTXExtendInit": LTXExtendInit,
    "LTXExtendStep": LTXExtendStep,
    "LTXExtendCollect": LTXExtendCollect,
    "LTXReviewGate": LTXReviewGate,
    "LTXDirectorGuide": LTXDirectorGuide,
    "LTXDirectorCropGuides": LTXDirectorCropGuides,
    "LTXICLoraSelector": LTXICLoraSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXKeyframer": "LTX Keyframer",
    "MultiImageLoader": "Multi Image Loader",
    "LTXSequencer": "LTX Sequencer",
    "SpeechLengthCalculator": "Speech Length Calculator",
    "LoadAudioUI": "Load Audio UI",
    "LoadVideoUI": "Load Video UI",
    "LTXDirector": "LTX Director",
    "LTXKeyframeOut": "LTX Keyframe Out",
    "LTXAutoExtend": "LTX Auto Extend",
    "LTXExtendInit": "LTX Extend Init",
    "LTXExtendStep": "LTX Extend Step",
    "LTXExtendCollect": "LTX Extend Collect",
    "LTXReviewGate": "LTX Review Gate",
    "LTXDirectorGuide": "LTX Director Guide",
    "LTXDirectorCropGuides": "LTX Director Crop Guides",
    "LTXICLoraSelector": "LTX IC-LoRA Selector",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']