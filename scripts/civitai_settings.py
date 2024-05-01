import gradio as gr
from civitai.link import on_civitai_link_key_changed
from civitai import actions
from modules import shared, script_callbacks


class OptionButton(shared.OptionInfo):
    @staticmethod
    def wrapped_button_component(on_click, **kwargs):
        button = gr.Button(**kwargs)
        button.click(fn=on_click)
        return button

    def __init__(self, button_text, on_button_click, **kwargs):
        super().__init__(button_text, component=lambda **component_kwargs: OptionButton.wrapped_button_component(on_button_click, **component_kwargs), **kwargs)
        self.do_not_save = True


def on_ui_settings():
    section = ('civitai_link', "Civitai")
    shared.opts.add_option("civitai_link_key", shared.OptionInfo("", "Your Civitai Link Key", section=section, onchange=on_civitai_link_key_changed))
    shared.opts.add_option("civitai_link_logging", shared.OptionInfo(True, "Show Civitai Link events in the console", section=section))
    shared.opts.add_option("civitai_api_key", shared.OptionInfo("", "Your Civitai API Key", section=section))
    shared.opts.add_option("civitai_download_previews", shared.OptionInfo(True, "Download missing preview images on startup", section=section))
    shared.opts.add_option("civitai_download_triggers", shared.OptionInfo(True, "Download missing activation triggers on startup", section=section))
    shared.opts.add_option("civitai_nsfw_previews", shared.OptionInfo(False, "Download NSFW (adult) preview images", section=section))
    shared.opts.add_option("civitai_download_missing_models", shared.OptionInfo(False, "Download missing models upon reading generation parameters from prompt", section=section))
    shared.opts.add_option("civitai_hashify_resources", shared.OptionInfo(True, "Include resource hashes in image metadata (for resource auto-detection on Civitai)", section=section))
    shared.opts.add_option("civitai_folder_model", shared.OptionInfo("", "Models directory (if not default)", section=section))
    shared.opts.add_option("civitai_folder_lora", shared.OptionInfo("", "LoRA directory (if not default)", section=section))
    shared.opts.add_option("civitai_folder_lyco", shared.OptionInfo("", "LyCORIS directory (if not default)", section=section))
    shared.opts.add_option("civitai_get_previews", OptionButton('get previews', actions.run_load_previews, section=section))
    shared.opts.add_option("civitai_get_metadata", OptionButton('get metadata', actions.run_get_load_info, section=section))
    shared.opts.add_option("civitai_get_previews_metadata", OptionButton('get previews and metadata', actions.run_get_info, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
