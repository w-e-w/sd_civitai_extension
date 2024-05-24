import gradio as gr
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
    shared.opts.add_option("civitai_nsfw_previews", shared.OptionInfo(True, "Download NSFW (adult) preview images", section=section))
    shared.opts.add_option("civitai_get_previews", OptionButton('get previews', actions.run_load_previews, section=section))
    shared.opts.add_option("civitai_get_metadata", OptionButton('get metadata', actions.run_get_load_info, section=section))
    shared.opts.add_option("civitai_get_previews_metadata", OptionButton('get previews and metadata', actions.run_get_info, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
