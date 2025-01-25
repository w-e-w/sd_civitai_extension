import gradio as gr
from civitai_ext import actions, opencc_utils, mhtml_button
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
    # shared.opts.add_option("civitai_nsfw_previews", shared.OptionInfo(True, "Download NSFW (adult) preview images", section=section))
    shared.opts.add_option("civitai_get_previews_metadata", OptionButton('get metadata and preview', actions.run_get_info, section=section))
    shared.opts.add_option("civitai_get_metadata", OptionButton('get metadata', actions.load_info, section=section))
    shared.opts.add_option("civitai_get_previews", OptionButton('get preview', actions.load_previews_v2, section=section))
    shared.opts.add_option("civitai_convert_chinese", shared.OptionInfo('Disable', 'Convert chinese characters auto-generated description', gr.Dropdown, lambda: {'choices': opencc_utils.read_config()}, section=section, refresh=opencc_utils.install_opencc))
    # shared.opts.add_option("civitai_re_preview", OptionButton('re download previews from cache', actions.re_download_preview_from_cache, section=section))
    # shared.opts.add_option("civitai_open_local_mhtml", shared.OptionInfo(True, "Add button to open local mhtml if available", section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(mhtml_button.add_api)
