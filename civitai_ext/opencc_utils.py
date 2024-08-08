from modules import shared, launch_utils, errors
import json
import os


def install_opencc():
    try:
        import opencc
    except ImportError:
        try:
            launch_utils.run_pip('install OpenCC', 'OpenCC')
        except Exception:
            errors.report(f'Civitai: Error installing OpenCC', exc_info=True)


def read_config():
    config_dsc = ['Disable']
    try:
        import opencc
        for config in opencc.CONFIGS:
            try:
                config_name = os.path.splitext(config)[0]
                with open(os.path.join(opencc._opencc_share_dir, config), 'r', encoding='utf-8') as f:
                    config_dsc.append(f"{config_name}: {json.load(f).get('name', config_name)}")
            except Exception as e:
                print(f'Civitai: Error reading OpenCC config {config}: {e}')
    except ImportError:
        config_dsc.append('Click refresh to install OpenCC')
    return config_dsc


class Placeholder:
    @staticmethod
    def convert(text):
        return text


def converter():
    if (config := shared.opts.civitai_convert_chinese.partition(':')[0]) == 'Disable' or not config:
        return Placeholder
    try:
        install_opencc()
        import opencc
        return opencc.OpenCC(config)
    except Exception:
        errors.report('Civitai: Error initializing OpenCC', exc_info=True)
        return Placeholder
