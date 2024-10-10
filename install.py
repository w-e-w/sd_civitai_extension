import launch

if not launch.is_installed("filetype"):
    launch.run_pip("install filetype ", "requirements for civitai extension")
