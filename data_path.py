from pathlib import Path

def get_data_path(path):
    def get_project_root() -> Path:

        return Path(__file__).parent.parent

    return get_project_root().joinpath(path)

