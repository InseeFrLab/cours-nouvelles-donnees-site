"""
Utils.
"""
from pathlib import Path
import warnings
from constants import TEXT_FEATURE, Y


def get_root_path() -> Path:
    """
    Returns root path of project.

    Returns:
        Path: Root path.
    """
    return Path(__file__).parent.parent


def write_training_data(df, params, training_data_path=None):
    warnings.filterwarnings("ignore", "Setuptools is replacing distutils.")
    if training_data_path is None:
        training_data_path = get_root_path() / "data/training_data.txt"

    with open(training_data_path, "w", encoding="utf-8") as file:
        for _, item in df.iterrows():
            formatted_item = f"{params['label_prefix']}{item[Y]} {item[TEXT_FEATURE]}"
            file.write(f"{formatted_item}\n")
    return training_data_path.as_posix()