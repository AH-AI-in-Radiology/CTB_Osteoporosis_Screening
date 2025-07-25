import json
from typing import Literal

import numpy as np
import pydicom
import torch

from monai.transforms import Resize, ResizeWithPadOrCrop

from models import FE_Additional


def load_config(config_path: str) -> dict:
    """
    Load the model configuration from a JSON file.

    Args:
        config_path (str): Path to the model configuration json file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_model(config: dict, model_path: str, device: torch.device) -> FE_Additional:
    """
    Load a PyTorch model from the specified path and move it to the given device.

    Args:
        config_path (str): Path to the model configuration json file.
        model_path (str): Path to the model file.
        device (torch.device): Device to load the model onto (e.g., 'cuda' or 'cpu').

    Returns:
        torch.nn.Module: The loaded model.
    """

    # Create the model instance with the loaded configuration
    model = FE_Additional(
        model_name=config.get("model_name", "seresnet18"),
        dropout=config.get("dropout", 0.3),
        additional_features=config.get("additional_features", 2),
        pretrained=config.get("pretrained", False),
        device=device,
        sep_metadata_layer=config.get("sep_metadata_layer", False),
        activation=config.get("activation", "sigmoid"),
    ).to(device)

    # Load the model state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device)[0])

    return model


def create_metadata_tensor(
    age: int,
    sex: Literal["male", "female"],
    age_mean: float,
    age_std: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a metadata tensor from age and sex features and normalize the age.

    Args:
        age (int): The age of the patient.
        sex (Literal["male", "female"]): The sex of the patient.
        age_mean (float): The mean age for normalisation.
        age_std (float): The standard deviation of age for normalisation.
        device (torch.device): Device to place the tensor on (e.g., 'cuda' or 'cpu').

    Raises:
        ValueError: If `sex` is not in male or female.

    Returns:
        torch.Tensor: A tensor containing the normalized age and sex.
    """

    if sex not in ["male", "female"]:
        raise ValueError(f"`sex` must be either male or female. Received: {sex}")

    # Convert sex to a numerical value
    sex = int(sex == "male")
    # Create a tensor with age and sex
    normalised_age = (age - age_mean) / age_std
    metadata = torch.tensor([normalised_age, sex], dtype=torch.float32)
    metadata = metadata.to(device)  # Move to the specified device
    return metadata


def load_slices(slice_paths: list[str]) -> list[np.array]:
    """
    Load DICOM slices from the specified paths.

    Args:
        slice_paths (list[str]): List of paths to the DICOM slices.

    Returns:
        list[np.array]: List of loaded DICOM slices.
    """

    slices = []
    for path in slice_paths:
        ct_slice = pydicom.dcmread(path)
        slices.append(ct_slice)

    return slices


def transform_to_hu(slices: list[np.array]) -> list[np.array]:
    """
    Transform DICOM slices to Hounsfield Units (HU).

    Args:
        slices (list[np.array]): List of DICOM slices to be transformed.

    Returns:
        list[np.array]: List of transformed slices in Hounsfield Units.
    """

    return [
        pydicom.pixel_data_handlers.apply_modality_lut(slice_.pixel_array, slice_)
        for slice_ in slices
    ]


def window(
    slices: list[np.array], window_center: int, window_width: int
) -> list[np.array]:
    """
    Apply a windowing operation to the slices.

    Args:
        slices (list[np.array]): List of slices to be windowed.
        window_center (int): Center of the window.
        window_width (int): Width of the window.

    Returns:
        list[np.array]: List of windowed slices.
    """

    lower_bound = window_center - (window_width / 2)
    upper_bound = window_center + (window_width / 2)

    slices = [np.clip(slice_, lower_bound, upper_bound) for slice_ in slices]
    return slices


def tensor_stack(slices: list[np.array], device: torch.device) -> torch.Tensor:
    """
    Convert a list of numpy arrays to a list of PyTorch tensors.

    Args:
        slices (list[np.array]): List of numpy arrays to be converted.
        device (torch.device): Device to place the tensors on (e.g., 'cuda' or 'cpu').

    Returns:
        list[torch.Tensor]: List of PyTorch tensors.
    """

    slices = torch.tensor(np.stack(slices, axis=0), dtype=torch.float32)
    slices = slices.to(device)  # Move to the specified device
    return slices


def zoom_resize(slices: torch.Tensor, dims: tuple[int, int]) -> torch.Tensor:
    """
    Resize slices to the specified dimensions.

    Args:
        slices (torch.Tensor): Tensor of slices to be resized.
        dims (tuple[int, int]): Target dimensions for resizing (height, width).

    Returns:
        torch.Tensor: Resized tensor of slices.
    """

    resize_func = Resize(max(dims), size_mode="longest")
    pad_func = ResizeWithPadOrCrop(
        spatial_size=dims, constant_values=slices.min(), mode="constant"
    )
    t = pad_func(resize_func(slices))

    return t


def normalise_slices(slices: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Normalize the slices using the specified mean and standard deviation.

    Args:
        slices (torch.Tensor): Tensor of slices to be normalized.
        mean (float): Mean value for normalization.
        std (float): Standard deviation value for normalization.
    Returns:
        torch.Tensor: Normalized tensor of slices.
    """
    if std <= 0:
        raise ValueError(
            "Standard deviation must be greater than zero for normalization."
        )

    return (slices - mean) / std
