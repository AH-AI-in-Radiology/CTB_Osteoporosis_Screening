import json
from typing import Literal

import numpy as np
import pydicom
import torch
from pathlib import Path

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

def load_slices(dir_path: str) -> list[np.array]:
    """
    Load DICOM slices from the specified directory and sort them.

    Args:
        dir_path (str): Path to the directory containing the DICOM slices.

    Returns:
        list[np.array]: List of loaded DICOM slices.
    """

    # Get the slice file paths
    slice_paths = sorted(Path(dir_path).glob("*.dcm"))
    if len(slice_paths) == 0:
        raise ValueError(f"No DICOM files found in directory: {dir_path}")

    # Load the slices
    slices = []
    for path in slice_paths:
        ct_slice = pydicom.dcmread(path)
        slices.append(ct_slice)

    # Sort slices by InstanceNumber to ensure correct order
    slices.sort(key=lambda x: int(x.InstanceNumber))

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


def create_channels(slices: torch.Tensor) -> torch.Tensor:
    """
    Take a tensor of slices and create a 3-channel tensor with overlapping slices.

    Args:
        slices (torch.Tensor): Tensor of slices to be converted.

    Returns:
        torch.Tensor: Tensor containing the 3-channel overlapping slices with shape (batch, 3, 512, 512).
    """

    # Create three channels that are overlapping slices
    slices = slices.unfold(0, size=3, step=1)
    # Reshape to (batch, 3, 512, 512)
    slices = slices.reshape(-1, 3, 512, 512)
    return slices


def calculate_optimal_slice(mask: np.array) -> int:
    """
    Calculate the optimal slice index above the lateral ventricles based on segmentation data.

    Args:
        mask (np.array): 3D numpy array where each slice corresponds to a segmentation mask
            generated from a CT scan.
    Returns:
        int: The index of the optimal slice above the lateral ventricles.
    """

    # Count the number of pixels in each slice corresponding with label 10
    # (ventricular system)
    pixel_count = np.zeros((len(mask)))
    for i, array in enumerate(mask):
        pixel_count[i] = np.sum(array == 10)

    max_pixel_count_index = np.argmax(pixel_count)

    # Define a slice limit to to stop looking
    # Defined as 1/5 of the series length from the slice with highest pixel
    # count corresponding with the ventricular system
    sequence_length = len(pixel_count)
    window_size = int(sequence_length * 0.2)
    window_end = max_pixel_count_index + window_size
    # Ensure the window doesn't exceed the bounds of the entire scan
    window_end = min(window_end, sequence_length)

    # Identify the minimum value in the defined window
    min_val = min(pixel_count[max_pixel_count_index:window_end])

    # Set the threshold value of to be 5% of the max pixel count
    threshold_value = pixel_count[max_pixel_count_index] / 20

    # Adjust the window end to not exceed the array bounds
    adjusted_window_end = min(max_pixel_count_index + window_size, len(pixel_count))

    if min_val <= threshold_value:
        # Look for the first value below 1/20 of the max value within the window
        for idx in range(max_pixel_count_index, adjusted_window_end):
            if pixel_count[idx] < threshold_value:
                optimal_slice_idx = idx
                break
    else:
        # Catch situation where all pixel counts are above threshold in the window
        optimal_slice_idx = max_pixel_count_index + np.argmin(
            pixel_count[max_pixel_count_index:window_end]
        )

    return optimal_slice_idx


def extract_optimal_range(slices: np.array, optimal_slice_idx: int) -> np.array:
    """
    Extract a range of slices around the optimal slice index.

    Args:
        slices (np.array): 3D numpy array where each slice corresponds to a CT scan slice.
        optimal_slice_idx (int): The index of the optimal slice above the lateral ventricles.
    Returns:
        np.array: 3D numpy array containing the extracted range of slices.
    """

    start_idx = max(0, optimal_slice_idx - 6)
    end_idx = min(len(slices) - 1, optimal_slice_idx + 6)

    return slices[start_idx:end_idx]
