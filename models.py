import timm
import torch
from torch import nn
from typing import Literal


class FE_Additional(nn.Module):
    """CNN Feature extractor that can incorporate additional features"""

    def __init__(
        self,
        model_name: str = "seresnet18",
        dropout: float = 0.3,
        additional_features: int = 2,  # Number of additional features (e.g., age and sex)
        pretrained: bool = False,
        device: str = "cuda",
        # sep_metadata_layer: If True, processes age and sex via an extra dense layer
        #   after concatenation; otherwise, adds age and sex directly to the final layer.
        sep_metadata_layer: bool = True,
        activation: Literal["sigmoid"]
        | None = "sigmoid",  # Sigmoid for probability, None for Logit
    ):
        """
        Initialize the feature extractor model.

        Args:
            model_name (str, optional): The name of the model architecture to use. Defaults to "seresnet18".
            dropout (float, optional): Dropout rate for the additional layers. Defaults to 0.3.
            additional_features (int, optional): Number of additional features to incorporate. Defaults to 2.
            device (str, optional): Device to load the model onto (e.g., 'cuda' or 'cpu'). Defaults to "cuda".
            sep_metadata_layer (bool, optional): If True, processes age and sex via an extra dense layer after
                concatenation; otherwise adds age and sex directly to the final layer. Defaults to True.
            activation (Literal["sigmoid"] | None, optional): Activation function for the output layer.
                If "sigmoid", applies sigmoid activation; if None, no activation is applied.

        Raises:
            ValueError: The `activation` must be one of 'sigmoid' or None.
        """

        super().__init__()

        # If we have two classes, we can use a single output neuron with sigmoid activation
        self.additional_features = additional_features
        self.output_dims = 1
        self.sep_metadata_layer = sep_metadata_layer
        self.activation = activation
        if activation not in ["sigmoid", None]:
            raise ValueError(
                f"`activation` must be one of 'sigmoid' or None. Received: {activation}"
            )

        # Load the base model
        self.fe = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        ).to(device)  # Set num_classes to 0 to use as a feature extractor

        # Feature size depends on the base model, need to adjust this based on the specific architecture
        feature_size = self._get_feature_size(model_name)

        # Create additional layers
        if self.sep_metadata_layer:
            self.metadata_layers = nn.Sequential(
                nn.Linear(additional_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
            ).to("cuda")
            self.additional_layers = nn.Sequential(
                nn.Linear(feature_size + 512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, self.output_dims),
            ).to("cuda")
        else:
            self.additional_layers = nn.Sequential(
                nn.Linear(
                    feature_size + additional_features,
                    512,
                ),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, self.output_dims),
            ).to("cuda")

        self.sigmoid_func = nn.Sigmoid()

    def forward(self, x: torch.Tensor, additional_data=torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor (e.g., image data).
            additional_data (torch.Tensor, optional): Additional data tensor (e.g., metadata like
                age and sex). Defaults to None.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """

        # Get the features from the CNN
        features = self.fe(x)

        # Initialize list for concatenation
        feature_list = [features.to(torch.float32)]

        if self.sep_metadata_layer:
            # Separate metadata layer
            metadata_layer_in = []
            if self.additional_features > 0 and additional_data is not None:
                metadata_layer_in.append(additional_data.to(torch.float32))
            if len(metadata_layer_in) > 0:
                combined_metadata_in = torch.cat(metadata_layer_in, dim=1)
                feature_list.append(self.metadata_layers(combined_metadata_in))
        elif self.additional_features > 0 and additional_data is not None:
            feature_list.append(additional_data.to(torch.float32))

        # Concatenate all non-empty tensors in the list
        combined_features = torch.cat(feature_list, dim=1)

        # Pass through additional layers
        output = self.additional_layers(combined_features.to("cuda"))
        # Apply activation function if needed
        if self.activation == "sigmoid":
            output = self.sigmoid_func(output)

        return output.squeeze()

    def _get_feature_size(self, model_name):
        # Create a dummy input tensor. The size should match your input image size.
        # For example, if your input images are 512x512 with 3 channels, the dummy input would be:
        dummy_input = torch.zeros(1, 3, 512, 512).to("cuda")

        # Create a temporary model instance
        temp_model = timm.create_model(model_name, pretrained=False, num_classes=0).to(
            "cuda"
        )

        # Pass the dummy input through the model
        with torch.no_grad():
            temp_model.eval()
            output = temp_model(dummy_input)

        # The output size is the feature size
        feature_size = output.size(1)
        return feature_size
