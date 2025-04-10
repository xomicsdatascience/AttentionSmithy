from abc import ABC, abstractmethod
import torch
from torch import nn

class NumericEmbeddingStrategyBase(ABC, nn.Module):
    """
    Abstract base class for embedding strategies.
    Subclasses must implement the create_positional_or_custom_embedding method.
    Examples include sinusoidal and learned positional embeddings.
    """

    @abstractmethod
    def create_positional_or_custom_embedding(self, **kwargs) -> torch.Tensor:
        """
        Generate a positional or custom embedding based on the given input data.

        Returns:
            torch.Tensor: The output tensor represents positional or custom value encoding.
        """
        pass

class MatrixModificationStrategyBase(ABC, nn.Module):
    """
    Abstract base class for strategies that modify or transform matrices.
    Subclasses must implement the modify_matrices method.
    Examples include rotary positional embeddings.
    """

    @abstractmethod
    def modify_matrix(self, target_matrix, **kwargs) -> torch.Tensor:
        """
        Modify or transform the given matrices.

        Returns:
            Modified versions of the provided matrices.
        """
        pass

class AttentionBiasStrategyBase(ABC, nn.Module):
    """
    Abstract base class for strategies that generate bias tensors to be added to attention score matrices.
    Subclasses must implement the create_bias_tensor method.
    Examples include ALiBi positional embeddings.
    """

    @abstractmethod
    def create_bias_tensor(self, **kwargs) -> torch.Tensor:
        """
        Create a bias tensor that can be added to attention score matrices.

        Returns:
            A bias tensor of the same shape.
        """
        pass