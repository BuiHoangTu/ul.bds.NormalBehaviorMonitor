from typing import Any, Protocol


class IAutoencoder(Protocol):
    def encode(self, x: Any) -> Any:
        """Encodes the input data."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def decode(self, latent: Any) -> Any:
        """Decodes the encoded data back to the original space."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class ITrainableByLayer(Protocol):
    def getNStages(self) -> int:
        """Returns the number of stages in the model."""
        return 0

    def getStage(self, stage: int) -> Any:
        """Returns the model for the specified stage."""
        raise NotImplementedError("This method should be implemented by subclasses.")
