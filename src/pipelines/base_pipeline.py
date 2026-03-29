from abc import ABC, abstractmethod

class BasePipeline(ABC):
    """
    Abstract base class for video generation pipelines.
    Defines the interface and common utilities for all pipelines.
    """
    @abstractmethod
    def diffuse(self, prompt, id):
        pass