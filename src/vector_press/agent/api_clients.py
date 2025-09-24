from abc import ABC


class BaseAPIClient(ABC):  # ABC prevents creating instances of incomplete classes and forces subclasses to implement all required abstract methods.
