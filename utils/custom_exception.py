"""
Custom exceptions for the Tourism Recommendation System.
Provides clear, descriptive error handling across all modules.
"""


class TourismRecBaseException(Exception):
    """Base exception for all Tourism Recommendation errors."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ─────────────────────────── Data Exceptions ───────────────────────────

class DataGenerationError(TourismRecBaseException):
    """Raised when dataset generation fails."""
    pass


class DataNotFoundError(TourismRecBaseException):
    """Raised when required data files are missing."""
    pass


class DataValidationError(TourismRecBaseException):
    """Raised when data fails validation checks."""
    pass


class PreprocessingError(TourismRecBaseException):
    """Raised when data preprocessing fails."""
    pass


# ─────────────────────────── Model Exceptions ───────────────────────────

class ModelNotFoundError(TourismRecBaseException):
    """Raised when a saved model checkpoint is not found."""
    pass


class ModelTrainingError(TourismRecBaseException):
    """Raised when model training encounters an error."""
    pass


class ModelInferenceError(TourismRecBaseException):
    """Raised when model prediction/inference fails."""
    pass


class InvalidHyperparameterError(TourismRecBaseException):
    """Raised when invalid hyperparameters are provided."""
    pass


# ─────────────────────────── Config Exceptions ───────────────────────────

class ConfigurationError(TourismRecBaseException):
    """Raised when configuration is invalid or missing."""
    pass


# ─────────────────────────── App Exceptions ───────────────────────────

class AppInitializationError(TourismRecBaseException):
    """Raised when the Streamlit app fails to initialize."""
    pass


class RecommendationError(TourismRecBaseException):
    """Raised when recommendation generation fails."""
    pass
