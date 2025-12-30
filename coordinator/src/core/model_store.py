"""
Model Store Module

Handles filesystem-based persistence of global models.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


class ModelStore:
    """
    Manages model persistence to filesystem.
    
    Stores models as JSON files in the coordinator/models/ directory.
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the model store.
        
        Args:
            models_dir: Directory path for storing models.
                       Defaults to coordinator/models/ relative to this file.
        """
        if models_dir is None:
            # Get the coordinator directory (parent of src)
            current_file = Path(__file__)
            coordinator_dir = current_file.parent.parent.parent
            models_dir = str(coordinator_dir / "models")
        
        self.models_dir = Path(models_dir)
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Ensure the models directory exists."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_model_path(self, version: str) -> Path:
        """
        Get the file path for a model version.
        
        Args:
            version: Model version string (e.g., "v1", "v2")
            
        Returns:
            Path object for the model file
        """
        # Sanitize version string for filename
        filename = f"model_{version}.json"
        return self.models_dir / filename
    
    def save_model(self, version: str, model_data: Dict) -> None:
        """
        Save a model to disk.
        
        Args:
            version: Model version string (e.g., "v1", "v2")
            model_data: Dictionary containing model data
            
        Raises:
            ValueError: If version format is invalid
            IOError: If file write fails
        """
        if not version.startswith("v"):
            raise ValueError(f"Invalid version format: {version}. Must start with 'v'")
        
        model_path = self._get_model_path(version)
        
        try:
            # Write model data as JSON
            with open(model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to save model {version}: {e}")
    
    def load_model(self, version: str) -> Dict:
        """
        Load a model from disk.
        
        Args:
            version: Model version string (e.g., "v1", "v2")
            
        Returns:
            Dictionary containing model data
            
        Raises:
            FileNotFoundError: If model file does not exist
            ValueError: If model file is corrupted or invalid
        """
        model_path = self._get_model_path(version)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model version {version} not found at {model_path}")
        
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            return model_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Model file {model_path} is corrupted: {e}")
        except Exception as e:
            raise IOError(f"Failed to load model {version}: {e}")
    
    def latest_model_version(self) -> Optional[str]:
        """
        Get the latest model version available on disk.
        
        Returns:
            Latest version string (e.g., "v3") or None if no models exist
        """
        if not self.models_dir.exists():
            return None
        
        # Find all model files
        model_files = list(self.models_dir.glob("model_v*.json"))
        
        if not model_files:
            return None
        
        # Extract version numbers and find maximum
        versions = []
        for model_file in model_files:
            # Extract version from filename: model_v1.json -> v1
            filename = model_file.stem  # "model_v1"
            version_str = filename.replace("model_", "")
            
            # Validate version format
            if version_str.startswith("v") and version_str[1:].isdigit():
                versions.append(version_str)
        
        if not versions:
            return None
        
        # Sort by version number
        versions.sort(key=lambda v: int(v[1:]))
        
        return versions[-1]
    
    def model_exists(self, version: str) -> bool:
        """
        Check if a model version exists on disk.
        
        Args:
            version: Model version string (e.g., "v1", "v2")
            
        Returns:
            True if model exists, False otherwise
        """
        model_path = self._get_model_path(version)
        return model_path.exists()
    
    def list_models(self) -> list:
        """
        List all available model versions.
        
        Returns:
            List of version strings (e.g., ["v1", "v2", "v3"])
        """
        if not self.models_dir.exists():
            return []
        
        model_files = list(self.models_dir.glob("model_v*.json"))
        versions = []
        
        for model_file in model_files:
            filename = model_file.stem
            version_str = filename.replace("model_", "")
            
            if version_str.startswith("v") and version_str[1:].isdigit():
                versions.append(version_str)
        
        # Sort by version number
        versions.sort(key=lambda v: int(v[1:]))
        
        return versions

