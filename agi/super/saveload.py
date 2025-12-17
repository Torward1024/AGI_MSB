# agi/super/saveload.py
from common.super.super import Super
from typing import Dict, Any
import torch
import json
from agi.project import GraphProject
from common.utils.logging_setup import logger

class SaveLoadSuper(Super):
    """Super class for saving and loading AGI state.

    Handles serialization of entire GraphProject, including models and memory.
    """
    OPERATION = "saveload"

    def __init__(self, manipulator):
        super().__init__(manipulator=manipulator)

    def _saveload_graphproject(self, project: GraphProject, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Save or load based on mode."""
        mode = attributes.get("mode")
        path = attributes.get("path")
        if mode == "save":
            data = project.to_dict()
            with open(path + ".json", "w") as f:
                json.dump(data, f, indent=4)
            # Save models separately if needed; here assumed in to_dict
            logger.info(f"Saved AGI state to '{path}'")
            return self._build_response(project, True, "_saveload_graphproject", {"saved": path})
        elif mode == "load":
            with open(path + ".json", "r") as f:
                data = json.load(f)
            loaded_project = GraphProject.from_dict(data)
            self._manipulator.set_managing_object(loaded_project)
            logger.info(f"Loaded AGI state from '{path}'")
            return self._build_response(project, True, "_saveload_graphproject", {"loaded": path})
        else:
            raise ValueError("Mode must be 'save' or 'load'")

    def _saveload(self, obj: Any, attributes: Dict[str, Any]) -> Dict[str, Any]:
        return self._default_result(obj)