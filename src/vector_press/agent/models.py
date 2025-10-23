"""
Shared data models for the agent system.

This module contains Pydantic models that are used across multiple modules
to avoid circular imports. Keep this file dependency-free - it should only
import from external libraries (pydantic, typing, etc.), never from other
agent modules.
"""

from pydantic import BaseModel
from typing import Literal


class ToDo(BaseModel):
    """
    A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """

    content: str
    status: Literal['pending', 'in_progress', 'completed']