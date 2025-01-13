from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class StreamOptions(BaseModel):
    include_usage: bool = False
