from pydantic import BaseModel

class LangfuseKeyInfo(BaseModel):
    public_key: str
    secret_key: str
