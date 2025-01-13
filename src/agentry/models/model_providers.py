from pydantic import BaseModel

class OpenAiCompatibleApiConfig(BaseModel):
    url_base: str 
    api_key: str