from pydantic import BaseModel


class TextToImgModel(BaseModel):
    text: str
