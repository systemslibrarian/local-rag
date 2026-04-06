from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    answer: str | None = Field(default=None, description="Answer on the question",)


class Citation(BaseModel):
    file_name: str = Field(..., description="Source file name")
    page: int | None = Field(default=None, description="1-based page number")
    excerpt: str | None = Field(default=None, description="Relevant excerpt from the source")


class AIAnswer(BaseModel):
    answer: str = Field(..., description="Grounded answer")
    citations: list[Citation] = Field(default_factory=list, description="Supporting citations")