from pydantic import BaseModel, Field


class Answer(BaseModel):

    answer: str = Field(
        description="The final answer to the user's question."
    )

    context: str = Field(
        description="The relevant retrieved context used to generate the answer. "
                    "Include the supporting information from the provided context."
    )
