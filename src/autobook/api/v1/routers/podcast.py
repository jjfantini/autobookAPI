# api/v1/routers/podcast.py
import io
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl, model_validator

from autobook.core.podcast_generator import PodcastGenerator
from autobook.core.standard_models.conversation_model import (
    ConversationConfig,
    ConversationStyle,
)

router = APIRouter(prefix="/api/v1/podcast", tags=["podcast"])
podcast_generator = PodcastGenerator()


class PodcastGenerationRequest(BaseModel):
    style: ConversationStyle
    urls: list[str] | None = None
    text: str | None = None
    custom_config: dict | None = None
    tts_model: str = "openai"

    @model_validator(mode="after")
    def validate_model(self) -> "PodcastGenerationRequest":
        if not self.text and not self.urls:
            raise ValueError("Either 'text' or 'urls' must be provided")
        return self


class PodcastGenerationResponse(BaseModel):
    status: str
    message: str
    audio_url: str | None = None


@router.post("/generate-podcast", response_model=PodcastGenerationResponse)
async def generate_podcast(
    request: PodcastGenerationRequest,
) -> StreamingResponse:
    try:
        # Validate style and get template
        if request.style not in ConversationConfig.STYLE_TEMPLATES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid style. Choose from: {list(ConversationConfig.STYLE_TEMPLATES.keys())}",
            )

        # Initialize audio buffer
        audio_buffer = io.BytesIO()

        # Create generator for streaming
        async def generate():
            async for chunk in podcast_generator.generate_podcast_stream(
                style=request.style,
                urls=request.urls,
                text=request.text,
                custom_config=request.custom_config,
                tts_model=request.tts_model,
            ):
                # Write to buffer and yield chunk
                audio_buffer.write(chunk)
                yield chunk

        # Return streaming response
        return StreamingResponse(
            generate(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=podcast_{request.style.value}.mp3"
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating podcast: {e!s}"
        )


@router.get("/styles", response_model=list[str])
async def get_available_styles():
    """Get list of available podcast styles."""
    return [style.value for style in ConversationStyle]


@router.get("/style/{style}/template")
async def get_style_template(style: ConversationStyle):
    """Get the configuration template for a specific style."""
    if style not in ConversationConfig.STYLE_TEMPLATES:
        raise HTTPException(status_code=404, detail=f"Style {style} not found")
    return ConversationConfig.STYLE_TEMPLATES[style]
