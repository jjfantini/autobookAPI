# core/podcast.py
import asyncio
import io
from collections.abc import AsyncGenerator
from typing import Any

from podcastfy.client import generate_podcast
from pydantic import BaseModel, HttpUrl

from autobook.core.standard_models.conversation_model import (
    ConversationConfig,
    ConversationStyle,
)


class PodcastGenerator:
    def __init__(self):
        self._config_templates = ConversationConfig.STYLE_TEMPLATES

    async def generate_podcast_stream(
        self,
        style: ConversationStyle,
        urls: list[str] | None = None,
        text: str | None = None,
        custom_config: dict | None = None,
        tts_model: str = "openai",
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate a podcast with streaming response.

        Args:
            style: The conversation style to use
            urls: Optional list of URLs to generate content from
            text: Optional raw text to generate content from
            custom_config: Optional dict for customizing generation
            tts_model: Text-to-speech model to use (default: "openai")
        """
        # Get base config for selected style
        base_config = self._config_templates[style].copy()

        # Create a complete config dict by merging base with custom
        if custom_config:
            config_dict = {**base_config, **custom_config}
        else:
            config_dict = base_config

        try:
            # Run podcastfy generation in a thread pool to avoid blocking
            audio_file = await asyncio.to_thread(
                generate_podcast,
                urls=urls,
                text=text,
                conversation_config=config_dict,
                tts_model=tts_model,
            )

            # Stream the generated audio file
            chunk_size = 1024 * 8  # 8KB chunks
            if audio_file:  # Add null check
                with open(audio_file, "rb") as f:
                    while chunk := f.read(chunk_size):
                        yield chunk
            else:
                raise Exception("No audio file was generated")

        except Exception as e:
            raise Exception(f"Failed to generate podcast: {e!s}")
