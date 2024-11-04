from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field


class ConversationStyle(str, Enum):
    """Enumeration of conversation styles for podcast generation.

    Parameters
    ----------
    ACADEMIC : str
        Academic style focused on formal discussion and analysis
    STORYTELLING : str
        Narrative-driven style focused on engaging stories
    EXPLORATORY : str
        Open-ended style focused on exploring ideas and concepts
    """

    ACADEMIC = "academic"
    STORYTELLING = "storytelling"
    EXPLORATORY = "exploratory"


class ConversationConfig(BaseModel):
    """Configuration for generating podcast conversations.

    Parameters
    ----------
    word_count : int
        Target word count for the conversation, between 500-5000 words
    conversation_style : list[str]
        List of style descriptors for the conversation tone
    roles_person1 : str
        Role/persona for the first speaker
    roles_person2 : str
        Role/persona for the second speaker
    dialogue_structure : list[str]
        Ordered list of conversation segments/sections
    podcast_name : str
        Name of the podcast
    podcast_tagline : str
        Tagline/description for the podcast
    engagement_techniques : list[str]
        List of techniques to maintain audience engagement
    creativity : float
        Creativity level from 0-1 for content generation
    output_language : str, optional
        Language for the output content, defaults to "English"
    user_instructions : str, optional
        Additional instructions from user, defaults to empty string
    """

    word_count: int = Field(..., ge=100, le=5000)
    conversation_style: list[str]
    roles_person1: str
    roles_person2: str
    dialogue_structure: list[str]
    podcast_name: str
    podcast_tagline: str
    engagement_techniques: list[str]
    creativity: float = Field(..., ge=0, le=1)
    output_language: str = "English"
    user_instructions: str = ""

    STYLE_TEMPLATES: ClassVar[dict[ConversationStyle, dict]] = {
        ConversationStyle.ACADEMIC: {
            "word_count": 3000,
            "conversation_style": ["formal", "analytical", "critical"],
            "roles_person1": "thesis presenter",
            "roles_person2": "counterargument provider",
            "dialogue_structure": [
                "Opening Statements",
                "Thesis Presentation",
                "Counterarguments",
                "Rebuttals",
                "Closing Remarks",
            ],
            "podcast_name": "Scholarly Showdown",
            "podcast_tagline": "Where Ideas Clash and Knowledge Emerges",
            "engagement_techniques": [
                "socratic questioning",
                "historical references",
                "thought experiments",
            ],
            "creativity": 0,
            "user_instructions": "",
        },
        ConversationStyle.STORYTELLING: {
            "word_count": 1000,
            "conversation_style": ["narrative", "suspenseful", "descriptive"],
            "roles_person1": "storyteller",
            "roles_person2": "audience participator",
            "dialogue_structure": [
                "Scene Setting",
                "Character Introduction",
                "Rising Action",
                "Climax",
                "Resolution",
            ],
            "podcast_name": "Tale Spinners",
            "podcast_tagline": "Where Every Episode is an Adventure",
            "engagement_techniques": [
                "cliffhangers",
                "vivid imagery",
                "audience prompts",
            ],
            "creativity": 0.9,
            "user_instructions": "",
        },
        ConversationStyle.EXPLORATORY: {
            "word_count": 2000,
            "conversation_style": ["curious", "analytical", "open-minded"],
            "roles_person1": "explorer",
            "roles_person2": "co-explorer",
            "dialogue_structure": [
                "Topic Introduction",
                "Initial Observations",
                "Deep Dive Questions",
                "Alternative Perspectives",
                "Synthesis and New Questions",
            ],
            "podcast_name": "Curiosity Quest",
            "podcast_tagline": "Exploring Ideas Together",
            "engagement_techniques": [
                "thought experiments",
                "what-if scenarios",
                "connecting concepts",
                "challenging assumptions",
            ],
            "creativity": 0.7,
            "user_instructions": "",
        },
    }


class TTSVoices(BaseModel):
    """Configuration for text-to-speech voice settings.

    Parameters
    ----------
    question : str
        Voice ID/settings for questions
    answer : str
        Voice ID/settings for answers
    """

    question: str
    answer: str


class TTSConfig(BaseModel):
    """Configuration for a text-to-speech service.

    Parameters
    ----------
    default_voices : TTSVoices
        Default voice settings for the TTS service
    model : str
        Model identifier for the TTS service
    """

    default_voices: TTSVoices
    model: str


class TextToSpeechConfig(BaseModel):
    """Master configuration for text-to-speech settings.

    Parameters
    ----------
    default_tts_model : str, optional
        Default TTS service to use, defaults to "openai"
    elevenlabs : TTSConfig
        Configuration for ElevenLabs TTS service
    openai : TTSConfig
        Configuration for OpenAI TTS service
    edge : TTSConfig
        Configuration for Edge TTS service
    audio_format : str, optional
        Output audio format, defaults to "mp3"
    ending_message : str, optional
        Message to append at end of audio, defaults to "Bye Bye!"
    """

    default_tts_model: str = "openai"
    elevenlabs: TTSConfig
    openai: TTSConfig
    edge: TTSConfig
    audio_format: str = "mp3"
    ending_message: str = "Bye Bye!"
