"""
A simplified content transformation API that converts various content types
into either podcasts or books.
"""

import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi


class ContentType(Enum):
    """Supported content types for transformation"""

    TEXT = "text"
    URL = "url"
    YOUTUBE = "youtube"
    PDF = "pdf"
    QUERY = "query"  # New type for text queries


class OutputType(Enum):
    """Supported output formats"""

    PODCAST = "podcast"
    BOOK = "book"


@dataclass
class ContentItem:
    """Represents a piece of content to be processed"""

    content_type: ContentType
    source: str
    extracted_text: str | None = None


class ContentTransformer:
    """Main class for transforming content into podcasts or books"""

    def __init__(self, openai_api_key: str):
        """Initialize the transformer with necessary API keys"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def _generate_content_from_query(
        self, query: str, min_words: int = 500
    ) -> str:
        """Generate detailed content based on a text query using GPT-4"""
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable expert tasked with creating detailed, "
                        "well-structured content about a given topic. Include relevant facts, "
                        "examples, and explanations. Aim to be informative yet engaging. "
                        f"The content should be approximately {min_words} words long."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Create detailed content about the following topic: {query}",
                },
            ],
            max_tokens=2000,
            temperature=0.7,
        )
        return str(response.choices[0].message.content)

    def _extract_youtube_id(self, url: str) -> str:
        """Extract YouTube video ID from URL"""
        parsed = urlparse(url)
        if "youtu.be" in parsed.netloc:
            return parsed.path[1:]
        if "v=" in parsed.query:
            return parsed.query.split("v=")[1].split("&")[0]
        return ""

    def _extract_from_youtube(self, url: str) -> str:
        """Extract transcript from YouTube video"""
        video_id = self._extract_youtube_id(url)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(entry["text"] for entry in transcript)
        except Exception as e:
            raise ValueError(f"Failed to extract YouTube transcript: {e}")

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract PDF content: {e}")

    def _extract_from_url(self, url: str) -> str:
        """Extract text content from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            raise ValueError(f"Failed to extract URL content: {e}")

    def _process_content(self, item: ContentItem) -> str:
        """Process content based on its type"""
        if item.content_type == ContentType.YOUTUBE:
            return self._extract_from_youtube(item.source)
        if item.content_type == ContentType.PDF:
            return self._extract_from_pdf(item.source)
        if item.content_type == ContentType.URL:
            return self._extract_from_url(item.source)
        if item.content_type == ContentType.TEXT:
            return item.source
        if item.content_type == ContentType.QUERY:
            return self._generate_content_from_query(item.source)
        raise ValueError(f"Unsupported content type: {item.content_type}")

    def _generate_summary(self, text: str, max_tokens: int = 1000) -> str:
        """Generate a summary of the content using OpenAI"""
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize the following text:"},
                {"role": "user", "content": text},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _generate_audio(self, text: str, output_path: str, voice: str) -> None:
        """Generate audio for a single line of text."""
        response = self.openai_client.audio.speech.create(
            model="tts-1", voice=voice, input=text
        )
        response.stream_to_file(output_path)

    def _generate_interview_transcript(
        self, content: str, transcript_length: int = 1000
    ) -> str:
        """
        Generate an interview-style transcript from the content using GPT-4.
        Creates a natural conversation between an interviewer and an expert.

        Args:
            content: The content to transform into a conversation
            transcript_length: Target length of the transcript in words
        """
        prompt = f"""
        Transform the following content into a natural, engaging interview transcript
        between two people. Format the conversation using XML-style tags for each speaker.
        The final transcript should be approximately {transcript_length} words long.

        Format the conversation as:
        <Person1>Host's question or comment</Person1>
        <Person2>Expert's detailed response</Person2>

        Make the conversation flow naturally, with the host asking follow-up questions
        and occasionally summarizing key points. The expert should provide detailed,
        clear explanations while maintaining an engaging tone.

        Guidelines:
        1. Start with a warm welcome and introduction
        2. Break down complex topics into digestible segments
        3. Use follow-up questions to clarify points
        4. End with a summary and thank you
        5. Aim for a total length of approximately {transcript_length} words
        6. Ensure responses are concise but informative

        Content to transform:
        {content}
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at creating engaging interview transcripts. "
                        f"Create a transcript that is approximately {transcript_length} words long."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,  # Increased to accommodate longer transcripts
            temperature=0.7,
        )

        return str(response.choices[0].message.content)

    def _split_transcript(self, transcript: str) -> list[tuple[str, str]]:
        """
        Split the transcript into pairs of host and expert dialogue.

        Args:
            transcript (str): The formatted transcript with Person1/Person2 tags

        Returns
        -------
            List[Tuple[str, str]]: List of (host_line, expert_line) pairs
        """
        # Regular expression pattern to match Person1 and Person2 dialogues
        pattern = r"<Person1>(.*?)</Person1>\s*<Person2>(.*?)</Person2>"

        # Find all matches in the transcript
        matches = re.findall(pattern, transcript, re.DOTALL)

        # Clean up the matches
        cleaned_pairs = [
            (
                " ".join(host.strip().split()),  # Clean host line
                " ".join(expert.strip().split()),  # Clean expert line
            )
            for host, expert in matches
        ]

        return cleaned_pairs

    def _generate_audio_segments(
        self, transcript_pairs: list[tuple[str, str]], temp_dir: str
    ) -> list[str]:
        """
        Generate audio segments for each line in the conversation.

        Args:
            transcript_pairs: List of (host_line, expert_line) pairs
            temp_dir: Directory to store temporary audio files

        Returns
        -------
            List[str]: List of paths to generated audio files
        """
        audio_files = []

        for idx, (host_line, expert_line) in enumerate(transcript_pairs, 1):
            # Generate host audio with one voice
            host_file = Path(temp_dir) / f"{idx:03d}_host.mp3"
            self._generate_audio(
                text=host_line,
                output_path=str(host_file),
                voice="alloy",  # Use a distinct voice for the host
            )
            audio_files.append(str(host_file))

            # Generate expert audio with different voice
            expert_file = Path(temp_dir) / f"{idx:03d}_expert.mp3"
            self._generate_audio(
                text=expert_line,
                output_path=str(expert_file),
                voice="echo",  # Use a different voice for the expert
            )
            audio_files.append(str(expert_file))

        return audio_files

    def _merge_audio_files(
        self, audio_files: list[str], output_path: str
    ) -> None:
        """
        Merge multiple audio files into a single podcast.

        Args:
            audio_files: List of paths to audio segments
            output_path: Path for the final merged audio file
        """
        combined = AudioSegment.empty()

        for audio_file in sorted(
            audio_files
        ):  # Sort to maintain conversation order
            segment = AudioSegment.from_mp3(audio_file)
            # Add a small pause between segments
            combined += segment + AudioSegment.silent(duration=500)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Export the final audio
        combined.export(output_path, format="mp3")

    def transform(
        self,
        sources: list[str | ContentItem],
        output_type: OutputType,
        output_file: str | None = None,
        transcript_length: int = 1000,
    ) -> str:
        """
        Transform content into desired output format.

        Args:
            sources: List of content sources (URLs, file paths, or raw text)
            output_type: Desired output format (PODCAST or BOOK)
            output_file: Optional output file path
            transcript_length: Target length of the transcript in words (for podcasts only)

        Returns
        -------
            Path to the generated output file(s)
        """
        # Process all content items
        processed_content = []
        for source in sources:
            if isinstance(source, str):
                # Detect content type
                if source.startswith(("http://", "https://")):
                    if "youtube.com" in source or "youtu.be" in source:
                        content_type = ContentType.YOUTUBE
                    else:
                        content_type = ContentType.URL
                elif source.lower().endswith(".pdf"):
                    content_type = ContentType.PDF
                else:
                    content_type = ContentType.TEXT
                item = ContentItem(content_type, source)
            else:
                item = source

            # Extract content
            extracted_text = self._process_content(item)
            processed_content.append(extracted_text)

        # Combine content
        combined_text = "\n\n".join(processed_content)

        # Generate appropriate format based on output type
        if output_type == OutputType.PODCAST:
            # Generate interview-style transcript with specified length
            transcript = self._generate_interview_transcript(
                combined_text, transcript_length
            )

            # Split transcript into conversation pairs
            transcript_pairs = self._split_transcript(transcript)

            # Create temporary directory for audio segments
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate audio segments
                audio_files = self._generate_audio_segments(
                    transcript_pairs, temp_dir
                )

                # Generate output paths
                if output_file is None:
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    base_name = f"output_{os.urandom(4).hex()}"
                    audio_path = output_dir / f"{base_name}.mp3"
                    transcript_path = output_dir / f"{base_name}.txt"
                else:
                    audio_path = Path(output_file)
                    transcript_path = audio_path.with_suffix(".txt")

                # Save transcript
                transcript_path.write_text(transcript)

                # Merge audio files
                self._merge_audio_files(audio_files, str(audio_path))

                return f"Audio: {audio_path}\nTranscript: {transcript_path}"
        else:
            # For books, generate a summary
            final_content = self._generate_summary(combined_text)

            # Generate output path
            if output_file is None:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"output_{os.urandom(4).hex()}.txt"
            else:
                output_path = Path(output_file)

            # Save content
            output_path.write_text(final_content)
            return str(output_path)

    def generate_from_query(
        self,
        query: str,
        output_type: OutputType,
        output_file: str | None = None,
        transcript_length: int = 1000,
    ) -> str:
        """
        Generate content from a text query describing a topic

        Args:
            query: Text description of the topic
            output_type: Desired output format (PODCAST or BOOK)
            output_file: Optional output file path
            transcript_length: Target length of the transcript in words (for podcasts only)

        Returns
        -------
            Path to the generated output file
        """
        content_item = ContentItem(ContentType.QUERY, query)
        return self.transform(
            [content_item],
            output_type,
            output_file,
            transcript_length=transcript_length,
        )


# Example usage
def main():
    # Initialize with your OpenAI API key
    transformer = ContentTransformer(openai_api_key="your-api-key")

    # Generate a short podcast (500 words)
    short_podcast = transformer.generate_from_query(
        query="Explain quantum computing basics",
        output_type=OutputType.PODCAST,
        transcript_length=500,
    )
    print(f"Generated short podcast: {short_podcast}")

    # Generate a longer podcast (2000 words)
    long_podcast = transformer.generate_from_query(
        query="Explain quantum computing and its applications in detail",
        output_type=OutputType.PODCAST,
        transcript_length=2000,
    )
    print(f"Generated long podcast: {long_podcast}")


if __name__ == "__main__":
    main()
