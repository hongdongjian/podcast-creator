import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from loguru import logger
from moviepy import AudioFileClip, concatenate_audioclips
from pydantic import BaseModel, Field, field_validator

# Compile regex pattern once for better performance
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_thinking_content(content: str) -> Tuple[str, str]:
    """
    Parse message content to extract thinking content from <think> tags.
    Handles complete pairs, standalone opening tags, and standalone closing tags.

    Args:
        content (str): The original message content

    Returns:
        Tuple[str, str]: (thinking_content, cleaned_content)
            - thinking_content: Content from within <think> tags
            - cleaned_content: Original content with <think> blocks removed

    Example:
        >>> content = "<think>Let me analyze this</think>Here's my answer"
        >>> thinking, cleaned = parse_thinking_content(content)
        >>> print(thinking)
        "Let me analyze this"
        >>> print(cleaned)
        "Here's my answer"
    """
    # Input validation
    if not isinstance(content, str):
        return "", str(content) if content is not None else ""

    # Limit processing for very large content (100KB limit)
    if len(content) > 100000:
        return "", content

    # Start with original content
    cleaned_content = content
    thinking_parts = []

    # First, remove complete pairs and extract their content
    thinking_matches = THINK_PATTERN.findall(content)
    if thinking_matches:
        thinking_parts.extend(match.strip() for match in thinking_matches)
        cleaned_content = THINK_PATTERN.sub("", cleaned_content)

    # Handle standalone closing tag (e.g., "</think>\n{...}")
    # This often happens when LLM puts everything in thinking but only closes it
    # Remove standalone closing tag at the beginning of content
    cleaned_content = re.sub(r"^\s*</think>\s*", "", cleaned_content, flags=re.MULTILINE)
    # Also remove any standalone closing tags elsewhere
    cleaned_content = re.sub(r"</think>\s*", "", cleaned_content, flags=re.MULTILINE)

    # Handle standalone opening tag (less common but possible)
    # Remove standalone opening tag if there's no matching closing tag
    if "<think>" in cleaned_content and "</think>" not in cleaned_content:
        cleaned_content = re.sub(r"<think>\s*", "", cleaned_content, flags=re.MULTILINE)

    # Join all thinking content with double newlines
    thinking_content = "\n\n".join(thinking_parts) if thinking_parts else ""

    # Clean up extra whitespace
    cleaned_content = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned_content).strip()

    return thinking_content, cleaned_content


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from text that may contain explanatory text before/after the JSON.
    
    Handles:
    1. JSON in markdown code blocks: ```json {...} ```
    2. Plain JSON object: {...}
    3. Text before/after JSON
    
    Args:
        text (str): Text that may contain JSON
        
    Returns:
        str: Extracted JSON string, or original text if no JSON found
        
    Example:
        >>> text = "Here's the result:\n```json\n{\"key\": \"value\"}\n```"
        >>> extract_json_from_text(text)
        '{"key": "value"}'
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Try to extract JSON from markdown code blocks first
    # Match ```json or ``` followed by JSON object
    json_code_block_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    match = json_code_block_pattern.search(text)
    if match:
        return match.group(1).strip()
    
    # Try to find JSON object by finding first { and matching closing }
    # This handles cases where JSON is not in code blocks
    brace_start = text.find("{")
    if brace_start == -1:
        # No JSON found, return original text
        return text
    
    # Find matching closing brace by counting braces
    # This properly handles nested objects and arrays
    brace_count = 0
    in_string = False
    escape_next = False
    brace_end = -1
    
    for i in range(brace_start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == "\\":
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                brace_end = i
                break
    
    if brace_end != -1:
        json_str = text[brace_start:brace_end + 1]
        return json_str.strip()
    
    # If no valid JSON found, return original text
    return text


def clean_thinking_content(content: str) -> str:
    """
    Remove thinking content from AI responses and extract JSON if present.
    
    This function:
    1. Removes <think> tags
    2. Extracts JSON from the cleaned content (handles text before/after JSON)
    
    Args:
        content (str): The original message content with potential <think> tags

    Returns:
        str: Cleaned content with thinking tags removed and JSON extracted

    Example:
        >>> content = "<think>Let me think...</think>Here's the result: {\"key\": \"value\"}"
        >>> clean_thinking_content(content)
        '{"key": "value"}'
    """
    _, cleaned_content = parse_thinking_content(content)
    # Extract JSON if present (handles cases where LLM adds explanatory text)
    json_content = extract_json_from_text(cleaned_content)
    return json_content


class Segment(BaseModel):
    name: str = Field(..., description="Name of the segment")
    description: str = Field(..., description="Description of the segment")
    size: Literal["short", "medium", "long"] = Field(
        ..., description="Size of the segment"
    )


class Outline(BaseModel):
    segments: list[Segment] = Field(..., description="List of segments")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        return {"segments": [segment.model_dump(**kwargs) for segment in self.segments]}


class Dialogue(BaseModel):
    speaker: str = Field(..., description="Speaker name")
    dialogue: str = Field(..., description="Dialogue")

    @field_validator("speaker")
    @classmethod
    def validate_speaker_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Speaker name cannot be empty")
        return v.strip()


class Transcript(BaseModel):
    transcript: list[Dialogue] = Field(..., description="Transcript")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        # Custom serialization: convert list of Dialogue models to list of dicts
        return {
            "transcript": [
                dialogue.model_dump(**kwargs) for dialogue in self.transcript
            ]
        }


def create_validated_transcript_parser(valid_speaker_names: List[str]):
    """
    Create a transcript parser that validates speaker names against a list of valid names

    Args:
        valid_speaker_names: List of valid speaker names

    Returns:
        PydanticOutputParser: Parser with speaker validation
    """

    class ValidatedDialogue(BaseModel):
        speaker: str = Field(..., description="Speaker name")
        dialogue: str = Field(..., description="Dialogue")

        @field_validator("speaker")
        @classmethod
        def validate_speaker_name(cls, v):
            if not v or len(v.strip()) == 0:
                raise ValueError("Speaker name cannot be empty")

            cleaned_name = v.strip()
            if cleaned_name not in valid_speaker_names:
                raise ValueError(
                    f"Invalid speaker name '{cleaned_name}'. Must be one of: {', '.join(valid_speaker_names)}"
                )

            return cleaned_name

    class ValidatedTranscript(BaseModel):
        transcript: list[ValidatedDialogue] = Field(..., description="Transcript")

        def model_dump(self, **kwargs) -> Dict[str, Any]:
            return {
                "transcript": [
                    dialogue.model_dump(**kwargs) for dialogue in self.transcript
                ]
            }

    return PydanticOutputParser(pydantic_object=ValidatedTranscript)


outline_parser = PydanticOutputParser(pydantic_object=Outline)
transcript_parser = PydanticOutputParser(pydantic_object=Transcript)


def get_outline_prompter():
    """Get outline prompter with configuration support."""
    from .config import ConfigurationManager

    config_manager = ConfigurationManager()
    return config_manager.get_template_prompter("outline", parser=outline_parser)


def get_transcript_prompter():
    """Get transcript prompter with configuration support."""
    from .config import ConfigurationManager

    config_manager = ConfigurationManager()
    return config_manager.get_template_prompter("transcript", parser=transcript_parser)


# Legacy exports for backward compatibility
outline_prompt = get_outline_prompter()
transcript_prompt = get_transcript_prompter()

# Legacy functions removed - use create_podcast from graph.py instead


async def combine_audio_files(
    audio_dir: Union[Path, str], final_filename: str, final_output_dir: Union[Path, str]
):
    """
    Combines multiple audio files into a single MP3 file using moviepy.
    Expects 'audio_segments_data' in inputs: a list of strings, where each string is a path to an audio file.
    Also expects 'final_filename' in inputs: a string for the desired output filename (e.g., "podcast_episode.mp3").
    Example input: {
        "audio_segments_data": ["path/to/audio1.mp3", "path/to/audio2.mp3"],
        "final_filename": "my_podcast.mp3"
    }
    Output: {"combined_audio_path": "output/audio/my_podcast.mp3"}
    """
    logger.info("[Core Function] combine_audio_files called.")
    if isinstance(audio_dir, str):
        audio_dir = Path(audio_dir)
    if isinstance(final_output_dir, str):
        final_output_dir = Path(final_output_dir)
    list_of_audio_paths = sorted(audio_dir.glob("*.mp3"))
    output_filename_from_input = final_filename

    logger.debug(list_of_audio_paths)

    if not list_of_audio_paths:
        logger.warning(
            "combine_audio_files: No audio segment data (list of paths) provided."
        )
        return {"combined_audio_path": "ERROR: No audio segment data"}

    if not isinstance(list_of_audio_paths, list):
        logger.error(
            f"combine_audio_files: 'audio_segments_data' is not a list. Received: {type(list_of_audio_paths)}"
        )
        return {
            "combined_audio_path": "ERROR: audio_segments_data must be a list of file paths"
        }

    clips = []
    valid_clips = []
    for i, file_path in enumerate(list_of_audio_paths):
        if not isinstance(file_path, Path):
            logger.warning(
                f"combine_audio_files: Item {i} in audio_segments_data is not a string path: {file_path}. Skipping."
            )
            continue

        try:
            if file_path.exists() and file_path.is_file():
                clips.append(AudioFileClip(str(file_path)))
                valid_clips.append(clips[-1])  # Keep track of valid clips for later
            else:
                logger.error(
                    f"combine_audio_files: File not found or not a file: {file_path}"
                )
        except Exception as e:
            logger.error(
                f"combine_audio_files: Error loading audio clip {file_path}: {e}"
            )

    if not clips:
        logger.error("combine_audio_files: No valid audio clips could be loaded.")
        return {"combined_audio_path": "ERROR: No valid clips"}

    try:
        # Ensure all clips are closed after concatenation, even if it fails during the process.
        # MoviePy's concatenate_audioclips might not close source clips if it errors out mid-way.
        final_clip = concatenate_audioclips(clips)
    except Exception as e:
        logger.error(f"Error during concatenate_audioclips: {e}")
        for clip_obj in clips:
            try:
                clip_obj.close()
            except Exception as close_exc:
                logger.debug(f"Error closing clip during error handling: {close_exc}")
        return {"combined_audio_path": f"ERROR: Concatenation failed - {e}"}

    output_dir = final_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the filename from input if provided, otherwise generate one.
    if output_filename_from_input and isinstance(output_filename_from_input, str):
        # Basic sanitization for filename (optional, depending on how robust it needs to be)
        # For now, assume it's a simple filename like 'episode.mp3'
        output_filename = Path(
            output_filename_from_input
        ).name  # Use only the filename part
        if not output_filename.endswith(".mp3"):
            output_filename += ".mp3"  # Ensure .mp3 extension
    else:
        output_filename = f"combined_{uuid.uuid4().hex}.mp3"
        logger.warning(
            f"'final_filename' not provided or invalid in inputs. Using generated name: {output_filename}"
        )

    output_path = output_dir / output_filename

    try:
        final_clip.write_audiofile(str(output_path), codec="mp3")
        logger.info(f"Successfully combined audio to: {output_path.resolve()}")
        return {
            "combined_audio_path": str(output_path.resolve()),
            "original_segments_count": len(valid_clips),
            "total_duration_seconds": final_clip.duration,
        }
    except Exception as e:
        logger.error(f"Error writing final audio file {output_path}: {e}")
        return {"combined_audio_path": f"ERROR: Failed to write output audio - {e}"}
    finally:
        final_clip.close()  # Close the final concatenated clip
        for clip_obj in clips:  # Ensure all source clips are closed
            try:
                clip_obj.close()
            except Exception as close_exc:
                logger.debug(f"Error closing source clip: {close_exc}")
