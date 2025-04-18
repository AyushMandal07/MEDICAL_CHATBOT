# If you're not using pipenv, make sure to load the .env manually
from dotenv import load_dotenv
load_dotenv()

import os
import subprocess
import platform
from gtts import gTTS # type: ignore
from elevenlabs.client import ElevenLabs # type: ignore
import elevenlabs # type: ignore


ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise ValueError("❌ ELEVENLABS_API_KEY is missing. Please set it in your .env file.")


def text_to_speech_with_gtts_old(input_text, output_filepath):
    """
    Convert text to speech using gTTS and save to a file.
    """
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(output_filepath)


def text_to_speech_with_elevenlabs_old(input_text, output_filepath):
    """
    Convert text to speech using ElevenLabs API (old style) and save to a file.
    """
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2" 
    )
    elevenlabs.save(audio, output_filepath)


def text_to_speech_with_gtts(input_text, output_filepath):
    """
    Convert text to speech using gTTS and automatically play the audio.
    """
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(output_filepath)

    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
        elif os_name == "Linux":
            subprocess.run(['aplay', output_filepath])
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"❌ Error playing audio: {e}")


def text_to_speech_with_elevenlabs(input_text, output_filepath):
    """
    Convert text to speech using ElevenLabs API and automatically play the audio.
    """
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)

    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
        elif os_name == "Linux":
            subprocess.run(['aplay', output_filepath])
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"❌ Error playing audio: {e}")


# Example usage (uncomment to test)
# input_text = "Hi this is AI with Hassan!"
# text_to_speech_with_gtts_old(input_text=input_text, output_filepath="gtts_testing.mp3")
# text_to_speech_with_gtts(input_text=input_text, output_filepath="gtts_autoplay.mp3")
# text_to_speech_with_elevenlabs_old(input_text=input_text, output_filepath="eleven_old.mp3")
# text_to_speech_with_elevenlabs(input_text=input_text, output_filepath="eleven_autoplay.mp3")
