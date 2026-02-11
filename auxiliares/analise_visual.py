import base64
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()


def extract_frame(video_path: str, output_path: str, timestamp: float = 2.0) -> None:
    """Extract a single frame from video at given timestamp."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ss", str(timestamp),
        "-vframes", "1",
        "-q:v", "2",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def analyze_video_frame(
    video_path: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    max_tokens: int = 500
) -> str:
    """
    Analyze video frame using GPT-4 Vision.
    
    Args:
        video_path: Path to video file
        api_key: OpenAI API key (defaults to env var)
        model: Vision model to use
        max_tokens: Max tokens in response
        
    Returns:
        Description of video content
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    client = OpenAI(api_key=api_key.strip())
    
    # Extract frame at 2 seconds
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        frame_path = tmp.name
    
    try:
        extract_frame(video_path, frame_path, timestamp=2.0)
        
        # Encode image to base64
        with open(frame_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Call GPT-4 Vision
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Descreva este frame de vídeo do Instagram em português, focando em:
1. Qual é o tema principal?
2. O que está acontecendo visualmente?
3. Qual emoção ou mensagem transmite?
4. Que tipo de conteúdo é (educacional, entretenimento, fitness, tecnologia, etc.)?

Seja conciso mas específico. Máximo 3 frases."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
        
    finally:
        # Cleanup
        if Path(frame_path).exists():
            Path(frame_path).unlink()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m virality.vision_analysis <video_path>")
        sys.exit(1)
    
    video = sys.argv[1]
    description = analyze_video_frame(video)
    print(description)
