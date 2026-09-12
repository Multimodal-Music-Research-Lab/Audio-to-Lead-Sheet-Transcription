FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies:
#   ffmpeg / libsm6      - audio/video decoding for datasets & librosa
#   fluidsynth           - MIDI synthesis (pretty_midi / music21)
#   rubberband-cli       - required by pyrubberband (data augmentation)
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsm6 \
        fluidsynth \
        rubberband-cli \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project source
COPY code/ ./code/
