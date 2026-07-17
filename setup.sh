#!/bin/bash
# Setup script for AUP Learning Cloud GPU notebook
# Run this after cloning the repo on the AMD GPU notebook

set -e

echo "============================================"
echo "  Devnagri AI — Indic Voice AI Booth Setup"
echo "  AMD AUP Learning Cloud GPU Environment"
echo "============================================"

# 1. Install espeak-ng (required for Kokoro TTS G2P)
echo ""
echo "[1/4] Installing espeak-ng..."
if command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq espeak-ng libsndfile1
elif command -v conda &>/dev/null; then
    conda install -y -c conda-forge espeak-ng
elif command -v apt-get &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq espeak-ng libsndfile1
else
    echo "  WARNING: Cannot install espeak-ng. TTS will fall back to English only."
fi

# 2. Install Python dependencies
echo ""
echo "[2/4] Installing Python dependencies..."
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install -q gradio transformers accelerate sentencepiece librosa soundfile numpy

# 3. Install IndicVoice + Indic G2P (Kokoro TTS)
echo ""
echo "[3/4] Installing IndicVoice (Kokoro-82M + Indic G2P)..."
pip install -q git+https://github.com/Bindkushal/indic-g2p.git
pip install -q git+https://github.com/Bindkushal/indic-voice.git

# 4. Verify GPU
echo ""
echo "[4/4] Checking AMD ROCm GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ROCm/HIP detected: YES')
else:
    print('  WARNING: No GPU detected. Check ROCm installation.')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Launch the demo with:  python3 app.py"
echo "============================================"
