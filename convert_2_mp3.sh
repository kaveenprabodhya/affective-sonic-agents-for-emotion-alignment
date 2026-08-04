mkdir -p data/stimuli_mp3

for file in data/stimuli/*.wav; do
    filename=$(basename "$file" .wav)

    ffmpeg \
        -hide_banner \
        -loglevel error \
        -y \
        -i "$file" \
        -codec:a libmp3lame \
        -b:a 192k \
        "data/stimuli_mp3/${filename}.mp3"

    echo "Converted: ${filename}.mp3"
done