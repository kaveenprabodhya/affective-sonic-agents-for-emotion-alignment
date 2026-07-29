DEAM=~/Desktop/Projects/affective-sonic-agents-for-emotion-alignment/datasets/DEAM/

python - "$DEAM" << 'EOF'
import sys, glob, os
root = sys.argv[1]
print("ROOT:", root)
dirs = sorted({os.path.relpath(d, root) for d, _, _ in os.walk(root)})
print("\n-- DIRS --"); [print("  ", d) for d in dirs[:40]]
aud = (glob.glob(os.path.join(root, "**", "*.mp3"), recursive=True) +
       glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
print(f"\n-- AUDIO -- {len(aud)} files; examples:")
[print("  ", os.path.relpath(a, root)) for a in aud[:3]]
csvs = [c for c in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
        if any(k in c.lower() for k in ("annot", "valence", "arousal", "static", "dynamic", "song"))]
print(f"\n-- CSVs of interest -- {len(csvs)}")
for c in csvs[:12]:
    with open(c) as f:
        head = f.readline().strip()
    print(f"  {os.path.relpath(c, root)}\n     header[:180]: {head[:180]}")
EOF