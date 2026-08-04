from pathlib import Path
from html import escape

folder = Path(__file__).resolve().parent
files = sorted(folder.glob("*.mp3"))

rows = []

for number, file in enumerate(files, start=1):
    name = file.stem

    rows.append(
        f"""
        <tr>
            <td>{number}</td>
            <td>{escape(name)}</td>
            <td>
                <audio controls preload="none">
                    <source src="{escape(file.name)}" type="audio/mpeg">
                    Your browser does not support audio playback.
                </audio>
            </td>
        </tr>
        """
    )

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sonic Logo Stimuli</title>
    <style>
        body {{
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 20px;
            font-family: Arial, sans-serif;
            line-height: 1.5;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }}

        th {{
            background: #f2f2f2;
            position: sticky;
            top: 0;
        }}

        audio {{
            width: 320px;
        }}
    </style>
</head>
<body>
    <h1>Sonic Logo Stimuli</h1>

    <p>
        This page contains the {len(files)} sonic logos used in the study,
        including optimised and non-optimised versions.
    </p>

    <table>
        <thead>
            <tr>
                <th>No.</th>
                <th>Stimulus</th>
                <th>Audio</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</body>
</html>
"""

output = folder / "index.html"
output.write_text(html, encoding="utf-8")

print(f"Created {output} with {len(files)} audio files.")