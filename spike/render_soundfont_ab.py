import sys
import json
import shutil
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from config_loader import load, ROOT
from generator.synth import render


MANIFEST = ROOT / "data/stimuli/manifest.json"

OLD_DIR = ROOT / "spike" / "outputs" / "render_ab" / "original"
NEW_DIR = ROOT / "spike" / "outputs" / "render_ab" / "fluidr3"

SOUNDFONT = Path(
    "/usr/share/sounds/sf2/FluidR3_GM.sf2"
)

BRIEFS = {
    "B01",
    "B05",
    "B09",
    "B13",
}


def main():

    if not SOUNDFONT.exists():
        raise SystemExit(
            f"SoundFont not found: {SOUNDFONT}"
        )

    exp = load("experiment.yaml")

    duration = float(
        exp["synthesis"]["duration_s"]
    )

    sr = int(
        exp["synthesis"]["sample_rate_hz"]
    )

    manifest = json.loads(
        MANIFEST.read_text()
    )

    OLD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    NEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    for row in manifest:

        brief = row["brief"]

        if brief not in BRIEFS:
            continue

        for key in (
            "non_optimised",
            "optimised",
        ):

            item = row[key]
            filename = item["file"]

            # Only generation run 0.
            if "_run0_" not in filename:
                continue

            original = (
                ROOT
                / "data/stimuli"
                / filename
            )

            old_copy = (
                OLD_DIR
                / filename
            )

            new_file = (
                NEW_DIR
                / filename
            )

            shutil.copy2(
                original,
                old_copy,
            )

            print(
                f"rendering {filename}"
            )

            render(
                item["params"],
                str(SOUNDFONT),
                str(new_file),
                duration,
                sr,
            )

            count += 1

    print()
    print(
        f"Created {count} A/B pairs."
    )

    print(
        f"Original: {OLD_DIR}"
    )

    print(
        f"FluidR3:  {NEW_DIR}"
    )


if __name__ == "__main__":
    main()