import sys
import json
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parents[1]
        / "src"
    ),
)

from config_loader import ROOT
from generator.synth import (
    RHYTHM_PATTERNS,
    render,
)


MANIFEST = (
    ROOT
    / "data"
    / "stimuli"
    / "manifest.json"
)

OUT = (
    ROOT
    / "spike"
    / "outputs"
    / "rhythm_gallery_generaluser"
)

SOUNDFONT = str(
    ROOT
    / "assets"
    / "soundfonts"
    / "GeneralUser-GS.sf2"
)

EXAMPLES = {
    "B01",
    "B13",
}


def main():

    manifest = json.loads(
        MANIFEST.read_text()
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = []

    for row in manifest:

        if row["brief"] not in EXAMPLES:
            continue

        item = row["optimised"]

        if "_run0_" not in item["file"]:
            continue

        selected.append(
            (
                row["brief"],
                item,
            )
        )

    for brief, item in selected:

        for rhythm in RHYTHM_PATTERNS:

            p = dict(
                item["params"]
            )

            p[
                "rhythm_pattern"
            ] = rhythm

            filename = (
                f"{brief}_"
                f"{rhythm}.wav"
            )

            print(
                f"rendering "
                f"{filename}"
            )

            render(
                p,
                SOUNDFONT,
                str(
                    OUT
                    / filename
                ),
                duration=3.0,
                sr=22050,
            )

    print()
    print(
        f"saved to {OUT}"
    )


if __name__ == "__main__":
    main()