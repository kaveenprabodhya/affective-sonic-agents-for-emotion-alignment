"""Build and freeze both estimators.

    # both:
    python src/estimators/build.py --deam datasets/DEAM --pmemo datasets/PMEmo
    # just one:
    python src/estimators/build.py --deam datasets/DEAM --only A
    python src/estimators/build.py --pmemo datasets/PMEmo --only B --family-b svr

Estimator A: DEAM, random forest, optimisation coach.
Estimator B: PMEmo, SVR (or --family-b mlp), held out and frozen for the H1 judge.
Both use the combined A+B feature set and output valence/arousal on [-1, 1].
Freeze B before any experimental generation.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimators.data import build_deam, build_pmemo, feature_columns   # noqa: E402
from estimators.model import train_and_freeze                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deam", help="Path to DEAM root")
    ap.add_argument("--pmemo", help="Path to PMEmo root")
    ap.add_argument("--only", choices=["A", "B"], help="Build just one estimator")
    ap.add_argument("--songs", type=int, default=None, help="Cap songs per corpus (for a quick build)")
    ap.add_argument("--family-b", default="svr", choices=["svr", "mlp"])
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    use_cache = not args.no_cache

    if args.only != "B":
        if not args.deam:
            sys.exit("Estimator A needs --deam")
        print("Estimator A (DEAM, random forest, optimisation coach)")
        df = build_deam(args.deam, n_songs=args.songs, use_cache=use_cache)
        train_and_freeze(df, feature_columns(df), name="estimator_A", corpus="DEAM",
                         family="rf", role="optimisation_coach", held_out=False)

    if args.only != "A":
        if not args.pmemo:
            sys.exit("Estimator B needs --pmemo")
        print(f"\nEstimator B (PMEmo, {args.family_b}, held-out H1 judge)")
        df = build_pmemo(args.pmemo, n_songs=args.songs, use_cache=use_cache)
        train_and_freeze(df, feature_columns(df), name="estimator_B", corpus="PMEmo",
                         family=args.family_b, role="held_out_H1_judge", held_out=True)

    print("\nDone. Frozen artefacts + metadata in models/. "
          "Estimator B must not be retrained or used inside the optimisation loop.")


if __name__ == "__main__":
    main()