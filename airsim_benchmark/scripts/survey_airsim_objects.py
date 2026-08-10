#!/usr/bin/env python3
"""Dump AirSim scene object names and poses for landmark catalog correction.

Compare dumped poses to airsim_benchmark/config/landmarks.yaml seeds
(red_car, rooftop_near_red_car, etc.) and edit the YAML if they are wrong
BEFORE a full collection.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airsim_benchmark.core.airsim_client import AirSimClient


def main():
    parser = argparse.ArgumentParser(
        description="Dump AirSim scene object names and poses for landmark catalog correction."
    )
    parser.add_argument("--out", default="data/airsim_objects.txt")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = AirSimClient()
    try:
        client.connect()
    except Exception as e:
        print(
            "Cannot reach AirSim RPC (connection refused).\n"
            "This script must run on a compute node where AirSim Neighborhood "
            "is already running — not on a login node.\n"
            "Submit:  sbatch scripts/run_survey_landmarks.slurm\n"
            f"Error: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        try:
            names = client.client.simListSceneObjects()
        except Exception as e:
            print(
                "Failed to list scene objects via simListSceneObjects(). "
                "Confirm AirSim is running with the Neighborhood environment "
                "and that the RPC API is reachable.\n"
                f"Error: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(out_path, "w") as f:
            for name in names:
                try:
                    pose = client.client.simGetObjectPose(name)
                    p = pose.position
                    f.write(f"{name}\t{p.x_val:.2f}\t{p.y_val:.2f}\t{p.z_val:.2f}\n")
                except Exception as e:
                    f.write(f"{name}\tERR\t{e}\n")
    finally:
        client.disconnect()

    print(f"Wrote {out_path} ({len(names)} objects)")


if __name__ == "__main__":
    main()
