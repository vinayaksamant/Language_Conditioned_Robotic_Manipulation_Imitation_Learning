from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from robot_manipulation_pi0.sim import VLA_SCENE_OBJECTS, VLA_SCENE_TARGETS
from robot_manipulation_pi0.vla.small_vla import SmallVLAPolicy
from robot_manipulation_pi0.vla.tasks import TASK_TEMPLATES, task_for_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the learned language task head without running MuJoCo."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/small_vla_strict_v7.pt"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = SmallVLAPolicy.load(args.checkpoint, device=args.device)
    if not policy.learned_task_routing:
        raise ValueError(
            f"Checkpoint {args.checkpoint} uses legacy rule-based task routing, not a learned head."
        )

    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    confidence_sum: Counter[str] = Counter()
    for scene_object in VLA_SCENE_OBJECTS:
        variant_count = len(TASK_TEMPLATES[scene_object.key])
        for scene_target in VLA_SCENE_TARGETS:
            expected = (scene_object.key, scene_target.key)
            label = f"{scene_object.key}->{scene_target.key}"
            for variant in range(variant_count):
                task = task_for_object(
                    scene_object.key,
                    variant=variant,
                    target_key=scene_target.key,
                )
                policy.reset()
                prediction = policy.ground_instruction(task.instruction)
                is_correct = (prediction.object_key, prediction.target_key) == expected
                counts[label] += 1
                correct[label] += int(is_correct)
                confidence_sum[label] += prediction.confidence
                status = "correct" if is_correct else "wrong"
                print(
                    f"[{status:7s}] expected={label:34s} "
                    f"predicted={prediction.object_key}->{prediction.target_key:12s} "
                    f"confidence={prediction.confidence:.3f} | {task.instruction}"
                )

    total = sum(counts.values())
    total_correct = sum(correct.values())
    print(f"Overall grounding: {total_correct}/{total} ({100.0 * total_correct / total:.1f}%)")
    for label in sorted(counts):
        average_confidence = confidence_sum[label] / counts[label]
        print(
            f"{label}: {correct[label]}/{counts[label]} "
            f"({100.0 * correct[label] / counts[label]:.1f}%), "
            f"mean confidence={average_confidence:.3f}"
        )


if __name__ == "__main__":
    main()
