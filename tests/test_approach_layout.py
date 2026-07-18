from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROACHES = (
    "absolute_probability", "hybrid_event", "trade_tape_hybrid",
    "optimal_exit", "empirical_reversion", "empirical_runner",
    "state_reversion", "residual_path", "mispricing",
)
REQUIRED = (
    "README.md", "Dockerfile", "architecture.py", "train.py",
    "build_data.py", "evaluate.py", "paper_trader.py", "study.json",
    "study/README.md",
)


class ApproachLayoutTests(unittest.TestCase):
    def test_every_approach_has_complete_runnable_layout(self):
        for name in APPROACHES:
            folder = PROJECT_ROOT / "approaches" / name
            with self.subTest(approach=name):
                self.assertTrue(folder.is_dir())
                for relative in REQUIRED:
                    self.assertTrue((folder / relative).is_file(), relative)

    def test_study_manifests_resolve_when_conducted(self):
        for name in APPROACHES:
            manifest = json.loads(
                (PROJECT_ROOT / "approaches" / name / "study.json").read_text()
            )
            with self.subTest(approach=name):
                self.assertIn("conducted", manifest)
                if manifest["conducted"]:
                    self.assertTrue((PROJECT_ROOT / manifest["path"]).is_dir())

    def test_dockerfiles_run_the_local_paper_entrypoint(self):
        for name in APPROACHES:
            text = (
                PROJECT_ROOT / "approaches" / name / "Dockerfile"
            ).read_text()
            with self.subTest(approach=name):
                self.assertIn(f"approaches.{name}.paper_trader", text)


if __name__ == "__main__":
    unittest.main()
