"""YAML pipeline parser."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepDef:
    """Definition of a single pipeline step."""

    id: str
    class_path: str
    depends_on: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class PipelineDef:
    """Parsed pipeline definition with steps and dependencies."""

    name: str
    steps: List[StepDef] = field(default_factory=list)

    def get_step(self, step_id: str) -> Optional[StepDef]:
        """Get step by ID."""
        for s in self.steps:
            if s.id == step_id:
                return s
        return None


def parse_pipeline(raw: Dict[str, Any]) -> PipelineDef:
    """
    Parse raw YAML dict into PipelineDef.

    Expected structure:
        pipeline:
          name: churn_pipeline
        steps:
          - id: preprocess
            class: steps.preprocess.PartFailurePreprocess
          - id: train
            class: steps.train.PartFailureTrain
            depends_on: [preprocess]
    """
    # Handle nested pipeline key
    top = raw.get("pipeline", raw)
    name = top.get("name", "unnamed_pipeline")
    steps_raw = top.get("steps", raw.get("steps", []))

    steps: List[StepDef] = []
    for s in steps_raw:
        step_id = s.get("id")
        class_path = s.get("class")
        if not step_id or not class_path:
            continue
        depends_on = s.get("depends_on", [])
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        inputs = s.get("inputs", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        outputs = s.get("outputs", [])
        if isinstance(outputs, str):
            outputs = [outputs]
        steps.append(
            StepDef(
                id=step_id,
                class_path=class_path,
                depends_on=depends_on,
                inputs=list(inputs),
                outputs=list(outputs),
            )
        )

    return PipelineDef(name=name, steps=steps)
