"""Run a real-window tensor and gradient smoke test for the SE-GSCL specialist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.continual_fdllm.domain_windows import build_domain_window_dataset  # noqa: E402
from se_gscl.losses import (  # noqa: E402
    cross_condition_supervised_contrastive_loss,
    cross_covariance_loss,
    global_prototype_alignment_loss,
)
from se_gscl.models import SEGSCLSpecialist  # noqa: E402
from se_gscl.semantics import FrozenPrototypeBank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("cwru4", "cwru10", "cwru19", "paderborn", "hustbearing"),
        default="cwru4",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--domains", default="0,1")
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--step-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domains = [int(value.strip()) for value in args.domains.split(",") if value.strip()]
    dataset = build_domain_window_dataset(
        args.data_root,
        dataset=args.dataset,
        domains=domains,
        window_size=args.window_size,
        step_size=args.step_size,
        max_windows_per_file=max(2, args.batch_size),
        normalize=False,
    )
    all_x = np.asarray(dataset["x"], dtype=np.float32)
    all_y = np.asarray(dataset["y"], dtype=np.int64)
    all_d = np.asarray(dataset["domain_id"], dtype=np.int64)
    per_domain = max(1, args.batch_size // max(1, len(domains)))
    selected: list[int] = []
    for domain in domains:
        selected.extend(np.flatnonzero(all_d == domain)[:per_domain].tolist())
    selected = selected[: args.batch_size]
    x = all_x[selected]
    y = all_y[selected]
    d = all_d[selected]
    if len(x) < 2:
        raise ValueError("Smoke test requires at least two windows.")

    device = torch.device(args.device)
    input_channels = int(x.shape[1]) if x.ndim == 3 else 1
    model = SEGSCLSpecialist(
        input_channels=input_channels,
        token_dim=args.token_dim,
        num_tokens=args.num_tokens,
        num_domains=len(domains),
        condition_dim=0,
    ).to(device)
    generator = torch.Generator().manual_seed(42)
    class_names = list(dataset["class_names"])
    placeholder_prototypes = torch.randn(
        len(class_names),
        args.token_dim,
        generator=generator,
    )
    bank = FrozenPrototypeBank(
        placeholder_prototypes,
        class_names,
        version="random-connectivity-placeholder",
    ).to(device)

    xb = torch.tensor(x, dtype=torch.float32, device=device)
    yb = torch.tensor(y, dtype=torch.long, device=device)
    db = torch.tensor(d, dtype=torch.long, device=device)
    output = model(xb)
    global_loss, logits = global_prototype_alignment_loss(
        output.fault_embedding,
        bank.prototypes,
        yb,
    )
    cross_condition = cross_condition_supervised_contrastive_loss(
        output.fault_embedding,
        yb,
        db,
    )
    decorrelation = cross_covariance_loss(
        output.fault_embedding,
        output.condition_embedding,
    )
    total = global_loss + cross_condition + 0.01 * decorrelation
    total.backward()

    report = {
        "status": "ok",
        "warning": "Random prototypes verify connectivity only; not a paper result.",
        "dataset": args.dataset,
        "domains": domains,
        "input_shape": list(xb.shape),
        "signal_token_shape": list(output.signal_tokens.shape),
        "fault_embedding_shape": list(output.fault_embedding.shape),
        "logit_shape": list(logits.shape),
        "global_loss": float(global_loss.detach().cpu()),
        "cross_condition_loss": float(cross_condition.detach().cpu()),
        "decorrelation_loss": float(decorrelation.detach().cpu()),
        "parameters_with_grad": sum(
            int(parameter.grad is not None) for parameter in model.parameters()
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
