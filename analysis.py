import pandas as pd
import torch
from typing import Dict, Any, Optional, Union

@torch.no_grad()
def evaluate_subjective_model(model,test_loader,device: Optional[torch.device] = None,) -> Dict[str, Any]:
    """
    Evaluates a multimodal evidential model on test_loader.
    Assumes model outputs evidences (Dirichlet α-1), NOT logits.

    Returns dict with:
        - per_view: list of view-level metrics
        - fused: fused metrics
        - per_class_evidence:
            {
            'unconditional': {'per_view': [K], 'fused': [K]},
            'true_class':    {'per_view': [K], 'fused': [K]}
            }
        where each [K] is a list of averages per class index.
    """
    model.eval()
    dev = device or next(model.parameters()).device
    K = int(getattr(model, "num_classes", None) or getattr(model, "num_labels", None))
    if K is None:
        raise ValueError("Could not infer num_classes from model; set model.num_classes.")

    def dirichlet_uncertainties(evi):  # evi: (B, C) evidence = alpha-1
        alphas = evi + 1.0
        S = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / S
        epistemic  = (K / S).squeeze(-1)  # matches your modules' "entropy" term
        aleatoric  = -torch.sum(probs * (torch.digamma(alphas + 1.0) - torch.digamma(S + 1.0)), dim=-1)
        total = epistemic + aleatoric
        return epistemic, aleatoric, total

    # -------- accumulators (global) --------
    per_view = None
    fused = {
        "N": 0, "correct": 0,
        "evidence_sum": 0.0, "epi_sum": 0.0, "ale_sum": 0.0, "tot_sum": 0.0,
        "inc_N": 0, "inc_evidence_sum": 0.0, "inc_epi_sum": 0.0, "inc_ale_sum": 0.0
    }

    # Per-class evidence accumulators
    fused_class_sum      = None  # (K,) unconditional sum over all samples
    fused_trueclass_sum  = None  # (K,) sum of evidence for the true class, per true class
    class_counts         = torch.zeros(K, device=dev)  # number of samples per true class
    N_total              = 0  # number of samples (for unconditional means)

    per_view_class_sum     = None  # list of (K,) per view
    per_view_trueclass_sum = None  # list of (K,) per view

    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            inputs = [b.to(dev).float() for b in batch[:-1]]
            target = batch[-1].to(dev)
        else:
            raise ValueError("Batch must be a (inputs..., labels) tuple/list.")

        # Strict path: identical to your Lightning flow
        if hasattr(model, "shared_step"):
            # shared_step returns: loss, evidences_a, target, evidences
            loss, fused_ev, target_out, evidences = model.shared_step([*inputs, target])
        else:
            # Fallback: forward -> evidences list, then aggregate
            ev_list = model(inputs)                              # list of (B, C) evidences
            evidences = torch.stack(ev_list, dim=1)              # (B, V, C)
            aggregator = getattr(model, "aggregation", None)
            if aggregator is None:
                raise ValueError("Model must expose an 'aggregation' function/attr.")
            fused_ev = aggregator(evidences)                     # (B, C)
            target_out = target

        B, V = target_out.size(0), evidences.size(1)

        # lazily init per-view accumulators
        if per_view is None:
            per_view = [{
                "N": 0, "correct": 0,
                "evidence_sum": 0.0, "epi_sum": 0.0, "ale_sum": 0.0, "tot_sum": 0.0,
                "inc_N": 0, "inc_evidence_sum": 0.0, "inc_epi_sum": 0.0, "inc_ale_sum": 0.0
            } for _ in range(V)]
            fused_class_sum      = torch.zeros(K, device=dev)
            fused_trueclass_sum  = torch.zeros(K, device=dev)
            per_view_class_sum     = [torch.zeros(K, device=dev) for _ in range(V)]
            per_view_trueclass_sum = [torch.zeros(K, device=dev) for _ in range(V)]

        # ----- fused metrics -----
        fused["N"] += B
        N_total    += B

        fused_evidence = fused_ev.sum(dim=-1)  # (B,)
        fe_epi, fe_ale, fe_tot = dirichlet_uncertainties(fused_ev)

        fused["evidence_sum"] += fused_evidence.sum().item()
        fused["epi_sum"]      += fe_epi.sum().item()
        fused["ale_sum"]      += fe_ale.sum().item()

        
        fused_preds = fused_ev.argmax(dim=-1)
        fused_correct_mask = (fused_preds == target_out)
        fused["correct"] += fused_correct_mask.sum().item()

        # incorrect-only (fused)
        if (~fused_correct_mask).any():
            idx = ~fused_correct_mask
            fused["inc_N"]            += idx.sum().item()
            fused["inc_evidence_sum"] += fused_evidence[idx].sum().item()
            fused["inc_epi_sum"]      += fe_epi[idx].sum().item()
            fused["inc_ale_sum"]      += fe_ale[idx].sum().item()

        # fused per-class evidence accumulations
        fused_class_sum += fused_ev.sum(dim=0)  # (C,)
        # sum of evidence assigned to TRUE class, bucketed by that class
        fused_trueclass_sum += torch.bincount(
            target_out,
            weights=fused_ev[torch.arange(B, device=dev), target_out],
            minlength=K
        )
        # update true-class counts
        class_counts += torch.bincount(target_out, minlength=K)

        # ----- per-view metrics -----
        for v in range(V):
            ev_v = evidences[:, v, :]                 # (B, C)
            pv = per_view[v]
            pv["N"] += B

            ev_scalar = ev_v.sum(dim=-1)
            v_epi, v_ale, v_tot = dirichlet_uncertainties(ev_v)
            pv["evidence_sum"] += ev_scalar.sum().item()
            pv["epi_sum"]      += v_epi.sum().item()
            pv["ale_sum"]      += v_ale.sum().item()

            preds_v = ev_v.argmax(dim=-1)
            correct_mask_v = (preds_v == target_out)
            pv["correct"] += correct_mask_v.sum().item()

            if (~correct_mask_v).any():
                idxv = ~correct_mask_v
                pv["inc_N"]            += idxv.sum().item()
                pv["inc_evidence_sum"] += ev_scalar[idxv].sum().item()
                pv["inc_epi_sum"]      += v_epi[idxv].sum().item()
                pv["inc_ale_sum"]      += v_ale[idxv].sum().item()

            # per-view per-class evidence accumulations
            per_view_class_sum[v]     += ev_v.sum(dim=0)  # (C,)
            per_view_trueclass_sum[v] += torch.bincount(
                target_out,
                weights=ev_v[torch.arange(B, device=dev), target_out],
                minlength=K
            )

    # ---- reduce helpers ----
    def reduce_block(b):
        return {
            "accuracy": (b["correct"] / b["N"]) if b["N"] > 0 else 0.0,
            "evidence_mean": (b["evidence_sum"] / b["N"]) if b["N"] > 0 else 0.0,
            "epistemic_mean": (b["epi_sum"] / b["N"]) if b["N"] > 0 else 0.0,
            "aleatoric_mean": (b["ale_sum"] / b["N"]) if b["N"] > 0 else 0.0,
            "incorrect_only": {
                "evidence_mean": (b["inc_evidence_sum"] / b["inc_N"]) if b["inc_N"] > 0 else 0.0,
                "epistemic_mean": (b["inc_epi_sum"] / b["inc_N"]) if b["inc_N"] > 0 else 0.0,
                "aleatoric_mean": (b["inc_ale_sum"] / b["inc_N"]) if b["inc_N"] > 0 else 0.0,
            }
        }

    # Per-class evidence means
    eps = 1e-12  # avoid div-by-zero
    fused_uncond_mean = (fused_class_sum / max(N_total, 1)).tolist()
    fused_truecls_mean = (fused_trueclass_sum / torch.clamp(class_counts, min=eps)).tolist()

    per_view_uncond_mean = [(per_view_class_sum[v] / max(N_total, 1)).tolist()
                            for v in range(len(per_view_class_sum))]
    per_view_truecls_mean = [(per_view_trueclass_sum[v] / torch.clamp(class_counts, min=eps)).tolist()
                                for v in range(len(per_view_trueclass_sum))]

    results = {
        "per_view": [reduce_block(pv) for pv in per_view],
        "fused": reduce_block(fused),
        "per_class_evidence": {
            "unconditional": {
                "per_view": per_view_uncond_mean,  # list[V][K]
                "fused": fused_uncond_mean         # list[K]
            },
            "true_class": {
                "per_view": per_view_truecls_mean, # list[V][K]
                "fused": fused_truecls_mean        # list[K]
            }
        }
    }
    return results

@torch.no_grad()
def evaluate_subjective_model_with_shared(
    model,
    test_loader,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluate a multimodal evidential model whose evidences list is:
        [shared, view_0, view_1, ..., view_{N-1}]
    Assumes outputs are evidences (Dirichlet α-1).

    Returns:
      {
        "shared": {accuracy, evidence_mean, epistemic_mean, aleatoric_mean,
                   incorrect_only:{evidence_mean, epistemic_mean, aleatoric_mean}},
        "per_view": [ ... one dict per specific view ... ],
        "fused":  {same keys as 'shared'},
        "per_class_evidence": {
           "unconditional": {"shared":[K], "per_view":[[K],...], "fused":[K]},
           "true_class":    {"shared":[K], "per_view":[[K],...], "fused":[K]}
        }
      }
    """
    model.eval()
    dev = device or next(model.parameters()).device
    K = int(getattr(model, "num_classes", None) or getattr(model, "num_labels", None))
    if K is None:
        raise ValueError("Could not infer num_classes from model; set model.num_classes.")

    def dirichlet_epistemic_aleatoric(evi):  # evi: (B, C) evidence = alpha-1
        alphas = evi + 1.0
        S = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / S
        epistemic = (K / S).squeeze(-1)  # same as your "entropy" term
        aleatoric = -torch.sum(probs * (torch.digamma(alphas + 1.0) - torch.digamma(S + 1.0)), dim=-1)
        return epistemic, aleatoric

    # ---------------- accumulators (no total uncertainty) ----------------
    def make_acc():
        return {
            "N": 0, "correct": 0,
            "evidence_sum": 0.0, "epi_sum": 0.0, "ale_sum": 0.0,
            "inc_N": 0, "inc_evidence_sum": 0.0, "inc_epi_sum": 0.0, "inc_ale_sum": 0.0,
        }

    shared_acc = make_acc()
    fused = make_acc()
    per_view = None  # lazily init after seeing V

    # per-class accumulators
    N_total = 0
    class_counts = None
    fused_class_sum = None
    fused_trueclass_sum = None
    shared_class_sum = None
    shared_trueclass_sum = None
    per_view_class_sum = None
    per_view_trueclass_sum = None

    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            inputs = [b.to(dev).float() for b in batch[:-1]]
            target = batch[-1].to(dev)
        else:
            raise ValueError("Batch must be a (inputs..., labels) tuple/list.")

        # Strict path: reuse model.shared_step if available
        if hasattr(model, "shared_step"):
            _, fused_ev, target_out, evidences = model.shared_step([*inputs, target])
        else:
            ev_list = model(inputs)  # list with ev_list[0] = shared
            evidences = torch.stack(ev_list, dim=1)  # (B, V, C)
            aggregator = getattr(model, "agg", None) or getattr(model, "aggregation", None)
            if aggregator is None:
                raise ValueError("Model must expose an 'agg' or 'aggregation' function.")
            fused_ev = aggregator(evidences)
            target_out = target

        B, V, C = evidences.shape
        if V < 2:
            raise ValueError("Expected at least one shared and one specific view (V >= 2).")
        N_total += B

        # lazy init
        if per_view is None:
            per_view = [make_acc() for _ in range(V - 1)]
            fused_class_sum = torch.zeros(K, device=dev)
            fused_trueclass_sum = torch.zeros(K, device=dev)
            shared_class_sum = torch.zeros(K, device=dev)
            shared_trueclass_sum = torch.zeros(K, device=dev)
            per_view_class_sum = [torch.zeros(K, device=dev) for _ in range(V - 1)]
            per_view_trueclass_sum = [torch.zeros(K, device=dev) for _ in range(V - 1)]
            class_counts = torch.zeros(K, device=dev)

        # -------- fused --------
        fused["N"] += B
        fused_evidence = fused_ev.sum(dim=-1)
        fe_epi, fe_ale = dirichlet_epistemic_aleatoric(fused_ev)
        fused["evidence_sum"] += fused_evidence.sum().item()
        fused["epi_sum"]      += fe_epi.sum().item()
        fused["ale_sum"]      += fe_ale.sum().item()
        preds = fused_ev.argmax(dim=-1)
        correct_mask = (preds == target_out)
        fused["correct"] += correct_mask.sum().item()
        if (~correct_mask).any():
            idx = ~correct_mask
            fused["inc_N"]            += idx.sum().item()
            fused["inc_evidence_sum"] += fused_evidence[idx].sum().item()
            fused["inc_epi_sum"]      += fe_epi[idx].sum().item()
            fused["inc_ale_sum"]      += fe_ale[idx].sum().item()

        fused_class_sum += fused_ev.sum(dim=0)
        fused_trueclass_sum += torch.bincount(
            target_out, weights=fused_ev[torch.arange(B, device=dev), target_out], minlength=K
        )
        class_counts += torch.bincount(target_out, minlength=K)

        # -------- shared (index 0) --------
        ev_sh = evidences[:, 0, :]
        shared_acc["N"] += B
        sh_evidence = ev_sh.sum(dim=-1)
        sh_epi, sh_ale = dirichlet_epistemic_aleatoric(ev_sh)
        shared_acc["evidence_sum"] += sh_evidence.sum().item()
        shared_acc["epi_sum"]      += sh_epi.sum().item()
        shared_acc["ale_sum"]      += sh_ale.sum().item()
        sh_preds = ev_sh.argmax(dim=-1)
        sh_correct_mask = (sh_preds == target_out)
        shared_acc["correct"] += sh_correct_mask.sum().item()
        if (~sh_correct_mask).any():
            idxs = ~sh_correct_mask
            shared_acc["inc_N"]            += idxs.sum().item()
            shared_acc["inc_evidence_sum"] += sh_evidence[idxs].sum().item()
            shared_acc["inc_epi_sum"]      += sh_epi[idxs].sum().item()
            shared_acc["inc_ale_sum"]      += sh_ale[idxs].sum().item()

        shared_class_sum += ev_sh.sum(dim=0)
        shared_trueclass_sum += torch.bincount(
            target_out, weights=ev_sh[torch.arange(B, device=dev), target_out], minlength=K
        )

        # -------- specific views (1..V-1 => view_0..view_{V-2}) --------
        for v in range(1, V):
            ev_v = evidences[:, v, :]
            pv = per_view[v - 1]
            pv["N"] += B
            ev_scalar = ev_v.sum(dim=-1)
            v_epi, v_ale = dirichlet_epistemic_aleatoric(ev_v)
            pv["evidence_sum"] += ev_scalar.sum().item()
            pv["epi_sum"]      += v_epi.sum().item()
            pv["ale_sum"]      += v_ale.sum().item()
            preds_v = ev_v.argmax(dim=-1)
            correct_mask_v = (preds_v == target_out)
            pv["correct"] += correct_mask_v.sum().item()
            if (~correct_mask_v).any():
                idxv = ~correct_mask_v
                pv["inc_N"]            += idxv.sum().item()
                pv["inc_evidence_sum"] += ev_scalar[idxv].sum().item()
                pv["inc_epi_sum"]      += v_epi[idxv].sum().item()
                pv["inc_ale_sum"]      += v_ale[idxv].sum().item()

            per_view_class_sum[v - 1] += ev_v.sum(dim=0)
            per_view_trueclass_sum[v - 1] += torch.bincount(
                target_out, weights=ev_v[torch.arange(B, device=dev), target_out], minlength=K
            )

    # ---------------- reduce ----------------
    def reduce_block(b):
        return {
            "accuracy": (b["correct"] / b["N"]) if b["N"] > 0 else 0.0,
            "evidence_mean": (b["evidence_sum"] / b["N"]) if b["N"] > 0 else 0.0,
            "epistemic_mean": (b["epi_sum"] / b["N"]) if b["N"] > 0 else 0.0,
            "aleatoric_mean": (b["ale_sum"] / b["N"]) if b["N"] > 0 else 0.0,
            "incorrect_only": {
                "evidence_mean": (b["inc_evidence_sum"] / b["inc_N"]) if b["inc_N"] > 0 else 0.0,
                "epistemic_mean": (b["inc_epi_sum"] / b["inc_N"]) if b["inc_N"] > 0 else 0.0,
                "aleatoric_mean": (b["inc_ale_sum"] / b["inc_N"]) if b["inc_N"] > 0 else 0.0,
            }
        }

    eps = 1e-12
    fused_uncond_mean = (fused_class_sum / max(N_total, 1)).tolist()
    fused_truecls_mean = (fused_trueclass_sum / torch.clamp(class_counts, min=eps)).tolist()
    shared_uncond_mean = (shared_class_sum / max(N_total, 1)).tolist()
    shared_truecls_mean = (shared_trueclass_sum / torch.clamp(class_counts, min=eps)).tolist()
    per_view_uncond_mean = [(per_view_class_sum[i] / max(N_total, 1)).tolist()
                            for i in range(len(per_view_class_sum))]
    per_view_truecls_mean = [(per_view_trueclass_sum[i] / torch.clamp(class_counts, min=eps)).tolist()
                             for i in range(len(per_view_trueclass_sum))]

    return {
        "shared": reduce_block(shared_acc),
        "per_view": [reduce_block(pv) for pv in per_view],
        "fused": reduce_block(fused),
        "per_class_evidence": {
            "unconditional": {
                "shared": shared_uncond_mean,
                "per_view": per_view_uncond_mean,
                "fused": fused_uncond_mean
            },
            "true_class": {
                "shared": shared_truecls_mean,
                "per_view": per_view_truecls_mean,
                "fused": fused_truecls_mean
            }
        }
    }


def flatten_sample_info(
    sample_info: Dict[str, Any],
    *,
    seed: Union[int, str],
    pct: Union[int, float, str],
    model: str
) -> Dict[str, Any]:
    """
    Flatten a single 'Sample information' dict to one row.
    Supports:
      - fused metrics (prefix 'fused_')
      - shared metrics (prefix 'shared_')         <-- NEW
      - per-view metrics (prefix 'view_{i}_')
      - per-class evidence (unconditional & true_class) for fused/shared/per-view
    """
    row = {"seed": seed, "dep": pct, "model": model}

    # ---- helper to flatten a metrics dict into row with a given prefix ----
    def add_block(prefix: str, block: Dict[str, Any]):
        if not isinstance(block, dict): 
            return
        for k in ["accuracy", "evidence_mean", "epistemic_mean", "aleatoric_mean"]:
            if k in block:
                row[f"{prefix}{k}"] = float(block[k])
        inc = block.get("incorrect_only", {})
        for k in ["evidence_mean", "epistemic_mean", "aleatoric_mean"]:
            if k in inc:
                row[f"{prefix}incorrect_only_{k}"] = float(inc[k])

    # ---- fused block (unchanged) ----
    add_block("fused_", sample_info.get("fused", {}))

    # ---- NEW: shared block ----
    add_block("shared_", sample_info.get("shared", {}))

    # ---- per-view blocks (view_0, view_1, ...) ----
    per_view = sample_info.get("per_view", [])
    for i, v in enumerate(per_view):
        add_block(f"view_{i}_", v)

    # ---- per-class evidence ----
    pce = sample_info.get("per_class_evidence", {})
    uncond = pce.get("unconditional", {})
    truec  = pce.get("true_class", {})

    # fused per-class
    fused_uncond = uncond.get("fused")
    if isinstance(fused_uncond, (list, tuple)):
        for k, val in enumerate(fused_uncond):
            row[f"fused_per_class_evidence_class_{k}"] = float(val)
    fused_truec = truec.get("fused")
    if isinstance(fused_truec, (list, tuple)):
        for k, val in enumerate(fused_truec):
            row[f"fused_per_class_evidence_true_class_{k}"] = float(val)

    # NEW: shared per-class
    shared_uncond = uncond.get("shared")
    if isinstance(shared_uncond, (list, tuple)):
        for k, val in enumerate(shared_uncond):
            row[f"shared_per_class_evidence_class_{k}"] = float(val)
    shared_truec = truec.get("shared")
    if isinstance(shared_truec, (list, tuple)):
        for k, val in enumerate(shared_truec):
            row[f"shared_per_class_evidence_true_class_{k}"] = float(val)

    # per-view per-class (lists of arrays)
    view_uncond = uncond.get("per_view", [])
    for i, arr in enumerate(view_uncond):
        if isinstance(arr, (list, tuple)):
            for k, val in enumerate(arr):
                row[f"view_{i}_per_class_evidence_class_{k}"] = float(val)

    view_truec = truec.get("per_view", [])
    for i, arr in enumerate(view_truec):
        if isinstance(arr, (list, tuple)):
            for k, val in enumerate(arr):
                row[f"view_{i}_per_class_evidence_true_class_{k}"] = float(val)

    return row


def build_metrics_dataframe(nested: Dict[Any, Dict[Any, Dict[str, Dict[str, Any]]]]) -> pd.DataFrame:
    """
    Convert:
        nested[seed][pct][model] = sample_info_dict
    into a tidy DataFrame.
    """
    rows = []
    for seed, d_pct in nested.items():
        for pct, d_model in d_pct.items():
            for model, sample_info in d_model.items():
                rows.append(flatten_sample_info(sample_info, seed=seed, pct=pct, model=model))
    df = pd.DataFrame(rows)

    id_cols = ["seed", "dep", "model"]
    other_cols = sorted([c for c in df.columns if c not in id_cols])
    return df[id_cols + other_cols]


def build_metrics_dataframe_datasets(nested: Dict[Any, Dict[Any, Dict[str, Dict[str, Any]]]]) -> pd.DataFrame:
    """
    Convert:
        nested[seed][pct][model] = sample_info_dict
    into a tidy DataFrame.
    """
    rows = []
    for seed, d_pct in nested.items():
        for typ, d_ds in d_pct.items():
            for ds, d_model in d_ds.items(): 
                for model, sample_info in d_model.items():
                    rows.append(flatten_sample_info_datasets(sample_info, seed=seed, typ=typ, ds=ds, model=model))
    df = pd.DataFrame(rows)

    id_cols = ["seed", "type","dataset", "model"]
    other_cols = sorted([c for c in df.columns if c not in id_cols])
    return df[id_cols + other_cols]

def flatten_sample_info_datasets(
    sample_info: Dict[str, Any],
    *,
    seed: Union[int, str],
    typ: str,
    ds:str,
    model: str
) -> Dict[str, Any]:
    """
    Flatten a single 'Sample information' dict to one row.
    Supports:
      - fused metrics (prefix 'fused_')
      - shared metrics (prefix 'shared_')         <-- NEW
      - per-view metrics (prefix 'view_{i}_')
      - per-class evidence (unconditional & true_class) for fused/shared/per-view
    """
    row = {"seed": seed, "type": typ, "dataset": ds,'model':model}

    # ---- helper to flatten a metrics dict into row with a given prefix ----
    def add_block(prefix: str, block: Dict[str, Any]):
        if not isinstance(block, dict): 
            return
        for k in ["accuracy", "evidence_mean", "epistemic_mean", "aleatoric_mean"]:
            if k in block:
                row[f"{prefix}{k}"] = float(block[k])
        inc = block.get("incorrect_only", {})
        for k in ["evidence_mean", "epistemic_mean", "aleatoric_mean"]:
            if k in inc:
                row[f"{prefix}incorrect_only_{k}"] = float(inc[k])

    # ---- fused block (unchanged) ----
    add_block("fused_", sample_info.get("fused", {}))

    # ---- NEW: shared block ----
    add_block("shared_", sample_info.get("shared", {}))

    # ---- per-view blocks (view_0, view_1, ...) ----
    per_view = sample_info.get("per_view", [])
    for i, v in enumerate(per_view):
        add_block(f"view_{i}_", v)

    # ---- per-class evidence ----
    pce = sample_info.get("per_class_evidence", {})
    uncond = pce.get("unconditional", {})
    truec  = pce.get("true_class", {})

    # fused per-class
    fused_uncond = uncond.get("fused")
    if isinstance(fused_uncond, (list, tuple)):
        for k, val in enumerate(fused_uncond):
            row[f"fused_per_class_evidence_class_{k}"] = float(val)
    fused_truec = truec.get("fused")
    if isinstance(fused_truec, (list, tuple)):
        for k, val in enumerate(fused_truec):
            row[f"fused_per_class_evidence_true_class_{k}"] = float(val)

    # NEW: shared per-class
    shared_uncond = uncond.get("shared")
    if isinstance(shared_uncond, (list, tuple)):
        for k, val in enumerate(shared_uncond):
            row[f"shared_per_class_evidence_class_{k}"] = float(val)
    shared_truec = truec.get("shared")
    if isinstance(shared_truec, (list, tuple)):
        for k, val in enumerate(shared_truec):
            row[f"shared_per_class_evidence_true_class_{k}"] = float(val)

    # per-view per-class (lists of arrays)
    view_uncond = uncond.get("per_view", [])
    for i, arr in enumerate(view_uncond):
        if isinstance(arr, (list, tuple)):
            for k, val in enumerate(arr):
                row[f"view_{i}_per_class_evidence_class_{k}"] = float(val)

    view_truec = truec.get("per_view", [])
    for i, arr in enumerate(view_truec):
        if isinstance(arr, (list, tuple)):
            for k, val in enumerate(arr):
                row[f"view_{i}_per_class_evidence_true_class_{k}"] = float(val)

    return row

