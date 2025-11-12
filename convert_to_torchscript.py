import argparse
import glob
import os
import shutil
from typing import List, Tuple

import torch
import torch.nn as nn
import re

from models.sequence_models import TextRNNClassifier


DOMAINS = ["Books", "Movies", "Games"]


def discover_weight_files(root: str) -> List[Tuple[str, str]]:
    """
    Return list of (domain, filepath) for .pt/.pth weights in known domain folders.
    """
    results: List[Tuple[str, str]] = []
    for domain in DOMAINS:
        dpath = os.path.join(root, domain)
        if not os.path.isdir(dpath):
            continue
        for pattern in ("*.pt", "*.pth"):
            for f in glob.glob(os.path.join(dpath, pattern)):
                results.append((domain, f))
    return results


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_as_artifact(domain: str, src_file: str, scripted: torch.jit.ScriptModule, root: str) -> str:
    """
    Save TorchScript module under artifacts/<domain>/ with a filename derived from src_file.
    Additionally, if the filename contains 'best', also save/overwrite model.ts as canonical.
    """
    artifacts_dir = os.path.join(root, "artifacts", domain.lower())
    ensure_dir(artifacts_dir)
    base = os.path.splitext(os.path.basename(src_file))[0]
    out_path = os.path.join(artifacts_dir, f"{base}.ts")
    torch.jit.save(scripted, out_path)

    # If 'best' in name, also create/overwrite model.ts as canonical
    if "best" in base.lower():
        canonical = os.path.join(artifacts_dir, "model.ts")
        # copy rather than symlink for portability
        shutil.copyfile(out_path, canonical)
        return f"{out_path} (and canonical model.ts)"
    return out_path


def try_convert_file(domain: str, filepath: str, root: str, seq_len: int = 256) -> Tuple[bool, str]:
    """
    Attempts the following:
      1) Load as TorchScript via torch.jit.load and re-save
      2) Load via torch.load:
         - If ScriptModule, re-save
         - If nn.Module, try script(), then fallback to trace() with LongTensor [1, seq_len]
         - If state_dict or unknown, report unsupported
    Returns (success, message_with_output_or_error).
    """
    try:
        # Case 1: Already TorchScript (jit saved)
        scripted = torch.jit.load(filepath, map_location="cpu")
        out = save_as_artifact(domain, filepath, scripted, root)
        return True, f"Loaded as TorchScript and saved to {out}"
    except Exception:
        pass

    try:
        obj = torch.load(filepath, map_location="cpu")
    except Exception as e:
        return False, f"Failed to torch.load: {e}"

    # Case 2a: ScriptModule via torch.load
    if isinstance(obj, torch.jit.ScriptModule):
        out = save_as_artifact(domain, filepath, obj, root)
        return True, f"Loaded ScriptModule and saved to {out}"

    # Case 2b: A pickled nn.Module that we can attempt to script/trace
    if isinstance(obj, nn.Module):
        model = obj
        model.eval()
        # Try scripting first
        try:
            scripted = torch.jit.script(model)
            out = save_as_artifact(domain, filepath, scripted, root)
            return True, f"Scripted nn.Module and saved to {out}"
        except Exception:
            # Fallback to tracing with dummy integer token ids
            try:
                dummy_input = torch.ones((1, seq_len), dtype=torch.long)
                traced = torch.jit.trace(model, dummy_input, strict=False)
                out = save_as_artifact(domain, filepath, traced, root)
                return True, f"Traced nn.Module and saved to {out}"
            except Exception as e:
                return False, f"Failed to script/trace nn.Module: {e}"

    # Case 2c: state_dict or unknown structure
    if isinstance(obj, dict):
        state = obj
        # Infer component names
        cell_type = "rnn"
        for ct in ("lstm", "gru", "rnn"):
            prefix = f"{ct}."
            if any(k.startswith(prefix) for k in state.keys()):
                cell_type = ct
                break

        # Detect bidirectionality via reverse parameters
        bidirectional = any("_reverse" in k for k in state.keys())

        # Determine number of layers by scanning *_l{n} occurrences for the selected cell
        layer_indices = set()
        pattern = re.compile(rf"^{cell_type}\.(?:weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)")
        for k in state.keys():
            m = pattern.match(k)
            if m:
                layer_indices.add(int(m.group(1)))
        num_layers = (max(layer_indices) + 1) if layer_indices else 1

        # Embedding dims
        if "embedding.weight" not in state:
            return False, "state_dict missing 'embedding.weight' – cannot infer vocab/embedding dims."
        vocab_size, embed_dim = state["embedding.weight"].shape

        # Hidden dim from recurrent weights (use weight_hh_l0 which is [gates*hidden, hidden] for LSTM/GRU or [hidden, hidden] for RNN)
        weight_hh_key = f"{cell_type}.weight_hh_l0"
        if weight_hh_key not in state:
            return False, f"state_dict missing '{weight_hh_key}'."
        hidden_dim = state[weight_hh_key].shape[1]

        # Output dim from classifier
        if "fc.weight" not in state:
            return False, "state_dict missing 'fc.weight' – cannot infer output_dim."
        output_dim = state["fc.weight"].shape[0]

        # Dropout: cannot be inferred from weights; choose common default (0.5 if present in training)
        dropout = 0.5

        # Build a compatible model and load weights (strict=False to allow harmless mismatches)
        model = TextRNNClassifier(
            cell_type=cell_type,
            vocab_size=vocab_size,
            embed_dim=int(embed_dim),
            hidden_dim=int(hidden_dim),
            output_dim=int(output_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            bidirectional=bool(bidirectional),
            padding_idx=0,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        # If critical parameters failed to load, report
        critical_prefixes = ("embedding.", f"{cell_type}.", "fc.")
        critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
        if critical_missing:
            return False, f"Failed to load critical params: {critical_missing}; unexpected: {unexpected}"

        model.eval()
        try:
            scripted = torch.jit.script(model)
        except Exception:
            # Fallback to tracing with dummy integer ids
            dummy_input = torch.ones((1, seq_len), dtype=torch.long)
            scripted = torch.jit.trace(model, dummy_input, strict=False)
        out = save_as_artifact(domain, filepath, scripted, root)
        return True, f"Built model (cell={cell_type}, bi={bidirectional}, layers={num_layers}) and saved to {out}"

    return False, f"Unsupported object type: {type(obj)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .pt/.pth weights to TorchScript bundles.")
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Project root containing Books/Movies/Games folders.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Dummy sequence length for tracing if needed.",
    )
    args = parser.parse_args()

    files = discover_weight_files(args.root)
    if not files:
        print("No .pt/.pth files found under Books/Movies/Games.")
        return

    print(f"Discovered {len(files)} weight files.")
    successes = 0
    for domain, f in files:
        ok, msg = try_convert_file(domain, f, args.root, seq_len=args.seq_len)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {domain}: {f} -> {msg}")
        if ok:
            successes += 1

    print(f"Converted {successes}/{len(files)} files.")
    print(
        "Note: For inference in the app, please also provide artifacts/<domain>/labels.txt "
        "and artifacts/<domain>/preprocessor.json to match each model's training setup."
    )


if __name__ == "__main__":
    main()


