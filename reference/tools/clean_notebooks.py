"""Strip widget state, HTML renders and Colab metadata from notebooks; keep code and text outputs."""
import json
import pathlib
import sys

WIDGET_MIME = "application/vnd.jupyter.widget-view+json"

def clean(path: pathlib.Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb.get("metadata", {}).pop("widgets", None)
    nb.get("metadata", {}).get("colab", {}).pop("authorship_tag", None)
    for cell in nb.get("cells", []):
        meta = cell.get("metadata", {})
        meta.pop("executionInfo", None)
        meta.pop("outputId", None)
        if "outputs" not in cell:
            continue
        kept = []
        for out in cell["outputs"]:
            data = out.get("data")
            if data is not None:
                if WIDGET_MIME in data:
                    continue                      # drop progress-bar widgets
                out["data"] = {k: v for k, v in data.items() if k == "text/plain"}
                out.get("metadata", {}).clear()
                if not out["data"]:
                    continue
            kept.append(out)                      # stream, error and text outputs survive
        cell["outputs"] = kept
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"cleaned {path}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        clean(pathlib.Path(p))