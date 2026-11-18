"""Assemble the Hugging Face deliverables into deploy/:

  deploy/hf_model/  -> push to a HF *model* repo (model card + joblib pipelines + metadata)
  deploy/hf_space/  -> push to a HF *Space* (Gradio SDK), self-contained

Then (after `huggingface-cli login`):
  python scripts/build_hf.py --push --user YOUR_HF_USERNAME
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


def build():
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    reqs = {l.split("==")[0].lower(): l for l in (ROOT / "requirements.txt").read_text().split() if "==" in l}

    # ---------------- model repo
    mdir = DEPLOY / "hf_model"
    shutil.rmtree(mdir, ignore_errors=True); (mdir / "src").mkdir(parents=True)
    for f in ("spike_classifier.joblib", "price_change_regressor.joblib", "metadata.json"):
        shutil.copy(ROOT / "models" / f, mdir / f)
    shutil.copytree(ROOT / "src" / "kiranbazaar", mdir / "src" / "kiranbazaar",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "deploy" / "MODEL_CARD.md", mdir / "README.md")

    # ---------------- space
    sdir = DEPLOY / "hf_space"
    shutil.rmtree(sdir, ignore_errors=True)
    (sdir / "models").mkdir(parents=True); (sdir / "data" / "processed").mkdir(parents=True); (sdir / "data" / "raw").mkdir()
    shutil.copy(ROOT / "app" / "gradio_app.py", sdir / "app.py")
    shutil.copytree(ROOT / "src", sdir / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("spike_classifier.joblib", "price_change_regressor.joblib", "metadata.json"):
        shutil.copy(ROOT / "models" / f, sdir / "models" / f)
    shutil.copy(ROOT / "data" / "processed" / "features.csv", sdir / "data" / "processed" / "features.csv")
    shutil.copy(ROOT / "data" / "raw" / "wfp_food_prices_pak.csv", sdir / "data" / "raw" / "wfp_food_prices_pak.csv")
    keep = ["scikit-learn", "pandas", "numpy", "scipy", "joblib", "matplotlib"]
    (sdir / "requirements.txt").write_text("\n".join(reqs[k] for k in keep if k in reqs) + "\n")
    (sdir / "README.md").write_text(f"""---
title: Kiran Bazaar
emoji: 🛒
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: {reqs['gradio'].split('==')[1]}
python_version: "3.13"
app_file: app.py
pinned: false
license: other
license_name: cc-by-igo-3.0
short_description: Will a staple food's price spike next month in Pakistan?
---
# Kiran Bazaar
Predicts whether the retail price of a food staple in a Pakistani city will rise by more than
{int(meta['spike_threshold'] * 100)}% next month. Model: {meta['classifier']} (test ROC-AUC
{meta['test_metrics_classifier']['roc_auc']:.2f}). Data: WFP Food Prices for Pakistan (HDX, CC BY-IGO).
""")
    print("built", mdir, "and", sdir)


def push(user):
    from huggingface_hub import HfApi
    api = HfApi()
    for kind, folder, name in [("model", "hf_model", "kiran-bazaar-spike-classifier"), ("space", "hf_space", "kiran-bazaar")]:
        repo = f"{user}/{name}"
        api.create_repo(repo, repo_type=kind, exist_ok=True, **({"space_sdk": "gradio"} if kind == "space" else {}))
        api.upload_folder(folder_path=str(DEPLOY / folder), repo_id=repo, repo_type=kind)
        print("pushed", f"https://huggingface.co/{'spaces/' if kind == 'space' else ''}{repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--user")
    a = ap.parse_args()
    build()
    if a.push:
        push(a.user)
