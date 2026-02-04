import argparse
import os
import sys
from typing import Any, Dict

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from portfolio import run_models


def _ensure_plotly():
    import plotly.express as px

    return px


def _save_weight_pies(out_dir: str, label: str, opt_block: Dict[str, Any]):
    px = _ensure_plotly()
    for obj in ["max_sharpe", "min_vol"]:
        w = opt_block[obj]["weights"]
        df = pd.DataFrame({"symbol": list(w.keys()), "weight": list(w.values())}).sort_values("weight", ascending=False)
        fig = px.pie(df, names="symbol", values="weight", title=f"{label.upper()} weights - {obj}")
        fig.write_html(os.path.join(out_dir, f"{label}_weights_{obj}.html"), include_plotlyjs="cdn", full_html=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json"))
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.json))
    payload = run_models(args.json, out_dir=out_dir)
    _save_weight_pies(out_dir, "assets", payload["assets"]["opt"])
    _save_weight_pies(out_dir, "stocks", payload["stocks"]["opt"])
    _save_weight_pies(out_dir, "universe", payload["universe"]["opt"])


if __name__ == "__main__":
    main()
