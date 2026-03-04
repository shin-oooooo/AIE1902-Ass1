import json
import pandas as pd
import os

try:
    with open('c:/Users/zaoji/Desktop/Ass1/Ass1/models.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Classification Metrics (LightGBM):")
    for scope in ['assets', 'stocks', 'universe']:
        lgb = data.get(scope, {}).get('lightgbm', {})
        if 'classification' in lgb:
            cls_data = lgb['classification']
            print(f"--- {scope} ---")
            for sym, res in cls_data.items():
                print(f"{sym}: Accuracy={res.get('accuracy', 'N/A')}")
        else:
             print(f"--- {scope} : No classification data ---")

except Exception as e:
    print(f"Error: {e}")
