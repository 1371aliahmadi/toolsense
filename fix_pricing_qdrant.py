"""
Patch Qdrant pricing_tier payloads — run this on Windows (once, from toolsense/).

    python fix_pricing_qdrant.py

The clean_tools.csv already has the correct pricing classifications.
This script reads it and updates the matching Qdrant point payloads.
No re-embedding needed — only the payload field is changed.
"""
import sys
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CLEAN_CSV   = ROOT / "data" / "clean_tools.csv"
QDRANT_PATH = str(ROOT / "data" / "qdrant")
CFG_PATH    = ROOT / "config" / "base_config.yaml"

def main():
    # Load updated CSV
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded {len(df)} tools")
    print("Pricing distribution:")
    print(df["pricing_tier"].value_counts().to_string())

    # Load config
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    collection = cfg["qdrant"]["collection_name"]

    # Connect to Qdrant
    from qdrant_client import QdrantClient
    client = QdrantClient(path=QDRANT_PATH)
    print(f"\nConnected to Qdrant collection: '{collection}'")

    # Patch each point's pricing_tier
    print("Patching payloads...")
    for i, row in df.iterrows():
        client.set_payload(
            collection_name=collection,
            payload={"pricing_tier": row["pricing_tier"]},
            points=[i],
        )
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)} updated")

    print(f"\nAll {len(df)} Qdrant payloads updated.")

    # Verify
    hits = client.scroll(collection_name=collection, limit=10, with_payload=True)[0]
    print("\nSample after patch:")
    for h in hits[:5]:
        print(f"  [{h.payload['pricing_tier']:10}] {h.payload['tool_name'][:45]}")

    client.close()
    print("\nDone. Restart the API server if it is running.")

if __name__ == "__main__":
    main()
