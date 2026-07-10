"""Q24 downloader: CoinMetrics Community daily on-chain metrics -> research/data/onchain/{asset}.csv

Source: https://community-api.coinmetrics.io/v4 (keyless community API; github mirror
coinmetrics-io/data was ~6 weeks stale as of 2026-07-10, API is fresh to T-2).
Assets: the 25 of our 141 CM-name-mapped perp bases that actually expose network metrics
(AdrActCnt+TxCnt) in the community set. TfrValUSD / FeeTotUSD do NOT exist in the
community set; we take FeeTotNtv (14 assets) and TxTfrCnt instead.
Resumable: skips an asset if its CSV already ends on/after END_DATE.
Gentle: 1 catalog + ~1 timeseries page per asset, 0.4s sleep.
"""
import csv
import io
import json
import os
import time
import urllib.request

BASE = "https://community-api.coinmetrics.io/v4"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "onchain")
ASSETS = ['aave', 'ada', 'algo', 'bch', 'bnb', 'btc', 'comp', 'crv', 'dash', 'doge',
          'dot', 'etc', 'eth', 'icp', 'ldo', 'link', 'ltc', 'mana', 'snx', 'trx',
          'uni', 'xlm', 'xrp', 'xtz', 'zec']
METRICS = ["AdrActCnt", "TxCnt", "TxTfrCnt", "FeeTotNtv", "CapMrktCurUSD", "PriceUSD"]
START = "2019-01-01"
END_DATE = "2026-07-05"  # consider file complete if it reaches this date (or asset discontinued)


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "traderbor-research-q24"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def main():
    os.makedirs(OUT, exist_ok=True)
    catalog = {}
    for a in ASSETS:
        j = get_json(f"{BASE}/catalog-v2/asset-metrics?assets={a}")
        mets = {}
        for m in j["data"][0]["metrics"]:
            for f in m["frequencies"]:
                if f["frequency"] == "1d":
                    mets[m["metric"]] = [f["min_time"][:10], f["max_time"][:10]]
        catalog[a] = {k: v for k, v in mets.items() if k in METRICS}
        time.sleep(0.2)
    with open(os.path.join(OUT, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=1, sort_keys=True)
    print("catalog written for", len(catalog), "assets")

    for a in ASSETS:
        path = os.path.join(OUT, f"{a}.csv")
        avail = sorted(catalog[a])
        max_avail = max(v[1] for v in catalog[a].values())
        if os.path.exists(path):
            with open(path, "rb") as f:
                try:
                    f.seek(-200, 2)
                except OSError:
                    pass
                last = f.read().decode().strip().splitlines()[-1][:10]
            if last >= min(END_DATE, max_avail):
                print(a, "up-to-date, skip")
                continue
        rows = []
        url = (f"{BASE}/timeseries/asset-metrics?assets={a}&metrics={','.join(avail)}"
               f"&frequency=1d&start_time={START}&page_size=10000")
        while url:
            j = get_json(url)
            rows += j["data"]
            url = j.get("next_page_url")
            time.sleep(0.4)
        if not rows:
            print(a, "NO DATA")
            continue
        cols = ["time"] + avail
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r["time"][:10]] + [r.get(m, "") for m in avail])
        with open(path, "w", newline="") as f:
            f.write(buf.getvalue())
        print(a, "rows:", len(rows), "cols:", avail, "last:", rows[-1]["time"][:10])
        time.sleep(0.4)


if __name__ == "__main__":
    main()
