import os, time, csv, requests, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_one(anime_id: int, outdir: str) -> None:
    jpg = os.path.join(outdir, f"{anime_id}.jpg")
    png = os.path.join(outdir, f"{anime_id}.png")
    if os.path.exists(jpg) or os.path.exists(png):
        return
    try:
        r = requests.get(f"https://api.jikan.moe/v4/anime/{anime_id}", timeout=8)
        if r.status_code != 200:
            return
        data = r.json() or {}
        images = (data.get("data") or {}).get("images") or {}
        jpg_url = (images.get("jpg") or {}).get("large_image_url") or (images.get("jpg") or {}).get("image_url")
        png_url = (images.get("png") or {}).get("large_image_url") or (images.get("png") or {}).get("image_url")
        url = jpg_url or png_url
        if not url:
            return
        img = requests.get(url, timeout=10)
        if img.status_code == 200:
            ext = ".jpg" if "jpg" in url else ".png"
            with open(os.path.join(outdir, f"{anime_id}{ext}"), "wb") as f:
                f.write(img.content)
            time.sleep(0.2)  # respeita API pública
    except Exception:
        pass

def main(csv_path: str, outdir: str, workers: int = 8):
    os.makedirs(outdir, exist_ok=True)
    ids = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                aid = int(row.get("anime_id") or row.get("animeid") or 0)
                if aid > 0: ids.append(aid)
            except Exception:
                continue
    ids = list(sorted(set(ids)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_one, aid, outdir) for aid in ids]
        for _ in as_completed(futs):
            pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Caminho para Anime.csv")
    ap.add_argument("--out", required=True, help="Pasta de saída (ex.: data/images)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(args.csv, args.out, args.workers)
