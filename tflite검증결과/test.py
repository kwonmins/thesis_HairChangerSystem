# test.py — TFLite evaluator (single-folder; Korean labels in labels.txt)
# 구조: test/ 안에 model_unquant.tflite, labels.txt, test.py, 그리고 모든 이미지 파일
# 파일명 예: 141realround.png, 00002egg.jpg, 00003angular.jpg, 00004diamond.png, ...

import argparse, json, time, csv, re
from pathlib import Path
import numpy as np
from PIL import Image

# ---- TFLite import ----
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ---------- labels.txt 읽기 (행이 "0 둥근형" 같은 형식일 때 숫자 무시하고 라벨만 추출) ----------
def load_labels(p: Path):
    items = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"^\s*(\d+)\s+(.+)$", ln)
        if m:
            idx, name = int(m.group(1)), m.group(2).strip()
        else:
            idx, name = len(items), ln
        items.append((idx, name))
    items.sort(key=lambda x: x[0])
    return [name for _, name in items]

def _norm(s: str) -> str:
    # 공백 제거 + 소문자 (한글은 그대로 유지)
    return re.sub(r"\s+", "", s).lower()

# ---------- 파일명 키워드 → labels.txt(한글) 인덱스 매칭 준비 ----------
def build_keyword_index(labels):
    """
    파일명에서 찾을 키워드 사전 구성.
    - 영문/한글 별칭을 labels 목록에 맞춰 동적으로 매핑
    - 길이가 긴 키워드를 우선 매칭
    """
    # labels.txt가 정확히 다음 다섯 클래스를 포함한다고 가정:
    # 둥근형, 긴형, 각진형, 다이아몬드형, 계란형
    # (순서는 파일의 줄 순서로 결정)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # 각 라벨에 대응하는 키워드들(파일명에 등장할 가능성 높은 것들)
    synonyms = {
        "둥근형": [
            "realround", "round", "circle", "원형", "둥근",
        ],
        "긴형": [
            "long", "oblong", "긴", "길쭉",
        ],
        "각진형": [
            "angular", "square", "rect", "rectangle", "사각", "각진",
        ],
        "다이아몬드형": [
            "diamond", "rhombus", "다이아", "마름모",
        ],
        "계란형": [
            "egg", "oval", "ellipse", "계란", "타원",
        ],
    }

    # labels.txt에 존재하는 라벨만 대상으로 키워드 매핑 구성
    token_to_idx = {}
    for lab, toks in synonyms.items():
        if lab not in label_to_idx:
            continue
        idx = label_to_idx[lab]
        # 라벨 자체(한글)도 키워드로 포함
        toks2 = toks + [lab]
        for t in toks2:
            token_to_idx[_norm(t)] = idx

    # 길이 긴 토큰 우선
    tokens_sorted = sorted(token_to_idx.items(), key=lambda kv: len(kv[0]), reverse=True)
    return tokens_sorted  # [(token_norm, idx), ...]

def infer_idx_from_filename(stem: str, tokens_sorted):
    s = _norm(stem)
    for tok, idx in tokens_sorted:
        if tok and tok in s:
            return idx
    return None

def list_images_from_folder(root: Path, tokens_sorted):
    items, not_matched = [], []
    for fp in root.iterdir():
        if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
            idx = infer_idx_from_filename(fp.stem, tokens_sorted)
            if idx is not None:
                items.append((fp, idx))
            else:
                not_matched.append(fp.name)
    if not_matched:
        print(f"[WARN] 라벨 매칭 실패 {len(not_matched)}개 (예시 10):", not_matched[:10])
    return items

def confusion_init(n): return [[0]*n for _ in range(n)]

# -------------------------------- main --------------------------------
def main():
    base = Path(__file__).resolve().parent  # 현재 test.py가 있는 폴더

    ap = argparse.ArgumentParser(description="TFLite evaluator (single-folder; Korean labels)")
    ap.add_argument("--model", default=str(base / "model_unquant.tflite"))
    ap.add_argument("--labels", default=str(base / "labels.txt"))
    ap.add_argument("--data", default=str(base))  # 이미지가 같은 폴더에 있으므로 기본값 = base
    ap.add_argument("--preds_csv", default=str(base / "preds_tflite.csv"))
    args = ap.parse_args()

    model_path, labels_path, data_root = map(lambda p: Path(p).resolve(), [args.model, args.labels, args.data])
    print(f"[PATH] model:{model_path}")
    print(f"[PATH] labels:{labels_path}")
    print(f"[PATH] data  :{data_root}")

    if not model_path.exists():  raise SystemExit(f"[ERR] no model: {model_path}")
    if not labels_path.exists(): raise SystemExit(f"[ERR] no labels: {labels_path}")
    if not data_root.exists():   raise SystemExit(f"[ERR] no data folder: {data_root}")

    labels = load_labels(labels_path)
    n = len(labels)
    tokens_sorted = build_keyword_index(labels)

    # 모델 로드(바이트 방식: 한글/공백 경로 안전)
    model_bytes = model_path.read_bytes()
    intr = tflite.Interpreter(model_content=model_bytes)
    intr.allocate_tensors()

    in_det  = intr.get_input_details()[0]
    out_det = intr.get_output_details()[0]
    _, H, W, C = in_det["shape"]
    in_dtype = in_det["dtype"]

    # 워밍업
    intr.set_tensor(in_det["index"], np.zeros((1, H, W, C), dtype=in_dtype))
    for _ in range(5): intr.invoke()

    # 데이터 수집(같은 폴더의 이미지 전부)
    items = list_images_from_folder(data_root, tokens_sorted)
    if not items:
        raise SystemExit("[ERR] 평가할 이미지가 없습니다. 파일명에 라벨 키워드가 들어있는지 확인하세요.")

    conf = confusion_init(n)
    times = []
    total = correct = 0

    with open(args.preds_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path","y_true","y_pred","correct","latency_ms"])
        for fp, y_true in items:
            img = Image.open(fp).convert("RGB").resize((W, H), Image.BICUBIC)
            x = np.asarray(img)
            if in_dtype == np.float32:
                x = x.astype(np.float32) / 255.0
            else:
                x = x.astype(np.uint8)
            x = np.expand_dims(x, 0)

            intr.set_tensor(in_det["index"], x)
            t0 = time.perf_counter_ns(); intr.invoke(); t1 = time.perf_counter_ns()
            ms = (t1 - t0) / 1e6; times.append(ms)

            y = intr.get_tensor(out_det["index"])[0]
            y_pred = int(np.argmax(y))

            conf[y_true][y_pred] += 1
            total += 1
            ok = int(y_pred == y_true); correct += ok
            w.writerow([str(fp), y_true, y_pred, ok, f"{ms:.3f}"])

    # 지표
    acc = correct / total
    per_class = []
    for c in range(n):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(n)) - tp
        fn = sum(conf[c]) - tp
        precision = 0 if (tp+fp)==0 else tp/(tp+fp)
        recall    = 0 if (tp+fn)==0 else tp/(tp+fn)
        f1        = 0 if (precision+recall)==0 else 2*precision*recall/(precision+recall)
        per_class.append({"precision":precision,"recall":recall,"f1":f1})

    avg_ms = float(np.mean(times)); med_ms = float(np.median(times))
    summary = {
        "framework":"tflite",
        "total": total,
        "accuracy": acc,
        "avg_ms": avg_ms,
        "median_ms": med_ms,
        "labels": labels,
        "confusion": conf,
        "perClass": per_class
    }
    out_json = base / "results_tflite.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] total={total}, acc={acc:.4f}, median_ms={med_ms:.2f}  -> {out_json}, {args.preds_csv}")

if __name__ == "__main__":
    main()
