# make_report.py
import json, os, sys, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---- (권장) 윈도우 한글 폰트 설정: '맑은 고딕' 우선, 없으면 기본 폰트 ----
import matplotlib
for font in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

def load_json(p):
    p = Path(p)
    if not p.exists(): return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_txt(path, text):
    Path(path).write_text(text, encoding="utf-8")

def bar_simple(title, labels, values, out):
    plt.figure(figsize=(6,5))
    plt.bar(labels, values)
    plt.ylim(0, 1.0 if "정확" in title or "Accuracy" in title or max(values)<=1.0 else None)
    plt.title(title)
    for i,v in enumerate(values):
        plt.text(i, v, f"{v:.3f}" if v<=1.0 else f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def bar_per_class_f1(title, class_names, f1_values, out):
    plt.figure(figsize=(9,6))
    x = np.arange(len(class_names))
    plt.bar(x, f1_values)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.title(title)
    for i,v in enumerate(f1_values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_confusion(conf, labels, out, title="혼동행렬"):
    conf = np.array(conf, dtype=float)
    plt.figure(figsize=(8,6))
    im = plt.imshow(conf, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("예측 클래스"); plt.ylabel("정답 클래스")
    plt.title(title)
    # 값 표시
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            val = int(conf[i,j]) if float(conf[i,j]).is_integer() else round(conf[i,j],2)
            plt.text(j, i, str(val), ha="center", va="center", color="white" if conf[i,j] > conf.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def tables_figure(conf, labels, per_class, out, title_left="혼동행렬", title_right="클래스별 지표"):
    conf = np.array(conf, dtype=int)
    n = len(labels)
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].axis('off'); axes[1].axis('off')

    # 혼동행렬 표
    cm_header = [""] + [f"pred\n{l}" for l in labels]
    cm_rows = [[f"true\n{labels[i]}"] + list(map(int, conf[i])) for i in range(n)]
    cm_table = axes[0].table(cellText=cm_rows, colLabels=cm_header, loc='center', cellLoc='center')
    cm_table.scale(1, 1.4)
    axes[0].set_title(title_left, pad=10, fontweight="bold")

    # per-class 표
    pc_header = ["라벨", "Precision", "Recall", "F1"]
    def fmt(x): return f"{x:.3f}"
    pc_rows = [[labels[i], fmt(per_class[i]["precision"]), fmt(per_class[i]["recall"]), fmt(per_class[i]["f1"])] for i in range(n)]
    pc_table = axes[1].table(cellText=pc_rows, colLabels=pc_header, loc='center', cellLoc='center')
    pc_table.scale(1, 1.4)
    axes[1].set_title(title_right, pad=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def summary_text(name, R):
    lines = []
    lines.append(f"== {name} ==")
    lines.append(f"total     : {R['total']}")
    lines.append(f"accuracy  : {R['accuracy']:.4f}")
    lines.append(f"avg_ms    : {R['avg_ms']:.3f}")
    lines.append(f"median_ms : {R['median_ms']:.3f}")
    lines.append("per-class :")
    for i,lbl in enumerate(R["labels"]):
        m = R["perClass"][i]
        lines.append(f"  - {lbl} | P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")
    return "\n".join(lines)

def main():
    # 파일 자동 탐색 (필요하면 이름 바꿔도 됨)
    tflite = load_json("results_tflite.json")
    tfjs   = load_json("results_tfjs.json")

    if not tflite and not tfjs:
        print("[ERR] results_tflite.json / results_tfjs.json 둘 다 없습니다.")
        sys.exit(1)

    report_lines = []

    # ---- 개별 프레임워크 처리 ----
    entries = []
    if tflite: entries.append(("TFLite", tflite))
    if tfjs:   entries.append(("TFJS", tfjs))

    # 공통 라벨 (가능하면 첫 엔트리의 라벨 사용)
    labels = entries[0][1]["labels"]

    # 정확도/지연시간 그래프용 데이터
    acc_x, acc_y = [], []
    lat_x, lat_y = [], []

    for name, R in entries:
        acc_x.append(name); acc_y.append(R["accuracy"])
        lat_x.append(name); lat_y.append(R["median_ms"])

        # per-class F1
        f1s = [pc["f1"] for pc in R["perClass"]]
        bar_per_class_f1("클래스별 F1", R["labels"], f1s, f"chart_f1_{name.lower()}.png")

        # 혼동행렬 그림 & 테이블
        plot_confusion(R["confusion"], R["labels"], f"cm_{name.lower()}.png", f"혼동행렬 - {name}")
        tables_figure(R["confusion"], R["labels"], R["perClass"], f"tables_{name.lower()}.png")

        report_lines.append(summary_text(name, R))
        report_lines.append("")

    # 정확도/지연시간 막대
    bar_simple("정확도(Accuracy)", acc_x, acc_y, "chart_accuracy.png")
    bar_simple("지연시간 중앙값 (ms)", lat_x, lat_y, "chart_latency.png")

    # 텍스트 요약
    save_txt("report.txt", "\n".join(report_lines))
    print("생성:",
          "chart_accuracy.png, chart_latency.png, "
          + ", ".join([f"chart_f1_{n.lower()}.png" for n,_ in entries]) + ", "
          + ", ".join([f"cm_{n.lower()}.png" for n,_ in entries]) + ", "
          + ", ".join([f"tables_{n.lower()}.png" for n,_ in entries]) + ", report.txt")

if __name__ == "__main__":
    main()
