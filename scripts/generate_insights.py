# generate_insights.py
import pandas as pd
import os
from PIL import Image
from datetime import datetime

TABLE_DIR = "../results/tables"
MODEL_DIR = "../results/modeling"
PLOT_DIRS = ["../results/plots/eda", "../results/modeling"]
OUT_DIR = "../results/reports"
os.makedirs(OUT_DIR, exist_ok=True)

def read_pareto():
    p = os.path.join(TABLE_DIR, "pareto_top10.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

def read_model_comp():
    p = os.path.join(MODEL_DIR, "model_comparison.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

def compose_text(pareto_df, model_df):
    lines = []
    lines.append(f"Insights Summary - generated {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    if pareto_df is not None:
        lines.append("Top equipment by failures:\n")
        for i, row in pareto_df.head(5).iterrows():
            lines.append(f" - {row['Equipment']}: {int(row['Count'])} failures, cumulative {row['Cumulative%']}%\n")
    if model_df is not None:
        best = model_df.sort_values("CV_Mean_Acc", ascending=False).iloc[0]
        lines.append(f"\nModel comparison (best model): {best['Model']} with CV mean acc {best['CV_Mean_Acc']:.3f} and test acc {best['Test_Acc']:.3f}\n")
    lines.append("\nRecommended next steps: prioritize maintenance for top Pareto equipment; validate best model on new data; run cost-effectiveness simulation.\n")
    # save
    with open(os.path.join(OUT_DIR, "insights_summary.txt"), "w") as f:
        f.writelines(lines)
    print("Saved insights_summary.txt")

def stitch_plots_to_pdf(out_pdf=os.path.join(OUT_DIR,"insights_report.pdf")):
    images = []
    for d in PLOT_DIRS:
        if os.path.exists(d):
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(".png"):
                    images.append(os.path.join(d, fn))
    pil_images = []
    for img_path in images:
        pil_images.append(Image.open(img_path).convert('RGB'))
    if pil_images:
        pil_images[0].save(out_pdf, save_all=True, append_images=pil_images[1:])
        print(f"Saved stitched PDF to {out_pdf}")
    else:
        print("No plots found to stitch.")

if __name__ == "__main__":
    pareto = read_pareto()
    models = read_model_comp()
    compose_text(pareto, models)
    stitch_plots_to_pdf()
