"""Build the assessment report (reports/assessment_report.docx) from saved results.

Draws three report-specific charts into reports/figures/ and assembles the DOCX with python-docx.
All numbers are read from outputs/ or copied from the script outputs listed next to each section.

Run:  python reports/build_report.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "outputs"
DOCX = ROOT / "reports" / "assessment_report.docx"

TEAL, ORANGE, GREY, INK, MUTED = "#0B6E7A", "#D9822B", "#B8C2C4", "#1F2A2E", "#5B6B70"
FONT = "Calibri"


# =============================================================================== report charts
def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9DAFB3")
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="x", color="#E3E8E9", linewidth=0.8)
    ax.set_axisbelow(True)


def chart_backtest_timeline(path: Path):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    folds = [("Fold 1", 4, 2), ("Fold 2", 6, 2), ("Fold 3", 7, 2), ("Fold 4", 8, 2), ("Final", 10, 2)]
    fig, ax = plt.subplots(figsize=(9, 3.1), dpi=200)
    for i, (name, n_train, n_pred) in enumerate(folds):
        y = len(folds) - 1 - i
        ax.barh(y, n_train, left=0, height=0.62, color=TEAL if name != "Final" else "#064A56", edgecolor="white", linewidth=2)
        ax.barh(y, n_pred, left=n_train, height=0.62, color=ORANGE, edgecolor="white", linewidth=2)
        ax.text(n_train / 2, y, "train" if name != "Final" else "train on all labelled data", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ax.text(n_train + n_pred / 2, y, "predict" if name != "Final" else "validation.csv", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f[0] for f in folds][::-1], fontsize=9.5, color=INK)
    ax.set_xticks([m + 0.5 for m in range(12)])
    ax.set_xticklabels(months, fontsize=9, color=INK)
    ax.set_xlim(0, 12)
    ax.axvline(10, color=MUTED, lw=1, ls=":")
    ax.text(10.05, -0.62, "labels end 31 Oct", fontsize=8, color=MUTED, va="center")
    ax.set_ylim(-0.85, len(folds) - 0.55)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title("Rolling-origin backtest: every fold predicts the two months after its training window", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_model_comparison(path: Path, rows: list[tuple[str, float, bool]]):
    fig, ax = plt.subplots(figsize=(9, 4.3), dpi=200)
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [TEAL if r[2] else GREY for r in rows][::-1]
    ax.barh(range(len(rows)), vals, color=colors, height=0.66, edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v + 3, i, f"${v:,.1f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(48.3, color=ORANGE, lw=1.6, ls="--")
    ax.set_ylim(-0.6, len(rows) + 0.25)
    ax.text(48.3 + 2, len(rows) - 0.15, "noise floor $48.3 (a perfect model)", fontsize=8, color=INK, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel("Backtest MAE, mean of 4 folds ($)", fontsize=9, color=INK)
    _style_axes(ax)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title("Backtest MAE by model (lower is better)", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_permutation(path: Path, perm: pd.Series):
    perm = perm.sort_values(ascending=True)
    perm = perm[perm > 0.2]
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=200)
    ax.barh(range(len(perm)), perm.values, color=TEAL, height=0.66, edgecolor="white", linewidth=2)
    for i, v in enumerate(perm.values):
        ax.text(v + 1.2, i, f"+${v:,.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(perm)))
    ax.set_yticklabels(perm.index, fontsize=9, color=INK)
    ax.set_xlabel("Increase in clean-row MAE when the feature is shuffled ($), fold 4", fontsize=9, color=INK)
    _style_axes(ax)
    ax.set_xlim(0, perm.max() * 1.15)
    ax.set_title("Permutation importance (features adding under $0.20 omitted)", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_experiments() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in glob.glob(str(OUT / "experiment_log*.csv"))]
    d = pd.concat(frames)
    return d[d.fold == "MEAN"].drop_duplicates("exp", keep="last").set_index("exp")


# =============================================================================== docx helpers
def shade(cell, hex_fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def set_cell_margins(table, top=60, bottom=60, left=100, right=100):
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        el = OxmlElement(f"w:{k}")
        el.set(qn("w:w"), str(v))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)


def table(doc, header: list[str], rows: list[list[str]], widths_cm: list[float], bold_last=False, highlight_row: int | None = None, font_size=9):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_cell_margins(t)
    for r_i, values in enumerate([header] + rows):
        for c_i, val in enumerate(values):
            cell = t.rows[r_i].cells[c_i]
            cell.width = Cm(widths_cm[c_i])
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            run = p.add_run(str(val))
            run.font.size = Pt(font_size)
            run.font.name = FONT
            if r_i == 0:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(cell, "0B6E7A")
            elif highlight_row is not None and r_i - 1 == highlight_row:
                run.bold = True
                shade(cell, "E3F1F2")
            elif bold_last and r_i == len(rows):
                run.bold = True
            if c_i > 0 and r_i > 0 and _looks_numeric(val):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    # repeat header row on page breaks
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def _looks_numeric(s: str) -> bool:
    s = str(s).strip().replace("$", "").replace("%", "").replace(",", "").replace("−", "-").replace("+", "")
    try:
        float(s)
        return True
    except ValueError:
        return False


def para(doc, text: str, bold_lead: str | None = None, size=10.5, after=6, italic=False, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    if bold_lead:
        r = p.add_run(bold_lead + " ")
        r.bold = True
        r.font.size = Pt(size)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.italic = italic
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    return p


def bullets(doc, items: list[str | tuple[str, str]], size=10.5):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        if isinstance(it, tuple):
            r = p.add_run(it[0] + " ")
            r.bold = True
            r.font.size = Pt(size)
            r = p.add_run(it[1])
            r.font.size = Pt(size)
        else:
            r = p.add_run(it)
            r.font.size = Pt(size)


def figure(doc, path: Path, caption: str, width_cm=16.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.LEFT
    c.paragraph_format.space_after = Pt(10)
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor.from_string("5B6B70")


def code(doc, text: str):
    for line in text.strip("\n").split("\n"):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(0.4)
        r = p.add_run(line)
        r.font.name = "Consolas"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        r.font.size = Pt(8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run("Freight rate prediction  |  page ")
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor.from_string("5B6B70")
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    rr = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "17")
    rpr.append(sz)
    rr.append(rpr)
    tt = OxmlElement("w:t")
    tt.text = "1"
    rr.append(tt)
    fld.append(rr)
    p._p.append(fld)


def setup_styles(doc):
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    st.font.size = Pt(10.5)
    st.font.color.rgb = RGBColor.from_string("1F2A2E")
    for name, size, before, after, color in (("Heading 1", 15, 16, 6, "064A56"), ("Heading 2", 12, 10, 4, "0B6E7A"), ("Title", 26, 0, 6, "064A56")):
        s = doc.styles[name]
        s.font.name = FONT
        s.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        rf = s.element.rPr.rFonts
        rf.set(qn("w:ascii"), FONT)
        rf.set(qn("w:hAnsi"), FONT)
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = RGBColor.from_string(color)
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True


# =============================================================================== build
def main():
    exp = load_experiments()
    cfg = json.loads((OUT / "final_config.json").read_text())
    summ = json.loads((OUT / "final_model_summary.json").read_text())
    perm = pd.read_csv(OUT / "permutation_importance.csv", index_col=0)["MAE_increase_$"]

    def m(exp_id, col="MAE"):
        return float(exp.loc[exp_id, col])

    final_id = "xgb_gpu+ridge|log_rpm|clean|level|lvlfe3d5|d4m50"
    best_no_level = "xgb_gpu|log|clean|full"
    comparison = [
        ("Predict the median $/mile", m("dummy_median|log_rpm|all|full"), False),
        ("Linear regression, price target", m("linear|raw|all|full"), False),
        ("Linear regression, log $/mile", m("linear|log_rpm|all|full"), False),
        ("LightGBM, log $/mile", m("lgbm_l2|log_rpm|clean|full"), False),
        ("CatBoost GPU, log price", m("cat_gpu|log|clean|full"), False),
        ("XGBoost GPU, log price", m(best_no_level), False),
        ("Ridge + market-level model", m("ridge|log_rpm|clean|level|lvlfe3|oof"), False),
        ("XGBoost + market-level model", m("xgb_gpu|log_rpm|clean|level|lvlfe3|oof"), False),
        ("Final: XGBoost + Ridge, level with drift", m(final_id), True),
    ]
    chart_backtest_timeline(FIG / "report_backtest_timeline.png")
    chart_model_comparison(FIG / "report_model_comparison.png", comparison)
    chart_permutation(FIG / "report_permutation_importance.png", perm)

    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    add_page_number_footer(sec)

    # ------------------------------------------------------------------ title
    t = doc.add_paragraph(style="Title")
    t.add_run("Freight Rate Prediction")
    para(doc, "Machine Learning Engineer assessment: approach, validation and results", size=13, after=2, color="0B6E7A")
    para(doc, "Prepared by: [Your name]   |   November 2026", size=10, after=14, color="5B6B70")

    # ------------------------------------------------------------------ 1 executive summary
    doc.add_heading("1. Executive summary", level=1)
    para(doc, "The task is to predict the posted rate of 12,000 truckloads dated November and December 2025, using 48,000 labelled loads from January to October 2025. "
              "The provided score.py checks file format and draws a fixed December chart, but computes no accuracy metric, so all accuracy figures below come from a time-based backtest that mirrors the real task.")
    bullets(doc, [
        ("Two data traps decided the outcome.", "The quote_signal column equals the label in half the training months and is random noise in the validation file, so it was removed. And 1.4% of labels are corrupted prices at about 3.4 times or 0.29 times normal, which no feature predicts."),
        ("The key modelling step was separating the market level.", "Tree models alone under-predict future months, because a low market_index meant low prices early in the year but not later. Modelling the daily price level explicitly cut backtest MAE from $113.6 to $96.5."),
        ("The final model is simple.", "XGBoost on the GPU averaged with Ridge regression, both predicting log rate per mile after the market level is removed."),
        ("Both output files pass the official scorer.", "validation_predictions.csv holds 12,000 positive predictions in template order, and the December chart is included in section 13."),
    ])
    table(doc, ["Backtest metric (mean of 4 folds)", "Final model", "Best model without level step", "Predict median $/mile"], [
        ["MAE", f"${m(final_id):.1f}", f"${m(best_no_level):.1f}", f"${m('dummy_median|log_rpm|all|full'):.1f}"],
        ["MAPE", f"{m(final_id, 'MAPE'):.2f}%", f"{m(best_no_level, 'MAPE'):.2f}%", f"{m('dummy_median|log_rpm|all|full', 'MAPE'):.2f}%"],
        ["MAE on clean-label rows", f"${m(final_id, 'MAE_clean'):.1f}", f"${m(best_no_level, 'MAE_clean'):.1f}", f"${m('dummy_median|log_rpm|all|full', 'MAE_clean'):.1f}"],
        ["R²", f"{m(final_id, 'R2'):.3f}", f"{m(best_no_level, 'R2'):.3f}", f"{m('dummy_median|log_rpm|all|full', 'R2'):.3f}"],
    ], [6.2, 3.3, 4.2, 3.5])
    para(doc, "Even a perfect model would score an MAE of about $48 because of the corrupted labels, so the clean-row MAE is the clearest measure of model quality.", size=9.5, italic=True, color="5B6B70")

    # ------------------------------------------------------------------ 2 problem
    doc.add_heading("2. Problem and scoring", level=1)
    para(doc, "Each load has a pickup and delivery city with coordinates, a distance in miles, an equipment type (Dry Van, Reefer or Flatbed), a weight, a pickup date, a daily market_index and a quote_signal. "
              "The label, posted_rate, is the total price of the load in dollars. Training covers 1 January to 31 October 2025 and the validation file covers 1 November to 31 December 2025, so this is a forecasting problem: predict the two months after the labels end.")
    para(doc, "score.py enforces these constraints, which the pipeline asserts before writing any file:")
    bullets(doc, [
        "exactly the columns load_id,predicted_rate in that order, and exactly 12,000 rows;",
        "IDs exactly TE-000001 to TE-012000, with no duplicates;",
        "every prediction finite and strictly positive;",
        "31 December rows for the fixed lane Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb.",
    ])
    para(doc, "Because the official metric is unknown, MAE in dollars was the primary objective, with RMSE, MAPE and R² reported alongside it.")

    # ------------------------------------------------------------------ 3 EDA
    doc.add_heading("3. Data exploration", level=1)
    figure(doc, FIG / "01_target_distribution.png", "Figure 1. Posted rate is right-skewed. Rate per mile is tightly clustered around $2.15, apart from two small groups of corrupted labels far from the centre (the spike at 5 collects every value above $5 per mile).")
    bullets(doc, [
        ("Distance drives price.", "Its rank correlation with posted rate is 0.98. Rate per mile falls from $2.75 under 250 miles to $1.87 over 2,500 miles, because short hauls carry fixed costs."),
        ("Equipment and weight matter.", "Median rate per mile is $2.05 for Dry Van, $2.22 for Flatbed and $2.31 for Reefer. It rises from $2.06 for loads under 15,000 lb to $2.21 over 45,000 lb."),
        ("Prices follow the market and drift upward.", "market_index rose from 0.93 in January to 1.30 in May and fell back to 0.89 by September. Yet September prices were about 6% above January at a similar market level."),
        ("Calendar effects are small.", "Midweek prices are about 2% above weekend prices. Federal holidays differed from the surrounding fortnight by between −2.9% and +0.5%, with no consistent direction."),
        ("Coordinates are synthetic but consistent.", "Philadelphia sits in the Atlantic, yet road distance is 1.18 times the straight-line distance between coordinates with a correlation of 0.9995, so geographic features remain usable."),
    ])

    # ------------------------------------------------------------------ 4 data quality
    doc.add_heading("4. Data quality issues and how they were handled", level=1)
    table(doc, ["Issue", "Evidence", "Treatment"], [
        ["Negative weights: 0.6% train, 1.2% validation", "Absolute values match positive weights (KS p = 0.70); prices no different", "Use absolute value"],
        ["Missing weight: 300 train, 165 validation", "Unrelated to price (p = 0.23)", "Training median, plus a missing flag"],
        ["Weight at exactly 47,500 lb: 2.5% of both sets", "Legal gross-weight cap", "Keep, plus a capped flag"],
        ["Missing market_index: 374 train, 249 validation", "Every day has other values; unrelated to price (p = 0.46)", "Fill with that day's mean"],
        ["652 corrupted labels (1.36%)", "Prices at 3.4x or 0.29x the lane norm; no feature predicts them", "Exclude outside $1.20-$5.00 per mile from training; keep in evaluation"],
    ], [4.6, 6.6, 5.0], font_size=8.5)
    para(doc, "The corrupted labels set a hard floor. If a model knew every load's true underlying price, it would still score MAE $48.3, RMSE $577 and R² 0.849, and these rows account for 15% of the target's variance. "
              "That is why RMSE and R² barely change across models in this report.")

    # ------------------------------------------------------------------ 5 leakage
    doc.add_heading("5. Leakage investigation", level=1)
    figure(doc, FIG / "10_quote_match_daily.png", "Figure 2. Share of loads each day whose quote_signal is within 1% of the realised rate per mile. It switches between about 65% and about 4% by calendar month.")
    para(doc, "quote_signal looked like the strongest feature. A dedicated audit showed it cannot be used:", after=3)
    bullets(doc, [
        ("In January to March, June and September it is the label.", "Its correlation with rate per mile is 0.9991."),
        ("In the other five months it runs backwards,", "with a rank correlation of −0.38 against rate per mile."),
        ("In validation.csv it is random noise.", "Its daily mean varies exactly as random sampling predicts, and the split-half reliability of its lane means is −0.02, against 0.77 where it carries information."),
    ])
    para(doc, "A model using quote_signal scores well under any split of the training data and then fails on the scored loads. It is also absent from the December chart inputs. It was removed. Other checks:")
    table(doc, ["Check", "Finding", "Action"], [
        ["load_id", "Correlates 0.99998 with date", "Excluded"],
        ["Lane target encoding", "Must not see the test window", "Refit inside every fold"],
        ["In-sample lane encoding", "Leaked each day's market level (correlation 0.44 with market_index)", "Out-of-fold encoding for training rows"],
        ["Customer or carrier IDs", "None in the data", "No entity leakage to manage"],
    ], [4.0, 7.6, 4.6], font_size=8.5)

    # ------------------------------------------------------------------ 6 drift
    doc.add_heading("6. Development data versus the final validation data", level=1)
    figure(doc, FIG / "12_drift_numeric.png", "Figure 3. Training (teal) versus validation (orange) feature distributions. Only market_index shifts, and it moves to a level that training already covers.", width_cm=14.5)
    table(doc, ["Feature", "PSI vs all training", "PSI vs Sep-Oct"], [
        ["distance", "0.0006", "0.0025"],
        ["absolute weight", "0.0005", "0.0018"],
        ["market_index", "5.10", "0.071"],
        ["coordinates", "0.001-0.003", "0.003-0.004"],
        ["equipment", "0.0001", ""],
    ], [6.0, 4.5, 4.5])
    bullets(doc, [
        ("The load mix is unchanged;", "only the market level shifts, to about 0.92-0.94, close to September and October and inside the training range."),
        ("Eight cities are new", "(Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego). They touch 12.1% of validation rows, never both ends of a load. Coordinates let the model price them."),
        ("88% of validation rows are on a lane seen in training,", "so lane history is available for most loads but cannot be relied on for all."),
        ("Missing and negative values occur at about twice the training rate,", "which the cleaning rules handle."),
    ])

    # ------------------------------------------------------------------ 7 validation
    doc.add_heading("7. Train and validation split", level=1)
    para(doc, "Because the task is to forecast two months beyond the labels, validation must also forecast forward. A rolling-origin backtest was used: each fold trains on all months up to a cut-off and predicts the following two months.")
    figure(doc, FIG / "report_backtest_timeline.png", "Figure 4. The four backtest folds and the final fit. Fold 4 is the closest analogue to the real task, since its market level matches November and December.")
    table(doc, ["Fold", "Train", "Predict", "Train rows", "Test rows"], [
        ["1", "Jan-Apr", "May-Jun", "19,110", "9,696"],
        ["2", "Jan-Jun", "Jul-Aug", "28,806", "9,671"],
        ["3", "Jan-Jul", "Aug-Sep", "33,718", "9,429"],
        ["4", "Jan-Aug", "Sep-Oct", "38,477", "9,523"],
    ], [2.0, 3.2, 3.2, 3.2, 3.2])
    para(doc, "Leakage controls:", bold_lead=None, after=3)
    bullets(doc, [
        "every training date precedes every test date, verified for each fold by src/check_features.py;",
        "lane encodings and the market-level model are refitted on each fold's training window only;",
        "early stopping uses a holdout drawn from inside the training window, never the test fold;",
        "validation.csv was used only for its unlabelled features: drift checks and the December market_index.",
    ])
    para(doc, "Why not a random split? The same model was evaluated both ways:", after=3)
    table(doc, ["Scheme", "MAE", "MAPE", "R²"], [
        ["Random 5-fold", "$83.5", "3.82%", "0.843"],
        ["Time-based backtest", "$121.2", "5.17%", "0.828"],
        ["Random 5-fold with quote_signal added", "$78.1", "3.55%", "0.843"],
    ], [7.4, 2.8, 2.8, 2.8])
    para(doc, "A random split places loads from the same day on both sides, leaking the market level, and makes MAE look about 31% better than an honest forecast. Adding quote_signal flatters it further.")

    # ------------------------------------------------------------------ 8 features
    doc.add_heading("8. Features and target", level=1)
    table(doc, ["Group", "Features"], [
        ["Distance", "distance, log of distance"],
        ["Load", "absolute weight, weight-missing flag, weight-capped flag, equipment"],
        ["Geography", "four coordinates, straight-line distance, circuity (road over straight-line), direction as sine and cosine"],
        ["Cities", "pickup and delivery as categories; unseen cities fall back to coordinates"],
        ["Lane history", "smoothed lane price premium (out-of-fold for training rows), lane load count"],
        ["Calendar", "day of week, weekend flag, US federal holiday flag"],
        ["Market", "market_index, used by the market-level model rather than the trees"],
    ], [3.2, 13.0], font_size=9)
    para(doc, "The target is the log of rate per mile. This removes the dominant distance scaling, reduces skew, and guarantees positive predictions after back-transformation, which score.py requires.")

    # ------------------------------------------------------------------ 9 level model
    doc.add_heading("9. Separating the market level", level=1)
    para(doc, "Every tree model, whatever its settings, under-predicted fold 4 by about 3.6%. Trees learned that a low market_index means low prices because January had both, but September and October also had a low index with higher prices. Trees cannot separate the market from the upward drift, and they cannot extrapolate.")
    figure(doc, FIG / "14_temporal_component.png", "Figure 5. Top: the daily price level after adjusting for the mix of loads. Bottom: market_index, which follows a weekly cycle, with the validation period to the right of the dotted line.", width_cm=15.5)
    para(doc, "The market-level model handles this in three steps:", after=3)
    bullets(doc, [
        "compute each day's average log rate per mile, adjusted for equipment and distance mix;",
        "fit that level as a coefficient times market_index plus a separate effect for each month, so the coefficient is estimated from movement within months and is not confused with the drift;",
        "forecast a future month with the average of the last three month effects plus half the recent monthly drift, then subtract the level from the training target so the trees model only the load-specific part.",
    ])
    para(doc, "Bias of the forecast daily level on each fold's two-month horizon:", after=3)
    table(doc, ["Forecast rule", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Worst"], [
        ["market_index only", "−1.56%", "−1.61%", "−3.98%", "−4.60%", "4.60%"],
        ["market_index + linear trend", "−0.06%", "+4.53%", "+0.56%", "−1.49%", "4.53%"],
        ["Carry last 30 days forward", "−3.42%", "+4.43%", "+2.15%", "−1.92%", "4.43%"],
        ["market_index + last 3 months' level", "−2.09%", "−0.26%", "−1.17%", "−1.98%", "2.09%"],
    ], [6.0, 2.0, 2.0, 2.0, 2.0, 2.2], highlight_row=3, font_size=8.5)
    para(doc, "The three-month rule has the smallest worst case. On all training data the market coefficient is 0.1275, so a rise of 0.1 in market_index adds about 1.3% to the rate per mile, and the drift is 0.63% per month.")

    # ------------------------------------------------------------------ 10 experiments
    doc.add_heading("10. Model experiments", level=1)
    para(doc, f"{len(exp)} configurations were run through the same four-fold backtest and logged per fold in outputs/experiment_log*.csv. The main results:")
    figure(doc, FIG / "report_model_comparison.png", "Figure 6. Mean backtest MAE for the main configurations. Grey bars predict without the market-level step, except the two labelled with it.")
    bullets(doc, [
        ("Predicting log rate per mile beat predicting price,", "cutting linear-model MAE from $160.7 to $121.1."),
        ("Corrupted labels must be handled.", "Training on clean rows with an ordinary loss matched an L1 loss on all rows ($121.2 versus $121.3), while an ordinary loss on all rows was much worse ($142.4)."),
        ("Without the market-level step, gradient boosting gave modest gains,", "with XGBoost and CatBoost at about $114 against $121 for linear regression."),
        ("With it, the learner barely matters.", "XGBoost, LightGBM, HistGradientBoosting and Ridge all land between $101.5 and $102.6."),
        ("Features and tuning mattered little.", "Removing lane or city features changed MAE by under $0.50. Seven hyperparameter variants and the default all fell within $1 of each other."),
        ("Blending and drift gave the last gains.", "Averaging XGBoost with Ridge, plus half the recent drift in the level forecast, brought MAE to $96.5."),
    ])
    table(doc, ["Level forecast", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Mean"], [
        ["No drift", "$115.0", "$84.8", "$97.7", "$106.2", "$100.9"],
        ["Half drift (final)", "$104.2", "$90.0", "$93.8", "$98.0", "$96.5"],
        ["Full drift", "$98.1", "$107.3", "$96.5", "$94.7", "$99.1"],
    ], [5.2, 2.2, 2.2, 2.2, 2.2, 2.2], highlight_row=1)
    para(doc, "Half drift has the best mean and the best worst fold ($104.2, against $107.3 for full drift and $115.0 for none). Three random seeds of XGBoost differed by under $0.10, so results are stable.")

    # ------------------------------------------------------------------ 11 final model
    doc.add_heading("11. Final model", level=1)
    p = cfg["params"]
    table(doc, ["Setting", "Value"], [
        ["Models", "XGBoost (CUDA) and Ridge, averaged in log space"],
        ["XGBoost", f"depth {p['max_depth']}, min child weight {p['min_child_weight']}, learning rate 0.03, {p['n_estimators']:,} trees, row and column sampling 0.8"],
        ["Seeds", "3 XGBoost seeds, averaged"],
        ["Target", "log rate per mile minus the market level"],
        ["Market level", "last 3 month effects plus half the recent drift"],
        ["Training rows", "all 47,348 rows with rate per mile between $1.20 and $5.00"],
    ], [3.4, 12.8], font_size=9)
    para(doc, "The tree count follows the early-stopping trend: it rose from 707 to 2,062 rounds as training grew from 19,000 to 38,000 rows, so 2,500 suits the full 47,000 rows. "
              "This configuration was chosen because it ties for the lowest mean and worst-fold MAE of all configurations (two near-identical variants, with a leaner feature set or a different early-stopping holdout, are within $0.15), uses only features available at prediction time, and stays simple: one boosted model, one linear model and an explainable level adjustment.")

    # ------------------------------------------------------------------ 12 error analysis
    doc.add_heading("12. Error analysis", level=1)
    para(doc, "On fold 4 (train January to August, predict September to October), the final configuration scored:")
    table(doc, ["Rows", "MAE", "RMSE", "MAPE", "Median abs. error", "R²"], [
        ["All", "$98.0", "$631", "4.32%", "$27.7", "0.829"],
        ["Clean labels", "$45.3", "$153", "1.84%", "$27.0", "0.988"],
    ], [3.2, 2.4, 2.4, 2.4, 3.2, 2.4])
    figure(doc, FIG / "err_final_residuals.png", "Figure 7. Fold 4 errors. Points off the diagonal are the corrupted labels. Clean-row errors are tightly centred just below zero, and the daily bias dips in late September when the market moved sharply.", width_cm=15.0)
    table(doc, ["Segment", "Clean-row MAE"], [
        ["Dry Van / Reefer / Flatbed", "$39 / $47 / $61"],
        ["Under 250 miles / over 2,500 miles", "$11 / $105"],
        ["Lanes with 1-4 / 20+ training loads", "$43 / $50"],
        ["September / October", "$56 / $35"],
    ], [8.5, 5.0])
    bullets(doc, [
        ("Errors scale with price, not with data scarcity.", "Rare lanes are predicted as well as busy ones, because coordinates carry the geography."),
        ("A small negative bias remains:", "the median clean-row error is −1.0%, reaching −2.9% in the last week of September."),
        ("The largest clean-band errors are milder corrupted labels,", "priced at about 2.3 times normal, just inside the $5.00 cut-off."),
    ])
    figure(doc, FIG / "report_permutation_importance.png", "Figure 8. Equipment, distance and weight drive the price, and city identity adds a regional premium. Calendar, direction and missing-value flags contribute almost nothing.")

    # ------------------------------------------------------------------ 13 predictions + chart
    doc.add_heading("13. Final predictions and the December chart", level=1)
    vp = summ["validation_pred"]
    table(doc, ["Check on validation_predictions.csv", "Result"], [
        ["Rows, columns, ID order", "12,000 rows; load_id,predicted_rate; identical to the template"],
        ["Missing, duplicate, non-finite or non-positive", "None"],
        ["Range", f"${vp['min']:,.0f} to ${vp['max']:,.0f}"],
        ["Median prediction", f"${vp['median']:,.0f} (training labels ${summ['train_label']['median']:,.0f})"],
        ["Median rate per mile", f"${summ['validation_pred_rpm']['median']:.3f} (training ${summ['train_label_rpm']['median']:.3f})"],
    ], [7.4, 8.8], font_size=9)
    para(doc, "score.py was run exactly as the README instructs:")
    code(doc, """$ python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\\candidate_december.png
Final validation metrics are calculated by Spotter after submission.""")
    figure(doc, ROOT / "scorer_results" / "candidate_december.png", "Figure 9. The official December chart, produced by score.py without modification.")
    bullets(doc, [
        ("Level:", "predictions run from $825 to $845, about $2.32 per mile. The lane's 21 clean Dry Van loads in training have a median of $2.23 per mile at a median 26,829 lb. The chart load is heavier at 32,000 lb and later in the year's upward drift, which fits the higher level."),
        ("Weekly wave:", "peaks fall on Thursdays and dips on Sundays. They come from validation.csv's December market_index, which cycles weekly from about 0.83 to 1.04."),
        ("No holiday dip:", "training showed no consistent holiday effect, so the model applies none on Christmas Day."),
        ("Inputs:", "the chart rows carry no market_index or coordinates. Coordinates come from the training city table and each day's market_index is that day's mean in validation.csv, an unlabelled feature."),
    ])

    # ------------------------------------------------------------------ 14 limitations
    doc.add_heading("14. Limitations and next steps", level=1)
    bullets(doc, [
        ("Unknown metric.", "The model optimises MAE. If Spotter scores with RMSE, the corrupted labels will dominate any model's score."),
        ("Untested holiday season.", "Training has no late-December data, so Christmas and New Year behaviour cannot be validated."),
        ("Mild selection risk.", "The drift setting and three-month window were chosen on the same four folds that report their accuracy."),
        ("Remaining bias.", "Clean-row predictions run about 1% low on the closest fold. A rolling recalibration on the latest weeks would be the first improvement to try in production."),
        ("New cities.", "About 12% of validation rows touch a city never seen in training and rely on coordinates."),
    ])

    # ------------------------------------------------------------------ 15 reproducibility
    doc.add_heading("15. Reproducing the results", level=1)
    para(doc, "The repository README lists every command. src/predict.py defaults to the final configuration and writes both files where score.py expects them, so three commands reproduce the submission. "
              "Repeated runs give identical predictions. Setting XGB_DEVICE=cpu runs without a GPU and gives the same backtest result.")
    code(doc, """python -m pip install -r requirements.txt
python src/predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv""")
    para(doc, "Analysis scripts (src/eda.py, leakage_audit.py, drift_analysis.py, temporal_analysis.py, random_split_diagnostic.py, evaluate.py, importance.py) write their full outputs to outputs/ and charts to reports/figures/. "
              "src/experiments.py reruns any backtest configuration and appends to the experiment log.", size=10)

    doc.save(DOCX)
    print("wrote", DOCX)


if __name__ == "__main__":
    main()
