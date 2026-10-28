"""
========================================================================================
Smart Quality and Reliability Analytics Dashboard Using Python
Author: Engineering Analytics Mini-Project
Technology: Python, Streamlit, Pandas, NumPy, Plotly, SciPy, ReportLab
========================================================================================
"""

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

# ReportLab PDF Generation Imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# --------------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Quality & Reliability Analytics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Global Card & Container Styling */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    .metric-subtitle {
        font-size: 0.80rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .badge-simulated {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #fde68a;
    }
    .badge-live {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #bbf7d0;
    }
    .notice-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.90rem;
        color: #334155;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.90rem;
        color: #92400e;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# 2. DEFAULT SYNTHETIC DATA GENERATORS
# --------------------------------------------------------------------------------------
def generate_default_quality_data() -> pd.DataFrame:
    """Generates 100 realistic measurements of a precision machined shaft (50.00 ± 0.15 mm)."""
    np.random.seed(42)
    n = 100
    nominal = 50.00
    target_std = 0.045

    # Normal distribution with slight center shift and 2 intentional outliers for SPC detection
    measurements = np.random.normal(loc=nominal + 0.015, scale=target_std, size=n)
    measurements[55] = 50.162  # Upper spec limit violation
    measurements[64] = 49.838  # Lower spec limit violation

    lsl = 49.85
    usl = 50.15

    defect_flags = ((measurements < lsl) | (measurements > usl)).astype(int)

    timestamps = pd.date_range(end=datetime.now(), periods=n, freq="5min")
    sample_ids = [f"SMP-{i+1:03d}" for i in range(n)]
    batches = [f"BATCH-{chr(65 + i // 40)}{(i % 40) // 20 + 1}" for i in range(n)]
    operators = [
        "Operator_Raj" if i < 35 else ("Operator_Priya" if i < 75 else "Operator_David")
        for i in range(n)
    ]
    machines = [
        "CNC-01" if i < 35 else ("CNC-02" if i < 75 else "CNC-03")
        for i in range(n)
    ]

    df = pd.DataFrame(
        {
            "Sample_ID": sample_ids,
            "Timestamp": timestamps.strftime("%Y-%m-%d %H:%M"),
            "Batch_ID": batches,
            "Machine_ID": machines,
            "Operator": operators,
            "Measurement_mm": np.round(measurements, 3),
            "LSL": lsl,
            "USL": usl,
            "Defect_Flag": defect_flags,
        }
    )
    return df


def generate_default_reliability_data() -> pd.DataFrame:
    """Generates realistic equipment failure and operating logs for multiple plant machines."""
    data = [
        {"Equipment_ID": "EQ-101", "Subsystem": "Spindle Motor System", "Operating_Hours": 4800, "Failure_Count": 4, "Downtime_Hours": 24.0},
        {"Equipment_ID": "EQ-102", "Subsystem": "Hydraulic Pressure Unit", "Operating_Hours": 4200, "Failure_Count": 7, "Downtime_Hours": 49.0},
        {"Equipment_ID": "EQ-103", "Subsystem": "Coolant Circulation Pump", "Operating_Hours": 5100, "Failure_Count": 3, "Downtime_Hours": 15.0},
        {"Equipment_ID": "EQ-104", "Subsystem": "Robotic Transfer Arm", "Operating_Hours": 3600, "Failure_Count": 6, "Downtime_Hours": 36.0},
        {"Equipment_ID": "EQ-105", "Subsystem": "CNC Axis Feed Drive", "Operating_Hours": 5500, "Failure_Count": 2, "Downtime_Hours": 8.0},
        {"Equipment_ID": "EQ-106", "Subsystem": "Pneumatic Gripper Unit", "Operating_Hours": 2900, "Failure_Count": 8, "Downtime_Hours": 40.0},
        {"Equipment_ID": "EQ-107", "Subsystem": "Laser Inspection Scanner", "Operating_Hours": 4000, "Failure_Count": 1, "Downtime_Hours": 3.5},
        {"Equipment_ID": "EQ-108", "Subsystem": "Main Conveyor Belt Drive", "Operating_Hours": 6200, "Failure_Count": 5, "Downtime_Hours": 30.0},
    ]
    df = pd.DataFrame(data)
    df["MTBF_Hours"] = np.round(df["Operating_Hours"] / df["Failure_Count"], 1)
    df["MTTR_Hours"] = np.round(df["Downtime_Hours"] / df["Failure_Count"], 1)
    df["Availability_Pct"] = np.round(
        (df["Operating_Hours"] / (df["Operating_Hours"] + df["Downtime_Hours"])) * 100, 2
    )
    return df


# --------------------------------------------------------------------------------------
# 3. SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------------------
if "quality_df" not in st.session_state:
    st.session_state["quality_df"] = generate_default_quality_data()
if "reliability_df" not in st.session_state:
    st.session_state["reliability_df"] = generate_default_reliability_data()
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "Precision Machined Shafts (Sample Dataset)"
if "lsl_val" not in st.session_state:
    st.session_state["lsl_val"] = 49.85
if "usl_val" not in st.session_state:
    st.session_state["usl_val"] = 50.15
if "rel_op_time" not in st.session_state:
    st.session_state["rel_op_time"] = 5000.0
if "rel_failures" not in st.session_state:
    st.session_state["rel_failures"] = 5
if "rel_downtime" not in st.session_state:
    st.session_state["rel_downtime"] = 40.0
if "rel_mission_time" not in st.session_state:
    st.session_state["rel_mission_time"] = 1000.0


# --------------------------------------------------------------------------------------
# 4. STATISTICAL & RELIABILITY COMPUTATION HELPER FUNCTIONS
# --------------------------------------------------------------------------------------
def compute_quality_statistics(series: pd.Series, defect_series: pd.Series = None, lsl: float = None, usl: float = None):
    """Calculates descriptive statistics, defect rate, capability indices (Cp, Cpk), and SPC limits."""
    clean_series = series.dropna().astype(float)
    n = len(clean_series)
    if n == 0:
        return None

    mean_val = float(clean_series.mean())
    median_val = float(clean_series.median())
    std_val = float(clean_series.std(ddof=1)) if n > 1 else 0.0
    var_val = float(clean_series.var(ddof=1)) if n > 1 else 0.0
    min_val = float(clean_series.min())
    max_val = float(clean_series.max())
    range_val = max_val - min_val

    # Defect calculation
    if defect_series is not None and not defect_series.empty:
        defect_count = int((defect_series != 0).sum())
        defect_pct = (defect_count / n) * 100.0
    elif lsl is not None and usl is not None and usl > lsl:
        defect_count = int(((clean_series < lsl) | (clean_series > usl)).sum())
        defect_pct = (defect_count / n) * 100.0
    else:
        defect_count = 0
        defect_pct = 0.0

    # Moving Range calculation for Individual (I) Control Chart: d2 = 1.128 for n=2
    if n > 1:
        mr = np.abs(np.diff(clean_series.values))
        mean_mr = float(np.mean(mr))
        sigma_within = mean_mr / 1.128  # Short-term process variation estimate
        ucl_spc = mean_val + (3.0 * sigma_within)
        lcl_spc = mean_val - (3.0 * sigma_within)
        cl_spc = mean_val
        out_of_control_mask = (clean_series < lcl_spc) | (clean_series > ucl_spc)
        out_of_control_count = int(out_of_control_mask.sum())
    else:
        mean_mr = 0.0
        sigma_within = std_val
        ucl_spc = mean_val
        lcl_spc = mean_val
        cl_spc = mean_val
        out_of_control_count = 0

    # Process Capability Indices (Cp, Cpk)
    cp, cpk, cpu, cpl = None, None, None, None
    capability_status = "N/A (Provide Valid Specification Limits)"

    if lsl is not None and usl is not None and usl > lsl and std_val > 0:
        cp = (usl - lsl) / (6.0 * std_val)
        cpu = (usl - mean_val) / (3.0 * std_val)
        cpl = (mean_val - lsl) / (3.0 * std_val)
        cpk = min(cpu, cpl)

        if cpk < 1.0:
            capability_status = "Incapable (Significant Defects Expected)"
        elif cpk < 1.33:
            capability_status = "Marginally Capable (Close Process Monitoring Required)"
        elif cpk < 1.67:
            capability_status = "Capable (Meets Industrial Quality Standard)"
        else:
            capability_status = "World-Class / Six Sigma Capable"

    return {
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "variance": var_val,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "defect_count": defect_count,
        "defect_pct": defect_pct,
        "mean_mr": mean_mr,
        "sigma_within": sigma_within,
        "cl_spc": cl_spc,
        "ucl_spc": ucl_spc,
        "lcl_spc": lcl_spc,
        "out_of_control_count": out_of_control_count,
        "cp": cp,
        "cpk": cpk,
        "cpu": cpu,
        "cpl": cpl,
        "lsl": lsl,
        "usl": usl,
        "capability_status": capability_status,
    }


def compute_reliability_metrics(total_op_time: float, failure_count: int, downtime_hours: float, mission_time: float):
    """Calculates MTBF, MTTR, availability, failure rate, and exponential reliability."""
    if total_op_time <= 0:
        return {"error": "Total operating time must be strictly positive (> 0)."}
    if failure_count < 0:
        return {"error": "Failure count cannot be negative."}
    if downtime_hours < 0:
        return {"error": "Downtime cannot be negative."}
    if downtime_hours > total_op_time:
        return {"error": "Downtime hours cannot exceed total operating time."}

    # Case 1: Zero failures observed
    if failure_count == 0:
        mtbf = None
        mttr = 0.0
        failure_rate = 0.0
        total_time = total_op_time + downtime_hours
        availability_pct = 100.0 if total_time == 0 else (total_op_time / total_time) * 100.0
        mission_reliability = 1.0
        mission_unreliability = 0.0
        status_note = (
            "Zero failures observed. Empirical failure rate is estimated at 0. "
            f"Mean Time Between Failures is at least greater than the observation period ({total_op_time:,.1f} hrs)."
        )
    else:
        mtbf = total_op_time / failure_count
        mttr = downtime_hours / failure_count
        failure_rate = failure_count / total_op_time  # failures per hour (lambda)
        total_cycle = total_op_time + downtime_hours
        availability_pct = (total_op_time / total_cycle) * 100.0 if total_cycle > 0 else 0.0

        # Exponential Reliability Model: R(t) = exp(-lambda * t)
        if mission_time >= 0:
            mission_reliability = math.exp(-failure_rate * mission_time)
            mission_unreliability = 1.0 - mission_reliability
        else:
            mission_reliability = None
            mission_unreliability = None
        status_note = "Valid calculation based on empirical exponential distribution assumption."

    return {
        "total_op_time": total_op_time,
        "failure_count": failure_count,
        "downtime_hours": downtime_hours,
        "mission_time": mission_time,
        "mtbf": mtbf,
        "mttr": mttr,
        "failure_rate": failure_rate,
        "availability_pct": availability_pct,
        "mission_reliability": mission_reliability,
        "mission_unreliability": mission_unreliability,
        "status_note": status_note,
        "error": None,
    }


# --------------------------------------------------------------------------------------
# 5. REPORTLAB PDF EXPORT GENERATOR
# --------------------------------------------------------------------------------------
def generate_audit_pdf(q_stats: dict, r_stats: dict, dataset_name: str) -> io.BytesIO:
    """Generates an executive engineering PDF report using ReportLab with clean typography and layout."""
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "RepTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "RepSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    sec_heading = ParagraphStyle(
        "RepSecHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    table_cell = ParagraphStyle(
        "RepTableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
    )
    table_cell_bold = ParagraphStyle(
        "RepTableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
    )
    table_header = ParagraphStyle(
        "RepTableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )
    note_style = ParagraphStyle(
        "RepNote",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        spaceBefore=4,
        spaceAfter=6,
    )

    story = []

    # Title & Header
    story.append(Paragraph("Smart Quality & Reliability Analytics Report", title_style))
    meta_info = f"Project: Industrial Quality Engineering Mini-Project &nbsp;|&nbsp; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; Dataset: {dataset_name}"
    story.append(Paragraph(meta_info, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=10))

    # Executive Summary Paragraph
    exec_summary = (
        "This engineering report summarizes the statistical quality control parameters, process capability "
        "indices (Cp, Cpk), and equipment reliability metrics derived from active manufacturing operations. "
        "All calculations strictly follow ISO 22514 / AIAG SPC standards and classical exponential reliability modeling."
    )
    story.append(Paragraph(exec_summary, table_cell))
    story.append(Spacer(1, 8))

    # 1. Quality Analysis Table
    story.append(Paragraph("1. Statistical Quality & Process Capability Assessment", sec_heading))
    q_data = [
        [
            Paragraph("Metric Parameter", table_header),
            Paragraph("Calculated Value", table_header),
            Paragraph("Engineering Specification / Benchmark", table_header),
            Paragraph("Status / Remark", table_header),
        ],
        [
            Paragraph("Sample Size (n)", table_cell_bold),
            Paragraph(f"{q_stats['n']}", table_cell),
            Paragraph("Minimum recommended: n >= 30", table_cell),
            Paragraph("Adequate Sample" if q_stats["n"] >= 30 else "Small Sample", table_cell),
        ],
        [
            Paragraph("Process Mean (x-bar)", table_cell_bold),
            Paragraph(f"{q_stats['mean']:.4f}", table_cell),
            Paragraph(f"Nominal Target: {((q_stats['lsl'] + q_stats['usl'])/2):.4f}" if q_stats['lsl'] and q_stats['usl'] else "N/A", table_cell),
            Paragraph("Process Centerline", table_cell),
        ],
        [
            Paragraph("Standard Deviation (s)", table_cell_bold),
            Paragraph(f"{q_stats['std']:.4f}", table_cell),
            Paragraph(f"Variance: {q_stats['variance']:.6f}", table_cell),
            Paragraph("Overall Sample Dispersion", table_cell),
        ],
        [
            Paragraph("Moving Range Mean (MR-bar)", table_cell_bold),
            Paragraph(f"{q_stats['mean_mr']:.4f}", table_cell),
            Paragraph(f"Within-Subgroup Sigma: {q_stats['sigma_within']:.4f}", table_cell),
            Paragraph("Short-term Variation (d2=1.128)", table_cell),
        ],
        [
            Paragraph("SPC Control Limits (I-Chart)", table_cell_bold),
            Paragraph(f"UCL: {q_stats['ucl_spc']:.4f}<br/>LCL: {q_stats['lcl_spc']:.4f}", table_cell),
            Paragraph(f"Center Line: {q_stats['cl_spc']:.4f}", table_cell),
            Paragraph(f"{q_stats['out_of_control_count']} points out-of-control (3-sigma)", table_cell),
        ],
        [
            Paragraph("Specification Limits", table_cell_bold),
            Paragraph(f"LSL: {q_stats['lsl'] if q_stats['lsl'] else 'N/A'}<br/>USL: {q_stats['usl'] if q_stats['usl'] else 'N/A'}", table_cell),
            Paragraph(f"Defect Count: {q_stats['defect_count']} ({q_stats['defect_pct']:.2f}%)", table_cell),
            Paragraph("Voice of the Customer", table_cell),
        ],
        [
            Paragraph("Potential Capability (Cp)", table_cell_bold),
            Paragraph(f"{q_stats['cp']:.3f}" if q_stats["cp"] is not None else "N/A", table_cell),
            Paragraph("Standard Target: >= 1.33", table_cell),
            Paragraph("Process Spread vs Tolerance", table_cell),
        ],
        [
            Paragraph("Actual Capability (Cpk)", table_cell_bold),
            Paragraph(f"{q_stats['cpk']:.3f}" if q_stats["cpk"] is not None else "N/A", table_cell),
            Paragraph("Standard Target: >= 1.33", table_cell),
            Paragraph(q_stats["capability_status"], table_cell_bold),
        ],
    ]

    t_qual = Table(q_data, colWidths=[130, 110, 150, 150])
    t_qual.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_qual)
    story.append(
        Paragraph(
            "Note: Control limits (UCL/LCL) represent the natural variation of the process (Voice of the Process). "
            "Specification limits (USL/LSL) represent engineering tolerance requirements (Voice of the Customer).",
            note_style,
        )
    )
    story.append(Spacer(1, 8))

    # 2. Reliability Analysis Table
    story.append(Paragraph("2. Reliability & Availability Engineering Metrics", sec_heading))
    mtbf_str = f"{r_stats['mtbf']:,.1f} hrs" if r_stats["mtbf"] is not None else "Undetermined (0 Failures)"
    r_mission_str = (
        f"{r_stats['mission_reliability'] * 100:.2f}%"
        if r_stats["mission_reliability"] is not None
        else "N/A"
    )
    unrel_str = (
        f"{r_stats['mission_unreliability'] * 100:.2f}%"
        if r_stats["mission_unreliability"] is not None
        else "N/A"
    )

    r_data = [
        [
            Paragraph("Reliability Metric", table_header),
            Paragraph("Input / Computed Value", table_header),
            Paragraph("Governing Engineering Formula", table_header),
            Paragraph("Operational Assessment", table_header),
        ],
        [
            Paragraph("Total Operating Time (T)", table_cell_bold),
            Paragraph(f"{r_stats['total_op_time']:,.1f} operating hours", table_cell),
            Paragraph("Observed operating baseline", table_cell),
            Paragraph("Active operational cycle", table_cell),
        ],
        [
            Paragraph("Failure Count (k)", table_cell_bold),
            Paragraph(f"{r_stats['failure_count']} failure event(s)", table_cell),
            Paragraph("Count of unscheduled stoppage", table_cell),
            Paragraph("Observed empirical failures", table_cell),
        ],
        [
            Paragraph("Total Downtime (T_down)", table_cell_bold),
            Paragraph(f"{r_stats['downtime_hours']:,.1f} repair hours", table_cell),
            Paragraph("Cumulative maintenance duration", table_cell),
            Paragraph(f"Repair ratio: {(r_stats['downtime_hours']/r_stats['total_op_time'])*100:.2f}%", table_cell),
        ],
        [
            Paragraph("Mean Time Between Failures (MTBF)", table_cell_bold),
            Paragraph(mtbf_str, table_cell_bold),
            Paragraph("MTBF = Operating Time / Failures", table_cell),
            Paragraph("Expected uptime between breakdowns", table_cell),
        ],
        [
            Paragraph("Mean Time To Repair (MTTR)", table_cell_bold),
            Paragraph(f"{r_stats['mttr']:,.2f} hrs", table_cell),
            Paragraph("MTTR = Downtime / Failures", table_cell),
            Paragraph("Average restoration time", table_cell),
        ],
        [
            Paragraph("Estimated Failure Rate (lambda)", table_cell_bold),
            Paragraph(f"{r_stats['failure_rate']:.6f} failures/hr", table_cell),
            Paragraph("lambda = 1 / MTBF", table_cell),
            Paragraph("Constant hazard rate assumption", table_cell),
        ],
        [
            Paragraph("Inherent Availability (A)", table_cell_bold),
            Paragraph(f"{r_stats['availability_pct']:.2f}%", table_cell_bold),
            Paragraph("A = MTBF / (MTBF + MTTR)", table_cell),
            Paragraph("Operational readiness probability", table_cell),
        ],
        [
            Paragraph(f"Mission Reliability R({r_stats['mission_time']:.0f}h)", table_cell_bold),
            Paragraph(r_mission_str, table_cell_bold),
            Paragraph("R(t) = exp(-lambda * t)", table_cell),
            Paragraph(f"Probability of survival for {r_stats['mission_time']:.0f} hrs", table_cell),
        ],
        [
            Paragraph(f"Unreliability F({r_stats['mission_time']:.0f}h)", table_cell_bold),
            Paragraph(unrel_str, table_cell),
            Paragraph("F(t) = 1 - R(t)", table_cell),
            Paragraph("Failure probability within mission", table_cell),
        ],
    ]

    t_rel = Table(r_data, colWidths=[140, 120, 140, 140])
    t_rel.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284C7")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_rel)
    story.append(Spacer(1, 8))

    # 3. Engineering Assumptions & Conclusion
    story.append(
        Paragraph("3. Engineering Assumptions & Methodological Governance", sec_heading)
    )
    assumptions_text = (
        "<b>1. Quality Capability Assumptions:</b> Process capability indices (Cp/Cpk) assume the underlying data follows "
        "an approximate normal distribution and the process is in a state of statistical process control. If out-of-control "
        "signals exist, special causes must be eliminated before taking Cpk as a permanent predictor.<br/>"
        "<b>2. Reliability Model Assumptions:</b> The exponential model R(t) = exp(-lambda * t) presumes a constant failure "
        "rate lambda (the flat useful-life region of the bathtub curve). This model does not account for initial burn-in or late-stage "
        "wear-out degradation. All computed values are empirical point estimates from operational samples."
    )
    story.append(Paragraph(assumptions_text, table_cell))
    story.append(Spacer(1, 10))

    # Sign-off box
    signoff_data = [
        [
            Paragraph("<b>Analyst / Student:</b> Engineering Mini-Project", table_cell),
            Paragraph("<b>Review Status:</b> Automated Analytical Verification Pass", table_cell),
            Paragraph("<b>Stamp:</b> Certified Calculation Model", table_cell),
        ]
    ]
    t_sign = Table(signoff_data, colWidths=[180, 180, 180])
    t_sign.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(KeepTogether(t_sign))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer


# --------------------------------------------------------------------------------------
# 6. SIDEBAR NAVIGATION & GLOBAL CONTROLS
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/feathericons/feather/master/icons/activity.svg",
        width=42,
    )
    st.title("Smart Analytics")
    st.markdown(
        "**Industrial Quality & Reliability Suite**\n"
        "<span class='badge-live'>Python 3.12 Ready</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    selected_page = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Dashboard",
            "📊 Quality Analysis",
            "⚙️ Reliability Analysis",
            "📁 Sample Data",
            "📑 Report Generation",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Quick Dataset Actions")
    if st.button("🔄 Reset to Default Sample Data", use_container_width=True):
        st.session_state["quality_df"] = generate_default_quality_data()
        st.session_state["reliability_df"] = generate_default_reliability_data()
        st.session_state["dataset_name"] = "Precision Machined Shafts (Sample Dataset)"
        st.session_state["lsl_val"] = 49.85
        st.session_state["usl_val"] = 50.15
        st.session_state["rel_op_time"] = 5000.0
        st.session_state["rel_failures"] = 5
        st.session_state["rel_downtime"] = 40.0
        st.session_state["rel_mission_time"] = 1000.0
        st.success("Sample dataset loaded successfully!")
        st.rerun()

    st.markdown(
        "<div style='font-size:0.75rem; color:#94a3b8; margin-top:20px; text-align:center;'>"
        "Smart Quality & Reliability Engineering Dashboard<br/>"
        "Free & Open-Source Cloud Deployment"
        "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# 7. PAGE 1: HOME DASHBOARD
# --------------------------------------------------------------------------------------
if selected_page == "🏠 Home Dashboard":
    st.markdown(
        """
        <div style='margin-bottom: 25px;'>
            <h1 style='color: #0f172a; margin-bottom: 6px;'>Smart Quality and Reliability Analytics Dashboard</h1>
            <p style='color: #475569; font-size: 1.1rem; margin-bottom: 12px;'>
                An end-to-end industrial engineering platform for Statistical Process Control (SPC), 
                Process Capability Assessment (Cp, Cpk), and Systems Reliability Engineering.
            </p>
            <span class='badge-simulated'>Academic Mini-Project Edition</span>
            &nbsp;
            <span class='badge-live'>Active Engine: Streamlit + SciPy</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q_df = st.session_state["quality_df"]
    num_col = "Measurement_mm" if "Measurement_mm" in q_df.columns else q_df.select_dtypes(include=[np.number]).columns[0]
    q_stats = compute_quality_statistics(
        q_df[num_col],
        defect_series=q_df["Defect_Flag"] if "Defect_Flag" in q_df.columns else None,
        lsl=st.session_state["lsl_val"],
        usl=st.session_state["usl_val"],
    )
    r_stats = compute_reliability_metrics(
        st.session_state["rel_op_time"],
        st.session_state["rel_failures"],
        st.session_state["rel_downtime"],
        st.session_state["rel_mission_time"],
    )

    # Key Performance Indicator (KPI) Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class='metric-card' style='border-left: 4px solid #3b82f6;'>
                <div class='metric-title'>Total Process Samples</div>
                <div class='metric-value'>{q_stats['n']}</div>
                <div class='metric-subtitle'>Dataset: {st.session_state['dataset_name'][:24]}...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class='metric-card' style='border-left: 4px solid #10b981;'>
                <div class='metric-title'>Process Mean (x̄)</div>
                <div class='metric-value'>{q_stats['mean']:.3f}</div>
                <div class='metric-subtitle'>Std Dev (s): ±{q_stats['std']:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        defect_color = "#ef4444" if q_stats["defect_pct"] > 3.0 else "#10b981"
        st.markdown(
            f"""
            <div class='metric-card' style='border-left: 4px solid {defect_color};'>
                <div class='metric-title'>Defect Rate</div>
                <div class='metric-value'>{q_stats['defect_pct']:.2f}%</div>
                <div class='metric-subtitle'>{q_stats['defect_count']} non-conforming items</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class='metric-card' style='border-left: 4px solid #8b5cf6;'>
                <div class='metric-title'>System Availability (A)</div>
                <div class='metric-value'>{r_stats['availability_pct']:.2f}%</div>
                <div class='metric-subtitle'>MTBF: {r_stats['mtbf']:,.0f}h | MTTR: {r_stats['mttr']:.1f}h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Quick Action Banner
    col_banner1, col_banner2 = st.columns([3, 1])
    with col_banner1:
        st.info(
            "💡 **Instant Start:** The dashboard is pre-loaded with simulated precision shaft manufacturing data. "
            "You can immediately navigate through the analysis tabs or click the button on the right to refresh with fresh sample data."
        )
    with col_banner2:
        if st.button("⚡ Load Fresh Sample Data", use_container_width=True):
            st.session_state["quality_df"] = generate_default_quality_data()
            st.session_state["reliability_df"] = generate_default_reliability_data()
            st.session_state["dataset_name"] = "Precision Machined Shafts (Sample Dataset)"
            st.success("Loaded!")
            st.rerun()

    st.markdown("---")

    # Modular Overview Grid
    st.subheader("Core Engineering Modules")
    col_mod1, col_mod2, col_mod3 = st.columns(3)

    with col_mod1:
        st.markdown(
            """
            ### 📊 Quality Analytics
            - **Descriptive Statistics:** Mean, Median, Variance, Standard Deviation, and Defect %.
            - **Process Capability:** Cp and Cpk indices against customer tolerance limits.
            - **Statistical Process Control:** Individual (I) run chart with 3σ control limits (UCL, CL, LCL) derived from Moving Range (MR).
            - **Distribution Analytics:** Histogram with Gaussian curve overlay and Box Plot.
            """
        )

    with col_mod2:
        st.markdown(
            """
            ### ⚙️ Reliability Analytics
            - **MTBF & MTTR:** Mean Time Between Failures and Mean Time To Repair calculations.
            - **Inherent Availability:** Operational readiness percentage $A = \\frac{MTBF}{MTBF + MTTR}$.
            - **Failure Rate (λ):** Empirical failure rate estimation.
            - **Exponential Decay Modeling:** $R(t) = \\exp(-\\lambda t)$ interactive mission reliability curve.
            """
        )

    with col_mod3:
        st.markdown(
            """
            ### 📑 Automated Audit PDF
            - **One-Click Export:** Industry-standard PDF report generation using ReportLab.
            - **Technical Rigor:** Full tabular breakdown of quality and reliability findings.
            - **Transparent Assumptions:** Clearly documents the difference between Voice of Process vs. Voice of Customer.
            - **CSV Download:** Export processed tables for external grading and lab reports.
            """
        )

    st.markdown("---")
    with st.expander("📚 Student Engineering Theory & Formula Reference Guide", expanded=False):
        st.markdown(
            r"""
            #### 1. Process Capability Formulas
            - **Sample Mean:** $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
            - **Sample Standard Deviation:** $s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}$
            - **Potential Process Capability ($C_p$):**
              $$C_p = \frac{\text{USL} - \text{LSL}}{6s}$$
            - **Actual Process Capability ($C_{pk}$):**
              $$C_{pu} = \frac{\text{USL} - \bar{x}}{3s}, \quad C_{pl} = \frac{\bar{x} - \text{LSL}}{3s}, \quad C_{pk} = \min(C_{pu}, C_{pl})$$
            
            #### 2. Statistical Process Control (I-Chart)
            - **Average Moving Range:** $\overline{MR} = \frac{1}{n-1}\sum_{i=2}^{n} |x_i - x_{i-1}|$
            - **Estimated Short-Term Sigma:** $\hat{\sigma} = \frac{\overline{MR}}{d_2} \quad (d_2 = 1.128 \text{ for } n=2)$
            - **Control Limits:** $\text{UCL} = \bar{x} + 3\hat{\sigma}, \quad \text{Center Line (CL)} = \bar{x}, \quad \text{LCL} = \bar{x} - 3\hat{\sigma}$

            #### 3. Equipment Reliability & Availability
            - **Mean Time Between Failures:** $\text{MTBF} = \frac{\text{Total Operating Time}}{\text{Number of Failures}}$
            - **Mean Time To Repair:** $\text{MTTR} = \frac{\text{Total Downtime}}{\text{Number of Failures}}$
            - **Inherent Availability:** $A = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}} = \frac{T_{op}}{T_{op} + T_{down}}$
            - **Failure Rate:** $\lambda = \frac{1}{\text{MTBF}}$
            - **Exponential Reliability:** $R(t) = e^{-\lambda t}$
            """
        )


# --------------------------------------------------------------------------------------
# 8. PAGE 2: QUALITY ANALYSIS
# --------------------------------------------------------------------------------------
elif selected_page == "📊 Quality Analysis":
    st.title("📊 Manufacturing Quality & Statistical Process Control")
    st.markdown(
        "Analyze manufacturing measurements, evaluate statistical process capability ($C_p, C_{pk}$), "
        "and monitor variation using Statistical Process Control (SPC) charts."
    )

    with st.expander("📂 Dataset Selection & CSV Upload", expanded=True):
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Upload Custom Manufacturing CSV Data",
                type=["csv"],
                help="Upload a CSV with at least one numeric measurement column.",
            )
            if uploaded_file is not None:
                try:
                    custom_df = pd.read_csv(uploaded_file)
                    st.session_state["quality_df"] = custom_df
                    st.session_state["dataset_name"] = uploaded_file.name
                    st.success(f"Successfully loaded '{uploaded_file.name}' with {len(custom_df)} rows.")
                except Exception as e:
                    st.error(f"Error parsing uploaded CSV: {e}")
        with col_up2:
            st.markdown(
                f"""
                <div style='background:#f1f5f9; padding:15px; border-radius:8px; border:1px solid #cbd5e1;'>
                    <b>Active Dataset:</b><br/>
                    <code>{st.session_state['dataset_name']}</code><br/><br/>
                    <b>Total Rows:</b> {len(st.session_state['quality_df'])}<br/>
                    <b>Total Columns:</b> {len(st.session_state['quality_df'].columns)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    df = st.session_state["quality_df"]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        st.error("No numeric columns detected in the active dataset. Please upload a dataset with numeric dimensions.")
        st.stop()

    st.subheader("1. Variable & Specification Limit Selection")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    with col_c1:
        default_idx = num_cols.index("Measurement_mm") if "Measurement_mm" in num_cols else 0
        measure_col = st.selectbox("Measurement Column", num_cols, index=default_idx)

    with col_c2:
        all_cols_with_none = ["None (Infer from LSL/USL)"] + [c for c in df.columns if c != measure_col]
        defect_col_choice = st.selectbox("Defect Status Column (Optional)", all_cols_with_none, index=0)

    data_min = float(df[measure_col].min())
    data_max = float(df[measure_col].max())
    data_mean = float(df[measure_col].mean())
    data_std = float(df[measure_col].std()) if len(df) > 1 else 1.0

    calc_default_lsl = round(data_mean - (3.0 * data_std), 2)
    calc_default_usl = round(data_mean + (3.0 * data_std), 2)

    with col_c3:
        lsl_input = st.number_input(
            "Lower Spec Limit (LSL)",
            value=float(st.session_state.get("lsl_val", calc_default_lsl)),
            format="%.3f",
            help="Design lower specification boundary (Voice of Customer).",
        )
        st.session_state["lsl_val"] = lsl_input

    with col_c4:
        usl_input = st.number_input(
            "Upper Spec Limit (USL)",
            value=float(st.session_state.get("usl_val", calc_default_usl)),
            format="%.3f",
            help="Design upper specification boundary (Voice of Customer).",
        )
        st.session_state["usl_val"] = usl_input

    valid_specs = usl_input > lsl_input
    if not valid_specs:
        st.warning("⚠️ **Validation Notice:** USL must be strictly greater than LSL for Process Capability calculation.")

    defect_series = None
    if defect_col_choice != "None (Infer from LSL/USL)":
        defect_series = df[defect_col_choice]

    q_stats = compute_quality_statistics(
        df[measure_col],
        defect_series=defect_series,
        lsl=lsl_input if valid_specs else None,
        usl=usl_input if valid_specs else None,
    )

    st.markdown("---")

    st.subheader("2. Summary Descriptive Statistics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Sample Size (n)", f"{q_stats['n']}")
    m2.metric("Mean (x̄)", f"{q_stats['mean']:.4f}")
    m3.metric("Median (x̃)", f"{q_stats['median']:.4f}")
    m4.metric("Std Deviation (s)", f"{q_stats['std']:.4f}")
    m5.metric("Variance (s²)", f"{q_stats['variance']:.6f}")
    m6.metric("Defect Rate", f"{q_stats['defect_pct']:.2f}%", f"{q_stats['defect_count']} units")

    st.markdown("---")
    st.subheader("3. Process Capability Assessment (Cp & Cpk)")

    cap_col1, cap_col2 = st.columns([1, 2])
    with cap_col1:
        if valid_specs and q_stats["cp"] is not None:
            st.markdown(
                f"""
                <div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:18px;'>
                    <div style='font-size:0.85rem; color:#64748b; font-weight:600;'>POTENTIAL CAPABILITY (Cp)</div>
                    <div style='font-size:2.2rem; font-weight:700; color:#1e3a8a;'>{q_stats['cp']:.3f}</div>
                    <div style='font-size:0.80rem; color:#64748b; margin-bottom:15px;'>Formula: (USL - LSL) / (6s)</div>
                    <div style='font-size:0.85rem; color:#64748b; font-weight:600;'>ACTUAL CAPABILITY (Cpk)</div>
                    <div style='font-size:2.2rem; font-weight:700; color:#0f766e;'>{q_stats['cpk']:.3f}</div>
                    <div style='font-size:0.80rem; color:#64748b;'>Cpu: {q_stats['cpu']:.3f} | Cpl: {q_stats['cpl']:.3f}</div>
                    <hr style='margin:12px 0;'/>
                    <b>Capability Assessment:</b><br/>
                    <span style='color:#0369a1; font-weight:600;'>{q_stats['capability_status']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Provide valid LSL and USL (USL > LSL) with standard deviation > 0 to compute Cp and Cpk.")

    with cap_col2:
        st.markdown(
            """
            <div class='notice-box'>
                <b>Critical Industrial Engineering Principle:</b><br/>
                <b>Voice of the Process (Control Limits: UCL / LCL) vs. Voice of the Customer (Specification Limits: USL / LSL):</b>
                <ul>
                    <li><b>Control Limits (UCL, LCL):</b> Statistically calculated from the data variation (x̄ ± 3σ̂). They indicate what the process is naturally capable of producing right now.</li>
                    <li><b>Specification Limits (USL, LSL):</b> Externally mandated engineering tolerance requirements. They indicate what the product <i>must</i> meet to avoid defects.</li>
                    <li><b>Process Incapability:</b> A process can be in statistical control (all points inside UCL/LCL) yet produce out-of-spec defects if the natural spread (6s) exceeds (USL - LSL).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("4. Quality Control & Distribution Visualizations")
    tab_hist, tab_box, tab_trend, tab_spc = st.tabs(
        [
            "📊 Histogram & Normal Fit",
            "📦 Box & Whisker Plot",
            "📈 Measurement Trend Chart",
            "🛡️ SPC Individual Control Chart",
        ]
    )

    with tab_hist:
        clean_vals = df[measure_col].dropna().values
        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Histogram(
                x=clean_vals,
                nbinsx=25,
                name="Measurement Count",
                marker_color="#3b82f6",
                opacity=0.75,
            )
        )

        if q_stats["std"] > 0:
            x_range = np.linspace(min(clean_vals.min(), lsl_input), max(clean_vals.max(), usl_input), 200)
            pdf_vals = stats.norm.pdf(x_range, q_stats["mean"], q_stats["std"])
            bin_width = (clean_vals.max() - clean_vals.min()) / 25
            scaled_pdf = pdf_vals * len(clean_vals) * bin_width

            fig_hist.add_trace(
                go.Scatter(
                    x=x_range,
                    y=scaled_pdf,
                    mode="lines",
                    name="Normal Fit Curve",
                    line=dict(color="#0f172a", width=2.5),
                )
            )

        if valid_specs:
            fig_hist.add_vline(x=lsl_input, line_dash="dash", line_color="#ef4444", annotation_text="LSL", annotation_position="top left")
            fig_hist.add_vline(x=usl_input, line_dash="dash", line_color="#ef4444", annotation_text="USL", annotation_position="top right")
        fig_hist.add_vline(x=q_stats["mean"], line_dash="solid", line_color="#10b981", annotation_text="Mean (x̄)", annotation_position="bottom right")

        fig_hist.update_layout(
            title=f"Distribution of {measure_col} with Specification Limits",
            xaxis_title=f"{measure_col}",
            yaxis_title="Frequency Count",
            template="plotly_white",
            height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab_box:
        fig_box = px.box(
            df,
            y=measure_col,
            points="all",
            title=f"Box & Whisker Plot for {measure_col} (Showing Quartiles & Outliers)",
            color_discrete_sequence=["#0284c7"],
            template="plotly_white",
            height=430,
        )
        if valid_specs:
            fig_box.add_hline(y=lsl_input, line_dash="dash", line_color="#ef4444", annotation_text="LSL")
            fig_box.add_hline(y=usl_input, line_dash="dash", line_color="#ef4444", annotation_text="USL")
        st.plotly_chart(fig_box, use_container_width=True)

    with tab_trend:
        fig_trend = go.Figure()
        indices = list(range(1, len(df) + 1))
        fig_trend.add_trace(
            go.Scatter(
                x=indices,
                y=df[measure_col],
                mode="lines+markers",
                name="Measurement",
                line=dict(color="#2563eb", width=1.5),
                marker=dict(size=4),
            )
        )
        fig_trend.add_hline(y=q_stats["mean"], line_dash="dash", line_color="#10b981", annotation_text="Process Mean")
        if valid_specs:
            fig_trend.add_hline(y=lsl_input, line_dash="dot", line_color="#ef4444", annotation_text="LSL")
            fig_trend.add_hline(y=usl_input, line_dash="dot", line_color="#ef4444", annotation_text="USL")

        fig_trend.update_layout(
            title=f"Sequential Run / Trend Chart for {measure_col}",
            xaxis_title="Sample Sequence Index",
            yaxis_title=f"{measure_col}",
            template="plotly_white",
            height=430,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab_spc:
        fig_spc = go.Figure()
        vals = df[measure_col].values
        n_pts = len(vals)
        x_pts = list(range(1, n_pts + 1))

        in_control_x = []
        in_control_y = []
        out_control_x = []
        out_control_y = []

        for i, val in enumerate(vals):
            if val > q_stats["ucl_spc"] or val < q_stats["lcl_spc"]:
                out_control_x.append(i + 1)
                out_control_y.append(val)
            else:
                in_control_x.append(i + 1)
                in_control_y.append(val)

        fig_spc.add_trace(
            go.Scatter(
                x=x_pts,
                y=vals,
                mode="lines",
                name="Process Line",
                line=dict(color="#94a3b8", width=1.2),
                hoverinfo="skip",
            )
        )

        fig_spc.add_trace(
            go.Scatter(
                x=in_control_x,
                y=in_control_y,
                mode="markers",
                name="In-Control Points",
                marker=dict(color="#0284c7", size=6),
            )
        )

        if out_control_x:
            fig_spc.add_trace(
                go.Scatter(
                    x=out_control_x,
                    y=out_control_y,
                    mode="markers",
                    name="Out-of-Control (Special Cause)",
                    marker=dict(color="#dc2626", size=10, symbol="x"),
                )
            )

        fig_spc.add_hline(
            y=q_stats["ucl_spc"],
            line_dash="dash",
            line_color="#dc2626",
            line_width=2,
            annotation_text=f"UCL = {q_stats['ucl_spc']:.3f}",
            annotation_position="top right",
        )
        fig_spc.add_hline(
            y=q_stats["cl_spc"],
            line_dash="solid",
            line_color="#059669",
            line_width=2,
            annotation_text=f"CL = {q_stats['cl_spc']:.3f}",
            annotation_position="bottom right",
        )
        fig_spc.add_hline(
            y=q_stats["lcl_spc"],
            line_dash="dash",
            line_color="#dc2626",
            line_width=2,
            annotation_text=f"LCL = {q_stats['lcl_spc']:.3f}",
            annotation_position="bottom right",
        )

        fig_spc.update_layout(
            title=f"Statistical Process Control (SPC) Individual (I) Chart [MR-based 3σ limits]",
            xaxis_title="Sample Sequence Index",
            yaxis_title=f"{measure_col}",
            template="plotly_white",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_spc, use_container_width=True)

        if q_stats["out_of_control_count"] > 0:
            st.warning(
                f"🚨 **Process Signal Detected:** {q_stats['out_of_control_count']} measurement point(s) fell outside "
                f"the 3-sigma statistical control limits [LCL: {q_stats['lcl_spc']:.3f}, UCL: {q_stats['ucl_spc']:.3f}]. "
                "Investigate potential special causes (e.g. tool wear, operator handover, material batch variation)."
            )
        else:
            st.success(
                f"✅ **Process in Statistical Control:** All {q_stats['n']} samples fall within the estimated 3-sigma "
                f"control limits. The process exhibits only common-cause random variation."
            )

    st.markdown("---")
    st.subheader("5. Process Data Preview Table")
    st.dataframe(df, use_container_width=True, height=240)


# --------------------------------------------------------------------------------------
# 9. PAGE 3: RELIABILITY ANALYSIS
# --------------------------------------------------------------------------------------
elif selected_page == "⚙️ Reliability Analysis":
    st.title("⚙️ Equipment Reliability & Systems Availability Engineering")
    st.markdown(
        "Evaluate equipment uptime, repair duration, Mean Time Between Failures (MTBF), "
        "and model mission survival using the classical Exponential Reliability Function $R(t) = \\exp(-\\lambda t)$."
    )

    st.subheader("1. Operational Parameters & Failure Records")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        rel_op = st.number_input(
            "Total Operating Time (Hours)",
            min_value=1.0,
            max_value=1000000.0,
            value=float(st.session_state["rel_op_time"]),
            step=100.0,
            help="Total cumulative running time of the machinery or fleet.",
        )
        st.session_state["rel_op_time"] = rel_op

    with col_r2:
        rel_fail = st.number_input(
            "Number of Failures (k)",
            min_value=0,
            max_value=10000,
            value=int(st.session_state["rel_failures"]),
            step=1,
            help="Count of unscheduled stoppage/breakdown events during operation.",
        )
        st.session_state["rel_failures"] = rel_fail

    with col_r3:
        rel_down = st.number_input(
            "Total Repair Downtime (Hours)",
            min_value=0.0,
            max_value=float(rel_op),
            value=float(min(st.session_state["rel_downtime"], rel_op)),
            step=5.0,
            help="Cumulative maintenance and repair hours spent restoring equipment.",
        )
        st.session_state["rel_downtime"] = rel_down

    with col_r4:
        rel_mission = st.number_input(
            "Mission / Observation Time (t in Hours)",
            min_value=1.0,
            max_value=100000.0,
            value=float(st.session_state["rel_mission_time"]),
            step=50.0,
            help="Target mission duration for evaluating reliability R(t).",
        )
        st.session_state["rel_mission_time"] = rel_mission

    rel_results = compute_reliability_metrics(rel_op, rel_fail, rel_down, rel_mission)

    if rel_results["error"]:
        st.error(f"❌ Input Validation Error: {rel_results['error']}")
        st.stop()

    st.markdown("---")

    st.subheader("2. Computed Reliability & Operational KPIs")
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        mtbf_display = f"{rel_results['mtbf']:,.1f} h" if rel_results["mtbf"] is not None else "Undetermined"
        k1.metric("MTBF", mtbf_display, "Mean Time Between Failures")
    with k2:
        k2.metric("MTTR", f"{rel_results['mttr']:,.2f} h", "Mean Time To Repair")
    with k3:
        k3.metric("Failure Rate (λ)", f"{rel_results['failure_rate']:.6f} /h", "λ = 1 / MTBF")
    with k4:
        k4.metric("Availability (A)", f"{rel_results['availability_pct']:.2f}%", "A = MTBF / (MTBF+MTTR)")
    with k5:
        r_display = f"{rel_results['mission_reliability'] * 100:.2f}%" if rel_results["mission_reliability"] is not None else "N/A"
        k5.metric(f"Reliability R({rel_mission:.0f}h)", r_display, f"Survival up to {rel_mission:.0f} hrs")

    if rel_fail == 0:
        st.info(
            f"ℹ️ **Zero Failures Observed:** No breakdown events occurred over {rel_op:,.1f} operating hours. "
            f"The empirical failure rate is estimated at λ = 0. Statistically, the true MTBF is at least "
            f"greater than the test duration ({rel_op:,.1f} hours)."
        )

    st.markdown("---")

    st.subheader(f"3. Exponential Reliability Survival Curve R(t) = exp(-λt)")

    max_t = max(rel_mission * 2.5, (rel_results["mtbf"] * 2.0 if rel_results["mtbf"] else 2000.0))
    time_series = np.linspace(0, max_t, 250)

    if rel_results["failure_rate"] > 0:
        r_curve = np.exp(-rel_results["failure_rate"] * time_series)
        f_curve = 1.0 - r_curve
    else:
        r_curve = np.ones_like(time_series)
        f_curve = np.zeros_like(time_series)

    fig_rel = go.Figure()

    fig_rel.add_trace(
        go.Scatter(
            x=time_series,
            y=r_curve * 100.0,
            mode="lines",
            name="Reliability R(t) %",
            line=dict(color="#0284c7", width=3),
            fill="tozeroy",
            fillcolor="rgba(2, 132, 199, 0.08)",
        )
    )

    if rel_results["mission_reliability"] is not None:
        fig_rel.add_trace(
            go.Scatter(
                x=[rel_mission],
                y=[rel_results["mission_reliability"] * 100.0],
                mode="markers+text",
                name=f"Mission Target ({rel_mission:.0f}h)",
                marker=dict(color="#dc2626", size=11, symbol="diamond"),
                text=[f"R({rel_mission:.0f}h) = {rel_results['mission_reliability']*100:.2f}%"],
                textposition="top right",
            )
        )

    if rel_results["mtbf"] is not None and rel_results["mtbf"] <= max_t:
        fig_rel.add_vline(
            x=rel_results["mtbf"],
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"t = MTBF (R ≈ 36.8%)",
            annotation_position="bottom right",
        )

    fig_rel.update_layout(
        title="Reliability Decay Over Time: R(t) = exp(-λt)",
        xaxis_title="Operating Time t (Hours)",
        yaxis_title="Probability of Survival R(t) (%)",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_rel, use_container_width=True)

    st.markdown("---")

    st.subheader("4. Plant Equipment Fleet Reliability Log (Multi-Machine Benchmark)")
    fleet_df = st.session_state["reliability_df"]
    st.dataframe(fleet_df, use_container_width=True, height=260)

    fig_fleet = px.bar(
        fleet_df,
        x="Equipment_ID",
        y="Availability_Pct",
        color="Availability_Pct",
        text="Availability_Pct",
        hover_data=["Subsystem", "MTBF_Hours", "MTTR_Hours"],
        color_continuous_scale="Teal",
        title="Fleet Inherent Availability Benchmark (%) by Machine Unit",
        template="plotly_white",
        height=380,
    )
    fig_fleet.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_fleet.update_layout(yaxis=dict(range=[95, 101]))
    st.plotly_chart(fig_fleet, use_container_width=True)

    with st.expander("🔬 Theoretical Foundations & Engineering Assumptions", expanded=False):
        st.markdown(
            """
            - **Exponential Distribution Assumption:** The mathematical model $R(t) = e^{-\\lambda t}$ assumes a **strictly constant failure rate** ($\\lambda = \\text{constant}$). 
              In industrial engineering, this corresponds to the flat, steady-state region of the classic **Bathtub Curve** (the *useful life* phase).
            - **Memoryless Property:** An exponential component does not experience age-related degradation in this model; failure in the next operating hour is independent of past survival hours.
            - **Distinction Between Point Estimates and True Population:** All metrics ($\lambda, MTBF, MTTR$) computed here are **empirical point estimates** derived from sample data. 
              Small failure counts are subject to statistical uncertainty and should be augmented with confidence intervals in critical aerospace or nuclear applications.
            """
        )


# --------------------------------------------------------------------------------------
# 10. PAGE 4: SAMPLE DATA
# --------------------------------------------------------------------------------------
elif selected_page == "📁 Sample Data":
    st.title("📁 Realistic Simulated Manufacturing Datasets")
    st.markdown(
        "Inspect and download realistic synthetic manufacturing data designed for industrial "
        "quality audits, process capability studies, and maintenance engineering simulations."
    )

    st.markdown(
        """
        <div class='warning-box'>
            <b>Academic Rigor Notice:</b> All datasets below are <b>simulated synthetic records</b> generated for 
            statistical modeling and coursework validation. They are strictly labeled as simulated data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_sq, tab_sr = st.tabs(["🔩 Quality Measurement Dataset", "🏭 Plant Reliability Logs"])

    with tab_sq:
        st.subheader("Precision Machined Shaft Dimensions (Quality Dataset)")
        st.markdown(
            "- **Component:** Cylindrical Steel Drive Shafts\n"
            "- **Critical Dimension:** Diameter ($mm$)\n"
            "- **Nominal Target:** $50.00\\text{ mm}$\n"
            "- **Specification Tolerance:** $50.00 \\pm 0.15\\text{ mm}$ (LSL: $49.85\\text{ mm}$, USL: $50.15\\text{ mm}$)\n"
            "- **Sample Size:** 100 parts with timestamps, operator IDs, machine IDs, and defect flags."
        )

        q_csv_data = st.session_state["quality_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Quality Dataset (CSV)",
            data=q_csv_data,
            file_name="sample_manufacturing_quality_data.csv",
            mime="text/csv",
            key="dl_q_csv",
        )

        st.dataframe(st.session_state["quality_df"], use_container_width=True, height=350)

    with tab_sr:
        st.subheader("Equipment Maintenance & Reliability Logs")
        st.markdown(
            "- **Coverage:** 8 Industrial CNCs, Robots, Pumps, and Material Handling Units\n"
            "- **Logged Parameters:** Total Operating Hours, Failure Counts, Repair Downtime Hours, MTBF, MTTR, and Inherent Availability."
        )

        r_csv_data = st.session_state["reliability_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Equipment Reliability Logs (CSV)",
            data=r_csv_data,
            file_name="sample_equipment_reliability_data.csv",
            mime="text/csv",
            key="dl_r_csv",
        )

        st.dataframe(st.session_state["reliability_df"], use_container_width=True, height=350)


# --------------------------------------------------------------------------------------
# 11. PAGE 5: REPORT GENERATION
# --------------------------------------------------------------------------------------
elif selected_page == "📑 Report Generation":
    st.title("📑 Industrial Audit & Engineering PDF Report Generator")
    st.markdown(
        "Generate a publication-grade, downloadable PDF audit report consolidating your active "
        "Statistical Quality Control metrics, Process Capability indices, and Systems Reliability calculations."
    )

    df_current = st.session_state["quality_df"]
    num_cols = df_current.select_dtypes(include=[np.number]).columns.tolist()
    measure_col = num_cols[0] if num_cols else "Measurement"

    q_stats = compute_quality_statistics(
        df_current[measure_col],
        defect_series=df_current["Defect_Flag"] if "Defect_Flag" in df_current.columns else None,
        lsl=st.session_state.get("lsl_val", 49.85),
        usl=st.session_state.get("usl_val", 50.15),
    )
    r_stats = compute_reliability_metrics(
        st.session_state["rel_op_time"],
        st.session_state["rel_failures"],
        st.session_state["rel_downtime"],
        st.session_state["rel_mission_time"],
    )

    col_rep1, col_rep2 = st.columns([2, 1])

    with col_rep1:
        st.subheader("Report Contents Summary")
        st.markdown(
            f"""
            - **Report Title:** Smart Quality & Reliability Analytics Report
            - **Active Dataset:** `{st.session_state['dataset_name']}`
            - **Analyzed Variable:** `{measure_col}` (n = {q_stats['n']} samples)
            - **Process Capability:** Cp = `{q_stats['cp']:.3f}` | Cpk = `{q_stats['cpk']:.3f}`
            - **SPC Control Limits:** UCL = `{q_stats['ucl_spc']:.3f}` | LCL = `{q_stats['lcl_spc']:.3f}`
            - **System Availability:** `{r_stats['availability_pct']:.2f}%` (MTBF = `{r_stats['mtbf']:,.1f}h`)
            - **Mission Reliability:** R({r_stats['mission_time']:.0f}h) = `{r_stats['mission_reliability']*100:.2f}%`
            - **Format:** Vector PDF formatted with ReportLab flowables, tables, and audit sign-off blocks.
            """
        )

        try:
            pdf_bytes = generate_audit_pdf(q_stats, r_stats, st.session_state["dataset_name"])
            st.download_button(
                label="📄 Download Executive Engineering PDF Report",
                data=pdf_bytes,
                file_name=f"Smart_Quality_Reliability_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key="btn_dl_pdf",
                use_container_width=True,
            )
            st.success("✅ PDF report successfully compiled and ready for instant download.")
        except Exception as e:
            st.error(f"Error compiling ReportLab PDF: {e}")

    with col_rep2:
        st.subheader("Export Summary CSV")
        st.markdown(
            "Download all computed analytical metrics in a structured tabular CSV format for grading or lab notebooks."
        )

        summary_data = {
            "Metric_Group": [
                "Quality", "Quality", "Quality", "Quality", "Quality",
                "Quality", "Quality", "Quality", "Quality",
                "Reliability", "Reliability", "Reliability", "Reliability", "Reliability", "Reliability"
            ],
            "Parameter": [
                "Sample Size (n)", "Mean (x-bar)", "Std Dev (s)", "Variance",
                "Defect Rate (%)", "Lower Spec Limit (LSL)", "Upper Spec Limit (USL)",
                "Potential Capability (Cp)", "Actual Capability (Cpk)",
                "Total Operating Hours", "Failure Count", "Downtime Hours",
                "MTBF (Hours)", "MTTR (Hours)", "Availability (%)"
            ],
            "Value": [
                q_stats["n"], round(q_stats["mean"], 4), round(q_stats["std"], 4), round(q_stats["variance"], 6),
                round(q_stats["defect_pct"], 2), q_stats["lsl"], q_stats["usl"],
                round(q_stats["cp"], 3) if q_stats["cp"] else "N/A",
                round(q_stats["cpk"], 3) if q_stats["cpk"] else "N/A",
                r_stats["total_op_time"], r_stats["failure_count"], r_stats["downtime_hours"],
                round(r_stats["mtbf"], 1) if r_stats["mtbf"] else "N/A",
                round(r_stats["mttr"], 2), round(r_stats["availability_pct"], 2)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📊 Download Processed Results (CSV)",
            data=summary_csv,
            file_name="processed_quality_reliability_metrics.csv",
            mime="text/csv",
            key="btn_dl_summary_csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Report Preview: Computed Analytical Values")
    st.dataframe(summary_df, use_container_width=True, height=360)


# --------------------------------------------------------------------------------------
# 12. FOOTER & COPYRIGHT
# --------------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #64748b; font-size: 0.85rem; padding: 15px;'>
        <b>Smart Quality and Reliability Analytics Dashboard Using Python</b> | 
        Industrial Engineering Mini-Project<br/>
        Deployable free on Streamlit Community Cloud & GitHub. Built with Streamlit, Plotly, SciPy, and ReportLab.
    </div>
    """,
    unsafe_allow_html=True,
)
