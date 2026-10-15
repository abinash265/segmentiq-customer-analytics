"""SegmentIQ — interactive customer intelligence workspace.

Local-first Streamlit frontend built on the project's existing data pipeline.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from src.data_loader import DATA_PATH, FEATURE_COLUMNS, ensure_dataset
from src import preprocessing, clustering, evaluation

# ---------- App configuration and visual system ----------
st.set_page_config(page_title="SegmentIQ | Customer Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { color-scheme: light; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background:#f6f8fc; color:#17233b; }
[data-testid="stHeader"] { background:rgba(246,248,252,.92); }
[data-testid="stSidebar"] { background:#101c33; border-right:1px solid #22314b; }
[data-testid="stSidebar"] * { color:#e7edf8 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#9eacc5 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:8px 10px; border-radius:9px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#1c2b46; }
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] { background:#8b9dff; color:#101c33 !important; border:0; font-weight:700; }
.block-container { padding-top:1.7rem; padding-bottom:3rem; max-width:1580px; }
h1,h2,h3 { font-family:'Manrope',sans-serif; color:#17233b; letter-spacing:-.035em; }
h1 { font-size:2.25rem !important; font-weight:800 !important; }
h2 { font-weight:750 !important; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e7ebf3; padding:17px 19px; border-radius:15px; box-shadow:0 5px 18px rgba(20,35,60,.035); }
[data-testid="stMetricLabel"] { color:#66738a !important; font-size:.83rem !important; }
[data-testid="stMetricValue"] { color:#17233b !important; font-family:'Manrope',sans-serif; font-weight:800; }
[data-testid="stMetricDelta"] { font-size:.76rem; }
[data-testid="stDataFrame"] { border:1px solid #e7ebf3; border-radius:12px; overflow:hidden; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #e7ebf3; border-radius:15px; padding:8px 8px 0 8px; box-shadow:0 5px 18px rgba(20,35,60,.025); }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#e7ebf3 !important; border-radius:15px !important; }
.stButton button, .stDownloadButton button { border-radius:10px; font-weight:650; min-height:42px; }
.stButton button[kind="primary"] { background:#465cf5; border-color:#465cf5; }
.stButton button[kind="primary"]:hover { background:#3549df; border-color:#3549df; }
[data-testid="stFileUploader"] { background:#fff; border:1px dashed #b9c4d7; border-radius:14px; padding:8px; }
hr { border-color:#e5eaf2; }
.eyebrow { color:#64748b; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.13em; margin-bottom:5px; }
.hero { background:linear-gradient(120deg,#111f3a 0%,#1b3157 70%,#263e6a 100%); border-radius:20px; padding:28px 30px; color:white; margin:4px 0 22px 0; box-shadow:0 14px 34px rgba(20,35,65,.12); }
.hero h2 { color:#fff; font-size:1.7rem; margin:0 0 7px 0; }
.hero p { color:#c4d1e8; margin:0; max-width:760px; line-height:1.6; }
.hero .pill { display:inline-block; margin-top:18px; border:1px solid #496184; border-radius:999px; padding:5px 10px; color:#dce7ff; font-size:.75rem; }
.section-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:22px 0 12px 0; }
.section-head h3 { margin:0; font-size:1.15rem; }
.muted { color:#68758c; font-size:.88rem; }
.status-chip { display:inline-block; padding:5px 9px; border-radius:999px; background:#e7f8ef; color:#18744a; font-weight:700; font-size:.74rem; }
.info-card { background:#fff; border:1px solid #e7ebf3; border-radius:15px; padding:17px 18px; height:100%; }
.info-card .small-label { color:#77839a; font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.info-card .big-number { font-family:'Manrope',sans-serif; font-size:1.55rem; font-weight:800; color:#17233b; margin:5px 0; }
.info-card .note { color:#718098; font-size:.8rem; }
[data-testid="stAlert"] { border-radius:12px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

PLOT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#526078"), margin=dict(l=18,r=18,t=56,b=18), colorway=["#536dfe", "#15a88a", "#f3a847", "#a16eff", "#f06d8c", "#38a7d8", "#8b9baf", "#b6c84d"])


def reset_data(raw: pd.DataFrame, name: str) -> bool:
    try:
        cleaned, scaled, scaler, quality = preprocessing.prepare_data(raw)
        st.session_state.raw = raw.copy()
        st.session_state.dataset_name = name
        st.session_state.cleaned = cleaned
        st.session_state.scaled = scaled
        st.session_state.scaler = scaler
        st.session_state.quality = quality
        st.session_state.labeled = None
        st.session_state.model = None
        st.session_state.summary = None
        st.session_state.elbow = None
        st.session_state.silhouette = None
        st.session_state.stale = False
        st.session_state.last_k = None
        return True
    except Exception as e:
        st.error(f"Could not prepare this dataset: {e}")
        return False


def money_or_number(value, decimals=1):
    return f"{value:,.{decimals}f}"


def current_result() -> bool:
    return st.session_state.get("labeled") is not None and not st.session_state.get("stale", False)


def cluster_action(profile: str) -> tuple[str, str]:
    p = profile.lower()
    if "high income" in p and "high spending score" in p:
        return "VIP & loyalty", "Pilot early access, premium service, or a loyalty tier. Compare repeat purchase rate against a holdout group."
    if "high income" in p and "low spending score" in p:
        return "Conversion opportunity", "Test relevant premium bundles or personalised recommendations. Measure conversion and unsubscribe rates before scaling."
    if "frequent purchases" in p:
        return "Retention & rewards", "Test a frequency-based reward or replenishment reminder. Track incremental orders rather than clicks alone."
    if "infrequent purchases" in p:
        return "Re-engagement", "Try a small reactivation experiment with tailored messaging. Use a holdout group to estimate incremental lift."
    if "long membership" in p:
        return "Member appreciation", "Test a member milestone benefit or feedback invitation to learn what sustains loyalty."
    if "low income" in p or "low order value" in p:
        return "Value-led offers", "Test useful bundles or threshold-based offers while monitoring margin and discount dependency."
    return "Learn & personalise", "Test one tailored message or product recommendation and measure conversion, repeat purchase, and customer feedback."


# ---------- Data bootstrap ----------
if "raw" not in st.session_state:
    try:
        ensure_dataset()
        reset_data(preprocessing.load_csv(DATA_PATH), "Included sample dataset")
    except Exception as e:
        st.error(f"Could not load sample dataset: {e}")
        st.stop()

with st.sidebar:
    st.markdown("<div style='font-family:Manrope;font-size:1.4rem;font-weight:800;color:#fff'>◈ SegmentIQ</div><div style='font-size:.78rem;color:#9eacc5;margin:2px 0 18px 0'>CUSTOMER INTELLIGENCE</div>", unsafe_allow_html=True)
    page = st.radio("WORKSPACE", ["Executive overview", "Segment explorer", "Customer explorer", "Data & quality", "Reports & export"], label_visibility="visible")
    st.divider()
    st.markdown("<div style='font-size:.75rem;font-weight:800;letter-spacing:.12em;color:#9eacc5'>MODEL CONTROLS</div>", unsafe_allow_html=True)
    k = st.select_slider("Number of segments", options=list(range(2, 9)), value=5, key="k_choice")
    st.caption("K-Means · StandardScaler · seed 42")
    if st.button("▶  Run segmentation", type="primary", use_container_width=True):
        try:
            if len(st.session_state.cleaned) < k:
                raise ValueError("There are fewer usable records than segments.")
            if np.unique(st.session_state.scaled, axis=0).shape[0] < k:
                raise ValueError("The number of segments cannot exceed distinct feature rows.")
            labeled, model = clustering.run_kmeans(st.session_state.cleaned, st.session_state.scaled, k)
            st.session_state.labeled = labeled
            st.session_state.model = model
            st.session_state.summary = clustering.summarize_clusters(labeled)
            st.session_state.last_k = k
            st.session_state.stale = False
            try:
                st.session_state.silhouette = float(silhouette_score(st.session_state.scaled, model.labels_)) if len(set(model.labels_)) > 1 and len(model.labels_) > k else None
            except Exception:
                st.session_state.silhouette = None
            st.session_state.elbow = None
            st.toast("Segmentation complete", icon="✅")
        except Exception as e:
            st.error(str(e))
    if st.session_state.get("labeled") is not None and st.session_state.get("last_k") != k:
        st.session_state.stale = True
    if st.session_state.get("stale"):
        st.warning("Model settings changed. Re-run segmentation to refresh results.")
    st.divider()
    st.markdown("<div style='font-size:.75rem;color:#9eacc5'>LOCAL WORKSPACE · DATA STAYS IN THIS APP</div>", unsafe_allow_html=True)

raw = st.session_state.raw
cleaned = st.session_state.cleaned
quality = st.session_state.quality
labeled = st.session_state.get("labeled")
summary = st.session_state.get("summary")
current = current_result()

# ---------- Shared page header ----------
left, right = st.columns([3, 1])
with left:
    st.markdown("<div class='eyebrow'>CUSTOMER INTELLIGENCE WORKSPACE</div>", unsafe_allow_html=True)
    st.title(page)
    st.markdown(f"<span class='muted'>Dataset: <b>{st.session_state.dataset_name}</b> &nbsp;·&nbsp; {len(cleaned):,} usable customer records</span>", unsafe_allow_html=True)
with right:
    st.markdown("<div style='text-align:right;padding-top:16px'><span class='status-chip'>{'SEGMENTS READY' if current else 'DATA READY'}</span><div class='muted' style='margin-top:7px'>Updated for this session</div></div>", unsafe_allow_html=True)

# ---------- Executive overview ----------
if page == "Executive overview":
    st.markdown(f"""<div class='hero'><h2>Understand your customers. Act with confidence.</h2><p>Turn customer attributes into interpretable segments, explore behavioural patterns, and shape measurable marketing experiments. Start with your data, then run segmentation from the left panel.</p><span class='pill'>● Local analysis</span> <span class='pill'>● Interactive insights</span> <span class='pill'>● Exportable reports</span></div>""", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers analysed", f"{len(cleaned):,}", delta=f"{quality['cleaned_rows']-quality['duplicate_rows']:,} after cleaning" if quality['duplicate_rows'] else None)
    c2.metric("Segments discovered", str(st.session_state.last_k) if current else "—", delta="Current model" if current else "Run segmentation")
    c3.metric("Data completeness", f"{max(0, 100 - quality['missing_values'] / max(1, quality['original_rows']*len(FEATURE_COLUMNS))*100):.1f}%")
    c4.metric("Silhouette score", f"{st.session_state.silhouette:.3f}" if current and st.session_state.silhouette is not None else "—", help="Higher is generally better separated, but interpret alongside business context.")
    st.markdown("<div class='section-head'><h3>Business snapshot</h3><span class='muted'>Portfolio-level signals from the current dataset</span></div>", unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a:
        st.markdown(f"<div class='info-card'><div class='small-label'>Average annual income</div><div class='big-number'>{money_or_number(cleaned['AnnualIncome'].mean())}</div><div class='note'>Dataset units; confirm currency and scale before business use.</div></div>", unsafe_allow_html=True)
    with b:
        st.markdown(f"<div class='info-card'><div class='small-label'>Average order value</div><div class='big-number'>{money_or_number(cleaned['AverageOrderValue'].mean(),2)}</div><div class='note'>Average per recorded customer, in source-data units.</div></div>", unsafe_allow_html=True)
    with c:
        st.markdown(f"<div class='info-card'><div class='small-label'>Average purchase frequency</div><div class='big-number'>{money_or_number(cleaned['PurchaseFrequency'].mean())}</div><div class='note'>Interpretation depends on the period represented in your source.</div></div>", unsafe_allow_html=True)
    l,r = st.columns([1.4,1])
    with l:
        st.markdown("<div class='section-head'><h3>Customer landscape</h3></div>", unsafe_allow_html=True)
        f = px.scatter(cleaned, x="AnnualIncome", y="SpendingScore", size="PurchaseFrequency", hover_data=["CustomerID","Age","AverageOrderValue"], opacity=.72, title="Income vs. spending behaviour")
        f.update_layout(**PLOT, xaxis_title="Annual income (source units)", yaxis_title="Spending score")
        st.plotly_chart(f, use_container_width=True)
    with r:
        st.markdown("<div class='section-head'><h3>Data readiness</h3></div>", unsafe_allow_html=True)
        q = pd.DataFrame({"Check":["Duplicate records","Missing feature values","Invalid values handled","Rows retained"],"Count":[quality['duplicate_rows'],quality['missing_values'],quality['invalid_values'],quality['cleaned_rows']]})
        f = px.bar(q, x="Count", y="Check", orientation="h", text="Count", title="Data quality checks")
        f.update_layout(**PLOT, showlegend=False, yaxis_title="")
        st.plotly_chart(f, use_container_width=True)
    if current:
        st.markdown("<div class='section-head'><h3>Segment distribution</h3><span class='muted'>Share of customers in each segment</span></div>", unsafe_allow_html=True)
        dist = summary.rename(columns={"Cluster":"Segment","Customers":"Customers"})
        fig = px.bar(dist, x="Segment", y="Customers", color="Segment", text=dist["Percentage"].map(lambda x:f"{x:.1f}%"), title="Current segmentation result")
        fig.update_layout(**PLOT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("**Next step:** choose the number of segments in the left panel and select **Run segmentation**. The segment explorer and customer explorer will then show cluster-level insights.")
    st.caption("Decision-support notice: clustering finds statistical similarities; it does not prove customer intent or guarantee campaign performance. Validate proposed actions with controlled experiments.")

# ---------- Segment explorer ----------
elif page == "Segment explorer":
    if not current:
        st.info("Run segmentation from the left panel to unlock segment comparisons, profiles, and experiment ideas.")
    else:
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Customers segmented", f"{len(labeled):,}")
        s2.metric("Segments", st.session_state.last_k)
        s3.metric("Model inertia", f"{st.session_state.model.inertia_:,.1f}")
        s4.metric("Silhouette score", f"{st.session_state.silhouette:.3f}" if st.session_state.silhouette is not None else "Unavailable")
        l,r = st.columns([1.4,1])
        with l:
            st.markdown("<div class='section-head'><h3>Explore the segment map</h3></div>", unsafe_allow_html=True)
            xf = st.selectbox("Horizontal axis", FEATURE_COLUMNS, index=1)
            yf = st.selectbox("Vertical axis", FEATURE_COLUMNS, index=2)
            fig = px.scatter(labeled, x=xf, y=yf, color=labeled["Cluster"].astype(str), hover_data=[c for c in ["CustomerID"]+FEATURE_COLUMNS if c in labeled], color_discrete_sequence=px.colors.qualitative.Safe, title="Each point represents one customer")
            centers = pd.DataFrame(st.session_state.scaler.inverse_transform(st.session_state.model.cluster_centers_), columns=FEATURE_COLUMNS)
            fig.add_trace(go.Scatter(x=centers[xf], y=centers[yf], mode="markers", name="Segment centres", marker=dict(symbol="x", size=16, color="#14213d", line=dict(width=2))))
            fig.update_layout(**PLOT, legend_title="Segment", xaxis_title=xf, yaxis_title=yf)
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.markdown("<div class='section-head'><h3>Segment size</h3></div>", unsafe_allow_html=True)
            counts = labeled["Cluster"].value_counts().sort_index().rename_axis("Segment").reset_index(name="Customers")
            counts["Share"] = counts["Customers"] / len(labeled) * 100
            fig = px.pie(counts, names=counts["Segment"].astype(str), values="Customers", hole=.68, title="Customer mix")
            fig.update_layout(**PLOT, legend_title="Segment")
            fig.update_traces(textposition="inside", textinfo="percent")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='section-head'><h3>Segment profiles</h3><span class='muted'>Average feature values for each group</span></div>", unsafe_allow_html=True)
        display_summary = summary.rename(columns={"Cluster":"Segment", "Customers":"Customers", "Percentage":"Share (%)", "AvgAge":"Avg. age", "AvgAnnualIncome":"Avg. annual income", "AvgSpendingScore":"Avg. spending score", "AvgPurchaseFrequency":"Avg. purchase frequency", "AvgOrderValue":"Avg. order value", "AvgMembershipYears":"Avg. membership (years)"})
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
        st.markdown("<div class='section-head'><h3>What makes each segment different?</h3></div>", unsafe_allow_html=True)
        means = labeled.groupby("Cluster")[FEATURE_COLUMNS].mean()
        zmeans = (means-cleaned[FEATURE_COLUMNS].mean()) / cleaned[FEATURE_COLUMNS].std(ddof=0).replace(0,1)
        fig = px.imshow(zmeans, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-2, zmax=2, title="Relative feature profile (standard deviations from overall mean)")
        fig.update_layout(**PLOT, xaxis_title="Customer attributes", yaxis_title="Segment")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='section-head'><h3>Experiment ideas</h3><span class='muted'>Hypotheses to validate—not automated decisions</span></div>", unsafe_allow_html=True)
        cards = st.columns(2)
        for i, (_, row) in enumerate(summary.iterrows()):
            action, idea = cluster_action(row["Profile"])
            with cards[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**Segment {int(row['Cluster'])} · {row['Percentage']:.1f}% of customers**")
                    st.write(row["Profile"])
                    st.markdown(f"**Potential play: {action}**")
                    st.write(idea)

# ---------- Customer explorer ----------
elif page == "Customer explorer":
    if not current:
        st.info("Run segmentation first to filter customers by segment. Data can still be previewed on the Data & quality page.")
    else:
        st.markdown("Use the filters to inspect a segment or find customer records matching a range of values. Filters affect the table and downloads on this page only.")
        f1,f2,f3 = st.columns([1,1,1.3])
        segment_options = ["All segments"] + [str(int(x)) for x in sorted(labeled["Cluster"].unique())]
        with f1: segment_filter = st.selectbox("Segment", segment_options)
        with f2: search = st.text_input("Customer ID contains", placeholder="e.g. C010")
        with f3: sort_by = st.selectbox("Sort customers by", FEATURE_COLUMNS, index=4)
        bounds = {}
        range_cols = st.columns(3)
        for i, feat in enumerate(["AnnualIncome", "SpendingScore", "AverageOrderValue"]):
            with range_cols[i]:
                lo, hi = float(labeled[feat].min()), float(labeled[feat].max())
                bounds[feat] = st.slider(feat, min_value=float(round(lo, 2)), max_value=float(round(hi, 2)), value=(float(round(lo,2)), float(round(hi,2))))
        view = labeled.copy()
        if segment_filter != "All segments": view = view[view["Cluster"] == int(segment_filter)]
        if search.strip(): view = view[view["CustomerID"].astype(str).str.contains(search.strip(), case=False, na=False)]
        for feat, (lo,hi) in bounds.items(): view = view[view[feat].between(lo,hi)]
        view = view.sort_values(sort_by, ascending=False)
        x1,x2,x3 = st.columns(3)
        x1.metric("Matching customers", f"{len(view):,}")
        x2.metric("Average order value", f"{view['AverageOrderValue'].mean():,.2f}" if len(view) else "—")
        x3.metric("Average spending score", f"{view['SpendingScore'].mean():,.1f}" if len(view) else "—")
        st.dataframe(view, use_container_width=True, hide_index=True, height=460)
        st.download_button("Download filtered customer list", view.to_csv(index=False).encode("utf-8-sig"), "filtered_customer_segments.csv", "text/csv", disabled=view.empty)
        st.caption("Use customer identifiers and attributes according to your organisation's privacy and access policies. Avoid using segment membership as the sole basis for consequential decisions.")

# ---------- Data and quality ----------
elif page == "Data & quality":
    st.markdown("<div class='section-head'><h3>Bring your own customer data</h3><span class='muted'>CSV import · schema validation · cleaning summary</span></div>", unsafe_allow_html=True)
    st.write("Required columns: `CustomerID` plus the six numeric attributes shown below. Uploads are processed in this app session; the source CSV is not overwritten.")
    with st.expander("Required CSV schema", expanded=False):
        st.dataframe(pd.DataFrame({"Column": ["CustomerID"]+FEATURE_COLUMNS, "Purpose":["Unique customer identifier", "Age", "Annual income (source units)", "Spending score", "Purchase frequency", "Average order value (source units)", "Membership duration in years"]}), use_container_width=True, hide_index=True)
        st.download_button("Download CSV template", pd.DataFrame(columns=["CustomerID"]+FEATURE_COLUMNS).to_csv(index=False).encode("utf-8"), "customer_data_template.csv", "text/csv")
    upload = st.file_uploader("Upload customer CSV", type=["csv"], key="customer_csv_upload")
    col1,col2 = st.columns([1,1])
    with col1:
        if upload is not None and st.button("Validate and use uploaded data", type="primary", use_container_width=True):
            try:
                frame = pd.read_csv(upload)
                if frame.empty: raise ValueError("The CSV contains no customer rows.")
                if reset_data(frame, upload.name): st.rerun()
            except Exception as e: st.error(f"Unable to load CSV: {e}")
    with col2:
        if st.button("Restore sample dataset", use_container_width=True):
            if reset_data(preprocessing.load_csv(DATA_PATH), "Included sample dataset"): st.rerun()
    st.markdown("<div class='section-head'><h3>Data quality overview</h3></div>", unsafe_allow_html=True)
    q1,q2,q3,q4 = st.columns(4)
    q1.metric("Source rows", f"{quality['original_rows']:,}")
    q2.metric("Rows after cleaning", f"{quality['cleaned_rows']:,}")
    q3.metric("Duplicate rows", f"{quality['duplicate_rows']:,}")
    q4.metric("Missing / invalid values", f"{quality['missing_values']:,} / {quality['invalid_values']:,}")
    left,right = st.columns([1.1,1])
    with left:
        st.markdown("**Raw data preview**")
        st.dataframe(raw.head(100), use_container_width=True, hide_index=True, height=300)
    with right:
        st.markdown("**Column health**")
        report = pd.DataFrame({"Column":raw.columns, "Data type":[str(x) for x in raw.dtypes], "Missing":[int(raw[c].isna().sum()) for c in raw.columns], "Unique values":[int(raw[c].nunique(dropna=True)) for c in raw.columns]})
        st.dataframe(report, use_container_width=True, hide_index=True, height=300)
    st.markdown("<div class='section-head'><h3>Cleaning audit</h3></div>", unsafe_allow_html=True)
    st.markdown("1. Exact duplicate rows are removed.  2. Numeric features are validated against broad plausibility ranges.  3. Missing or invalid feature values are imputed with feature medians.  4. Features are standardised with `StandardScaler`.  `CustomerID` is retained for reference and excluded from model fitting.")
    missing = pd.DataFrame({"Feature":list(quality["missing_by_column"]), "Missing values":list(quality["missing_by_column"].values())})
    fig = px.bar(missing, x="Feature", y="Missing values", title="Missing values in uploaded source")
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cleaned.describe().T, use_container_width=True)
    st.download_button("Download cleaned dataset", cleaned.to_csv(index=False).encode("utf-8-sig"), "cleaned_customers.csv", "text/csv")

# ---------- Reports and export ----------
elif page == "Reports & export":
    st.markdown("Export transparent, portable outputs for reporting and downstream analysis. Downloads reflect the dataset currently loaded in this session.")
    e1,e2 = st.columns(2)
    with e1:
        with st.container(border=True):
            st.markdown("**Cleaned customer data**")
            st.write("Validated numeric attributes, duplicate removal, and median imputation.")
            st.download_button("Download cleaned CSV", cleaned.to_csv(index=False).encode("utf-8-sig"), "cleaned_customers.csv", "text/csv", use_container_width=True)
    with e2:
        with st.container(border=True):
            st.markdown("**Source data quality report**")
            st.write("Counts of rows, duplicates, missing values, invalid values, and retained records.")
            quality_frame = pd.DataFrame([{"Dataset":st.session_state.dataset_name, "RowsLoaded":quality["original_rows"], "RowsAfterCleaning":quality["cleaned_rows"], "Duplicates":quality["duplicate_rows"], "MissingFeatureValues":quality["missing_values"], "InvalidValues":quality["invalid_values"], "ImputedValues":quality["imputed_values"]}])
            st.download_button("Download quality report", quality_frame.to_csv(index=False).encode("utf-8-sig"), "data_quality_report.csv", "text/csv", use_container_width=True)
    if current:
        st.markdown("<div class='section-head'><h3>Segmentation outputs</h3></div>", unsafe_allow_html=True)
        e1,e2,e3 = st.columns(3)
        with e1:
            st.download_button("Download segmented customers", labeled.to_csv(index=False).encode("utf-8-sig"), "customer_segments.csv", "text/csv", use_container_width=True)
        with e2:
            st.download_button("Download segment profiles", summary.to_csv(index=False).encode("utf-8-sig"), "segment_summary.csv", "text/csv", use_container_width=True)
        with e3:
            evalrow = pd.DataFrame([{"K":st.session_state.last_k, "Inertia":float(st.session_state.model.inertia_), "SilhouetteScore":st.session_state.silhouette}])
            st.download_button("Download model metrics", evalrow.to_csv(index=False).encode("utf-8-sig"), "clustering_evaluation.csv", "text/csv", use_container_width=True)
        if st.button("Calculate elbow curve", type="primary"):
            try: st.session_state.elbow = evaluation.run_elbow(st.session_state.scaled)
            except Exception as e: st.error(str(e))
        if st.session_state.get("elbow") is not None:
            fig = px.line(st.session_state.elbow, x="K", y="Inertia", markers=True, title="Elbow method: inertia by number of segments")
            fig.add_vline(x=st.session_state.last_k, line_dash="dash", annotation_text=f"Selected K={st.session_state.last_k}")
            fig.update_layout(**PLOT)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download elbow results", st.session_state.elbow.to_csv(index=False).encode("utf-8-sig"), "elbow_results.csv", "text/csv")
        report = f"""# SegmentIQ — Customer Segmentation Report\n\n**Dataset:** {st.session_state.dataset_name}\n\n## Data quality\n- Source rows: {quality['original_rows']}\n- Rows after cleaning: {quality['cleaned_rows']}\n- Duplicate rows removed: {quality['duplicate_rows']}\n- Missing feature values: {quality['missing_values']}\n- Invalid values handled: {quality['invalid_values']}\n- Values imputed: {quality['imputed_values']}\n\n## Model\n- Algorithm: K-Means\n- Number of segments: {st.session_state.last_k}\n- Inertia: {st.session_state.model.inertia_:.4f}\n- Silhouette score: {st.session_state.silhouette if st.session_state.silhouette is not None else 'Unavailable'}\n\n## Segment profiles\n{summary.to_markdown(index=False)}\n\n## Interpretation notes\nSegment labels are arbitrary. Clustering indicates similarity in the selected features, not verified customer intent. Confirm units and data provenance before business use. Treat campaign suggestions as hypotheses and test them with appropriate controls.\n"""
        st.download_button("Download executive analysis report (Markdown)", report.encode("utf-8"), "segmentiq_analysis_report.md", "text/markdown")
    else:
        st.info("Run segmentation from the left panel to generate segment-level exports and model metrics.")

st.divider()
st.markdown("<div style='display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;color:#8792a5;font-size:.75rem'><span>SEGMENTIQ · CUSTOMER INTELLIGENCE</span><span>Decision support, not a substitute for customer research or controlled experiments.</span></div>", unsafe_allow_html=True)
