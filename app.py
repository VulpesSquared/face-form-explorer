from __future__ import annotations

from pathlib import Path
from hashlib import sha256

import pandas as pd
import plotly.express as px
import streamlit as st

from face_analysis.clustering import cluster_people
from face_analysis.exporting import workbook_bytes
from face_analysis.geometry import FEATURE_GROUPS, FEATURE_LABELS
from face_analysis.pipeline import analyze_dataset
from face_analysis.similarity import build_similarity_matrix, compare_people, standardized_features
from face_analysis.uploads import records_from_uploads

st.set_page_config(
    page_title="Face Form Explorer",
    page_icon="◌",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {max-width: 1220px; padding-top: 2rem;}
.small-note {color: #6b7280; font-size: 0.9rem;}
.hero {padding: 1.25rem 0 .25rem 0;}
.hero h1 {margin-bottom: .15rem;}
.metric-card {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 12px 14px;}
</style>
""",
    unsafe_allow_html=True,
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


def fmt_trait(value: float) -> str:
    if value <= -1.25:
        return "very low"
    if value <= -0.50:
        return "low"
    if value < 0.50:
        return "near the dataset middle"
    if value < 1.25:
        return "high"
    return "very high"


st.markdown('<div class="hero">', unsafe_allow_html=True)
st.title("Face Form Explorer")
st.caption("An interpretable facial-morphology sandbox using MediaPipe Face Landmarker geometry.")
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("What this app does — and does not do", expanded=False):
    st.markdown(
        """
This app compares **visible facial geometry in user-labeled photographs**. It measures landmark ratios, aggregates multiple accepted photos per person, and lets you explore relative similarity, feature families, and unsupervised clusters.

It **does not identify unknown people**, estimate ancestry/ethnicity, infer personality, health, intelligence, attractiveness, or other traits that cannot be reliably established from geometry alone. Similarity scores are descriptive indices within the current dataset — not probabilities that two people are related or are the same person.

**Privacy:** when this app is hosted on Streamlit Community Cloud, uploaded images are sent to that hosted app for processing. MediaPipe runs the landmark model in the app process; the photos are not intentionally saved by this application. For more sensitive images, run the same repository locally instead.
"""
    )

st.subheader("1 · Build a dataset")
st.write(
    "Upload individual images, or a ZIP whose folders are named for each person. "
    "For loose files, use `Person Name__01.jpg` to auto-label them, or edit the label below."
)

uploads = st.file_uploader(
    "Images or ZIP",
    type=["jpg", "jpeg", "png", "webp", "zip"],
    accept_multiple_files=True,
)

if uploads:
    try:
        raw_records = records_from_uploads(uploads)
    except ValueError as exc:
        st.session_state.pop("analysis", None)
        st.error(str(exc))
        st.stop()
    if not raw_records:
        st.session_state.pop("analysis", None)
        st.warning("I couldn't find any supported images in those uploads.")
        st.stop()

    label_df = pd.DataFrame([
        {"filename": r["filename"], "person": r["person"]}
        for r in raw_records
    ])
    st.caption("Check or edit the person label for each image before analysis.")
    edited = st.data_editor(
        label_df,
        hide_index=True,
        width="stretch",
        column_config={
            "filename": st.column_config.TextColumn("Image", disabled=True),
            "person": st.column_config.TextColumn("Person", required=True),
        },
        key="labels",
    )

    # Match editor rows by position: separate uploads can have identical names.
    records = [
        {**r, "person": "" if pd.isna(label) else str(label).strip()}
        for r, label in zip(raw_records, edited["person"], strict=True)
    ]
    signature = tuple((r["filename"], r["person"], sha256(r["bytes"]).hexdigest()) for r in records)
    if st.session_state.get("analysis", {}).get("signature") != signature:
        st.session_state.pop("analysis", None)

    people = sorted({r["person"] for r in records if r["person"]})
    c1, c2, c3 = st.columns(3)
    c1.metric("Images", len(records))
    c2.metric("People", len(people))
    c3.metric("Suggested", "3–6 photos/person")

    if len(people) < 2:
        st.info("Add at least two labeled people for similarity comparisons.")

    if st.button("Analyze dataset", type="primary", width="stretch"):
        try:
            with st.spinner("Detecting landmarks, checking pose, and calculating facial ratios…"):
                image_df, person_df, failure_df = analyze_dataset(records, MODEL_PATH)
        except (RuntimeError, OSError, ValueError, ImportError) as exc:
            st.session_state.pop("analysis", None)
            st.error(f"Could not start analysis: {exc}")
            st.stop()
        st.session_state["analysis"] = {
            "signature": signature,
            "image_df": image_df,
            "person_df": person_df,
            "failure_df": failure_df,
        }
else:
    st.session_state.pop("analysis", None)

analysis = st.session_state.get("analysis")
if not analysis:
    st.info("Upload and label photos above, then run the analysis.")
    st.stop()

image_df = analysis["image_df"]
person_df = analysis["person_df"]
failure_df = analysis["failure_df"]

if person_df.empty:
    st.error("No images passed analysis. Try clearer, larger, more frontal photos with one face per image.")
    if not failure_df.empty:
        st.dataframe(failure_df, hide_index=True, width="stretch")
    st.stop()

st.divider()
st.subheader("2 · Explore")

tabs = st.tabs(["Overview", "Similarity", "Compare", "Features", "Clusters", "Export"])

with tabs[0]:
    a, b, c = st.columns(3)
    a.metric("People analyzed", len(person_df))
    b.metric("Images accepted", len(image_df))
    c.metric("Images rejected", len(failure_df))

    st.markdown("#### Accepted photos per person")
    st.dataframe(
        person_df[["person", "accepted_images"]].sort_values("accepted_images", ascending=False),
        hide_index=True,
        width="stretch",
    )
    if not failure_df.empty:
        with st.expander("Rejected images and why"):
            st.dataframe(failure_df, hide_index=True, width="stretch")

with tabs[1]:
    all_groups = list(FEATURE_GROUPS)
    selected_groups = st.multiselect(
        "Similarity feature families",
        all_groups,
        default=all_groups,
        help="Turn feature families on/off to see how the similarity map changes.",
    )
    selected_features = sorted({f for g in selected_groups for f in FEATURE_GROUPS[g] if f in person_df.columns})
    sim_df = build_similarity_matrix(person_df, selected_features or None)

    fig = px.imshow(
        sim_df,
        text_auto=".0f",
        zmin=0,
        zmax=100,
        aspect="auto",
        labels={"color": "Relative similarity"},
    )
    fig.update_layout(height=max(500, 34 * len(sim_df) + 180))
    st.plotly_chart(fig, width="stretch")

    if len(person_df) >= 2:
        focus = st.selectbox("Find closest matches for", person_df["person"].tolist())
        ranking = sim_df.loc[focus].drop(index=focus).sort_values(ascending=False).rename("similarity").rename_axis("person").reset_index()
        ranking["similarity"] = ranking["similarity"].round(1)
        st.dataframe(ranking.head(10), hide_index=True, width="stretch")
    st.caption("The similarity index is relative to this dataset and is not an identity or kinship probability.")

with tabs[2]:
    names = person_df["person"].tolist()
    if len(names) < 2:
        st.info("Add another person to compare.")
    else:
        col_a, col_b = st.columns(2)
        person_a = col_a.selectbox("Person A", names, index=0)
        candidates_b = [n for n in names if n != person_a]
        person_b = col_b.selectbox("Person B", candidates_b, index=0)
        comparison = compare_people(person_df, person_a, person_b)

        st.metric("Overall relative morphology similarity", f"{comparison['overall']:.1f} / 100")
        group_df = pd.DataFrame(
            [{"feature family": k, "similarity": round(v, 1)} for k, v in comparison["groups"].items()]
        ).sort_values("similarity", ascending=False)
        st.bar_chart(group_df.set_index("feature family"))

        left, right = st.columns(2)
        with left:
            st.markdown("#### Most alike")
            st.dataframe(
                pd.DataFrame(comparison["most_similar"])[["label", "standardized_difference"]].rename(
                    columns={"label": "Feature", "standardized_difference": "Difference"}
                ),
                hide_index=True,
                width="stretch",
            )
        with right:
            st.markdown("#### Most different")
            st.dataframe(
                pd.DataFrame(comparison["most_different"])[["label", "standardized_difference"]].rename(
                    columns={"label": "Feature", "standardized_difference": "Difference"}
                ),
                hide_index=True,
                width="stretch",
            )

with tabs[3]:
    Z, _ = standardized_features(person_df)
    feature = st.selectbox(
        "Feature",
        Z.columns,
        format_func=lambda x: FEATURE_LABELS.get(x, x.replace("_", " ").title()),
    )
    feature_table = pd.DataFrame({
        "person": person_df["person"],
        "standardized_value": Z[feature],
    }).sort_values("standardized_value")
    feature_table["dataset_position"] = feature_table["standardized_value"].map(fmt_trait)
    st.bar_chart(feature_table.set_index("person")["standardized_value"])
    st.dataframe(feature_table, hide_index=True, width="stretch")
    st.caption("Positions are relative to the people currently loaded, not population norms.")

with tabs[4]:
    max_clusters = min(6, len(person_df))
    if len(person_df) < 2:
        st.info("Add another person to cluster.")
        cluster_df = cluster_people(person_df, 1)
    else:
        default_clusters = min(3, max_clusters)
        if max_clusters == 2:
            k = 2
            st.caption("Two people: showing two clusters.")
        else:
            k = st.slider("Number of clusters", 2, max_clusters, default_clusters)
        cluster_df = cluster_people(person_df, k)
        fig = px.scatter(
            cluster_df,
            x="pca_x",
            y="pca_y",
            color=cluster_df["cluster"].astype(str),
            text="person",
            labels={"color": "Cluster", "pca_x": "Morphology axis 1", "pca_y": "Morphology axis 2"},
        )
        fig.update_traces(textposition="top center", marker={"size": 12})
        fig.update_layout(height=620)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(cluster_df.sort_values(["cluster", "person"]), hide_index=True, width="stretch")
        st.caption("Clusters are exploratory groupings of the measured ratios; they are not demographic categories.")

with tabs[5]:
    sim_df = build_similarity_matrix(person_df)
    cluster_df = cluster_people(person_df, min(3, max(1, len(person_df))))
    xlsx = workbook_bytes(person_df, image_df, failure_df, sim_df, cluster_df)
    st.download_button(
        "Download Excel workbook",
        data=xlsx,
        file_name="face_form_explorer_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    st.download_button(
        "Download person-level CSV",
        data=person_df.to_csv(index=False).encode("utf-8"),
        file_name="face_form_people.csv",
        mime="text/csv",
        width="stretch",
    )
