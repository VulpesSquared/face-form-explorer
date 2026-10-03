from __future__ import annotations

from io import BytesIO

import pandas as pd


def workbook_bytes(
    person_df: pd.DataFrame,
    image_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        person_df.to_excel(writer, sheet_name="People", index=False)
        image_df.to_excel(writer, sheet_name="Accepted Images", index=False)
        failure_df.to_excel(writer, sheet_name="Rejected Images", index=False)
        similarity_df.to_excel(writer, sheet_name="Similarity")
        cluster_df.to_excel(writer, sheet_name="Clusters", index=False)
    return out.getvalue()
