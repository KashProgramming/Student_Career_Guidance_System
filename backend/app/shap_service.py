from __future__ import annotations

import base64
from io import BytesIO
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import shap


def _get_shap_values(model,X_input:pd.DataFrame):
    explainer=shap.TreeExplainer(model)
    shap_values=explainer(X_input)

    if isinstance(shap_values,list):
        shap_values=shap_values[1]

    return shap_values[0]


def _plot_to_base64(fig)->str:
    buf=BytesIO()
    fig.savefig(buf,format="png",dpi=150,bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_explainability(model,X_input:pd.DataFrame)->Dict:
    shap_exp=_get_shap_values(model,X_input)

    impact_df=pd.DataFrame({
        "feature":X_input.columns,
        "impact":shap_exp.values
    })

    impact_df=impact_df.sort_values("impact",ascending=False)

    strengths=impact_df[impact_df["impact"]>0].head(5)
    weaknesses=impact_df[impact_df["impact"]<0].tail(5)

    shap.plots.bar(
        shap_exp,
        show=False,
        max_display=12
    )
    bar_plot_base64=_plot_to_base64(plt.gcf())

    shap.plots.waterfall(
        shap_exp,
        show=False,
        max_display=12
    )
    waterfall_plot_base64=_plot_to_base64(plt.gcf())

    return{
        "feature_impacts":impact_df.to_dict(orient="records"),
        "strengths":strengths.to_dict(orient="records"),
        "weaknesses":weaknesses.to_dict(orient="records"),
        "waterfall_plot_base64":waterfall_plot_base64,
        "bar_plot_base64":bar_plot_base64,
    }