"""Transparent, rule-based product matching. No ML here on purpose: every
recommendation can be explained by the filters and points below."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

POINTS_PER_CONCERN = 3
POINTS_PER_RATING_STAR = 1


@dataclass
class Preferences:
    skin_type: str
    concerns: list[str] = field(default_factory=list)
    budget_max: float = 50.0
    categories: list[str] = field(default_factory=list)
    fragrance_free_only: bool = False
    vegan_only: bool = False


@dataclass
class Recommendation:
    results: pd.DataFrame
    funnel: list[tuple[str, int]]  # (step description, products remaining)


def recommend(products: pd.DataFrame, prefs: Preferences, top_n: int = 6) -> Recommendation:
    df = products.copy()
    funnel = [("Full sample catalog", len(df))]

    df = df[df["skin_types"].map(lambda types: prefs.skin_type in types)]
    funnel.append((f"Listed for {prefs.skin_type} skin", len(df)))

    df = df[df["price_usd"] <= prefs.budget_max]
    funnel.append((f"Within ${prefs.budget_max:.0f} budget", len(df)))

    if prefs.categories:
        df = df[df["category"].isin(prefs.categories)]
        funnel.append((f"Product type: {', '.join(prefs.categories)}", len(df)))
    if prefs.fragrance_free_only:
        df = df[df["fragrance_free"]]
        funnel.append(("Fragrance-free only", len(df)))
    if prefs.vegan_only:
        df = df[df["vegan"]]
        funnel.append(("Vegan only", len(df)))

    if prefs.concerns:
        df = df.assign(matched=df["concerns"].map(lambda cs: [c for c in prefs.concerns if c in cs]))
        df = df[df["matched"].map(len) > 0]
        funnel.append(("Tagged for at least one of your concerns", len(df)))
    else:
        df = df.assign(matched=[[] for _ in range(len(df))])

    df = df.assign(
        score=df["matched"].map(len) * POINTS_PER_CONCERN + df["avg_rating"] * POINTS_PER_RATING_STAR
    )
    df = df.sort_values(["score", "price_usd"], ascending=[False, True]).head(top_n)
    df = df.assign(reasons=[_reasons(row, prefs) for _, row in df.iterrows()])
    return Recommendation(results=df.reset_index(drop=True), funnel=funnel)


def _reasons(row: pd.Series, prefs: Preferences) -> list[str]:
    reasons = [f"Catalog lists it for {prefs.skin_type} skin"]
    if row["matched"]:
        reasons.append(
            f"Tagged for {', '.join(row['matched'])} (+{len(row['matched']) * POINTS_PER_CONCERN} pts)"
        )
    reasons.append(f"Sample rating {row['avg_rating']:.1f}/5 (+{row['avg_rating'] * POINTS_PER_RATING_STAR:.1f} pts)")
    reasons.append(f"${row['price_usd']:.2f} is within your ${prefs.budget_max:.0f} budget")
    return reasons
