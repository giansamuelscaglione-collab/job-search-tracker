import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR-K2bPgkFpcavReVoXH-K4HQWy6QxN12o6jWB5ywewq32TEo0qG9JaZpJgCMkVOiNTZB5P0Rp3UnuL/pub?gid=1301135712&single=true&output=csv"

# ---------- Settings you can change ----------
GHOST_DAYS_AFTER_DEADLINE = 14   # ghosted if still no reply this many days after the deadline
GHOST_DAYS_NO_DEADLINE = 21      # no deadline listed: ghosted this many days after applying
SMALL_SAMPLE = 20                # show a caution until this many applications have resolved
MIN_TO_COMPARE = 3               # an option needs this many resolved applications before it is compared
SCORE_WINDOW = 5                 # pre-apply check: "similar score" means within this many points
# confidence label by number of resolved applications behind a rate
CONFIDENCE_STEPS = [(30, "High"), (15, "Medium"), (5, "Low")]   # below 5: "Too few to tell"

PENDING_STATUSES = {"No Response Yet", "Application Confirmed", ""}
FUNNEL_ORDER = ["No Response Yet", "Application Confirmed", "Interview Request", "Offer", "Rejected"]

REQUIRED = ["Company", "Role", "Date Applied", "Status", "AI Fit Score", "Last Updated"]
OPTIONAL = [
    "Posted Date", "Application Deadline", "Applicant Count",
    "Application Method", "Connection Level", "Furthest Stage",
    "Resume Version", "Cover Letter", "Follow-up Sent", "Role Type",
]
YES_NO_COLS = ["Cover Letter", "Follow-up Sent"]
ROLE_TYPES = ["Data / Analytics", "Business Analysis", "Operations / Strategy",
              "Project Coordination", "Finance", "Sales / Revenue", "Other"]
# AI Fit Score buckets (edges are the lowest score in each bucket)
SCORE_BINS = [(0, "Under 60"), (60, "60-74"), (75, "75+")]

FACTORS = [
    ("Connection Level", ["None", "Know someone there", "Referral submitted",
                          "Spoke with recruiter or hiring manager", "Undefined"]),
    ("Application Method", ["LinkedIn", "Indeed", "Company Site", "Direct Email", "School Portal", "Other", "Undefined"]),
    ("Applicant Count", ["<25", "Under 25", "25-100", "100+", ">100", "Undefined"]),
    ("Posting Age", ["0-2 days", "3-7 days", "8+ days", "Undefined"]),
    ("Score Bucket", [label for _, label in SCORE_BINS] + ["Undefined"]),
    ("Role Type", ROLE_TYPES + ["Undefined"]),
    ("Resume Version", ["Undefined"]),
    ("Cover Letter", ["Yes", "No", "Undefined"]),
    ("Follow-up Sent", ["Yes", "No", "Undefined"]),
]
# factors you choose yourself, as opposed to context you can only pick around
LEVERS = {"Application Method", "Connection Level", "Posting Age", "Resume Version", "Role Type",
          "Cover Letter", "Follow-up Sent"}
LABELS = {"Posting Age": "Days between posting and applying", "Score Bucket": "AI Fit Score"}
METRICS = {
    "Interview rate": ["Interview", "Offer"],
    "Response rate": ["Rejected", "Interview", "Offer"],
}

st.set_page_config(page_title="Job Search Auto-Tracker", layout="wide")
st.title("Job Search Auto-Tracker")

if st.button("Refresh data"):
    st.cache_data.clear()


@st.cache_data(ttl=300)
def load_data():
    # keep_default_na=False so the text "None" (no connection) is not treated as missing
    df = pd.read_csv(CSV_URL, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_text(series):
    return series.astype("string").str.strip().fillna("")


def score_bucket(scores):
    edges = [-1] + [low - 0.01 for low, _ in SCORE_BINS[1:]] + [100]
    return (pd.cut(scores, bins=edges, labels=[label for _, label in SCORE_BINS])
            .astype("string").fillna("Undefined"))


def yes_no(series):
    """Maps Y / yes / TRUE / x / 1 to Yes and N / no / FALSE / 0 to No; anything else stays as typed."""
    low = series.str.lower()
    out = series.where(~low.isin(["y", "yes", "true", "x", "1", "sent"]), "Yes")
    return out.where(~low.isin(["n", "no", "false", "0"]), "No")


def parse_dates(series):
    """Accepts 2026-10-04, 10/3/2026, and 10/3/2026 16:30:00 in the same column."""
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def prepare(raw):
    df = raw.copy()
    # accept either spelling of the applicant-count column
    if "Applicant Count" not in df.columns and "Application Count" in df.columns:
        df = df.rename(columns={"Application Count": "Applicant Count"})
    for col in OPTIONAL:
        if col not in df.columns:
            df[col] = ""

    df = df[clean_text(df["Company"]) != ""].copy()

    for col in ["Company", "Role", "Status", "Applicant Count", "Application Method",
                "Connection Level", "Furthest Stage", "Resume Version", "Cover Letter",
                "Follow-up Sent", "Role Type"]:
        df[col] = clean_text(df[col])
    df["Status"] = df["Status"].replace({"Job Offer": "Offer"})
    for col in YES_NO_COLS:
        df[col] = yes_no(df[col])

    for col in ["Applicant Count", "Application Method", "Connection Level",
                "Resume Version", "Cover Letter", "Follow-up Sent", "Role Type"]:
        blank = df[col].str.lower().isin(["", "unknown", "undefined"])
        df[col] = df[col].where(~blank, "Undefined")

    for col in ["Date Applied", "Posted Date", "Application Deadline", "Last Updated"]:
        df[col] = parse_dates(df[col])
    df["AI Fit Score"] = pd.to_numeric(df["AI Fit Score"], errors="coerce")

    # --- ghosting: clock starts at the deadline, or at the apply date if no deadline ---
    today = pd.Timestamp.today().normalize()
    ghost_date = (df["Application Deadline"] + pd.Timedelta(days=GHOST_DAYS_AFTER_DEADLINE)).where(
        df["Application Deadline"].notna(),
        df["Date Applied"] + pd.Timedelta(days=GHOST_DAYS_NO_DEADLINE),
    )
    still_open = df["Status"].isin(PENDING_STATUSES)
    is_ghosted = still_open & ghost_date.notna() & (ghost_date <= today)

    # --- tiered outcome: Ghosted < Rejected < Interview < Offer (else Pending) ---
    conds = [
        (df["Status"] == "Offer") | (df["Furthest Stage"] == "Offer"),
        (df["Status"] == "Interview Request") | (df["Furthest Stage"] == "Interview"),
        df["Status"] == "Rejected",
        is_ghosted,
    ]
    conds = [c.to_numpy(dtype=bool) for c in conds]
    df["Outcome"] = np.select(conds, ["Offer", "Interview", "Rejected", "Ghosted"], default="Pending")

    # --- derived factors ---
    gap = (df["Date Applied"] - df["Posted Date"]).dt.days
    df["Posting Age"] = (
        pd.cut(gap, bins=[-10000, 2, 7, 10000], labels=["0-2 days", "3-7 days", "8+ days"])
        .astype("string").fillna("Undefined")
    )
    df["Score Bucket"] = score_bucket(df["AI Fit Score"])
    return df


def ordered(present, preferred):
    present = list(present)
    first = [p for p in preferred if p in present]
    rest = sorted(p for p in present if p not in preferred)
    return first + rest


def response_table(df, col, preferred):
    rows = []
    for group in ordered(df[col].unique(), preferred):
        sub = df[df[col] == group]
        resolved = sub[sub["Outcome"] != "Pending"]
        responded = resolved[resolved["Outcome"] != "Ghosted"]
        interviews = resolved[resolved["Outcome"].isin(["Interview", "Offer"])]
        n_res = len(resolved)
        rows.append({
            col: group,
            "Applications": len(sub),
            "Still pending": len(sub) - n_res,
            "Resolved": n_res,
            "Got a response": len(responded),
            "Response rate %": round(100 * len(responded) / n_res, 1) if n_res else None,
            "Interview or offer": len(interviews),
            "Interview rate %": round(100 * len(interviews) / n_res, 1) if n_res else None,
        })
    return pd.DataFrame(rows).set_index(col)


def confidence_label(n):
    for minimum, label in CONFIDENCE_STEPS:
        if n >= minimum:
            return label
    return "Too few to tell"


def likely_range(k, n, z=1.96):
    """Wilson 95% interval for k successes out of n, as a 'lo-hi%' string."""
    if n == 0:
        return "n/a"
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return f"{100 * max(0, centre - half):.0f}-{100 * min(1, centre + half):.0f}%"


def rate_row(resolved, metric):
    """Counts and rate for one group of resolved applications."""
    n = len(resolved)
    k = int(resolved["Outcome"].isin(METRICS[metric]).sum())
    return n, k, (round(100 * k / n, 1) if n else None)


def factor_ranking(df, metric):
    """One row per factor option (Undefined left out), best rate first."""
    resolved = df[df["Outcome"] != "Pending"]
    _, _, base = rate_row(resolved, metric)
    rows = []
    for col, preferred in FACTORS:
        for group in ordered(resolved[col].unique(), preferred):
            if group == "Undefined":
                continue
            n, k, rate = rate_row(resolved[resolved[col] == group], metric)
            rows.append({
                "Factor": LABELS.get(col, col),
                "You control it": "Yes" if col in LEVERS else "No",
                "Option": group,
                "Resolved": n,
                f"{metric} %": rate,
                "Hits": f"{k} of {n}",
                "Likely range": likely_range(k, n),
                "vs. overall (pts)": round(rate - base, 1) if base is not None else None,
                "Confidence": confidence_label(n),
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return table, base
    table = table.sort_values([f"{metric} %", "Resolved"], ascending=[False, False])
    return table.reset_index(drop=True), base


def factor_takeaways(table, metric):
    """For each factor with 2+ comparable options, best vs worst, biggest gap first."""
    found, skipped = [], []
    for factor in [LABELS.get(c, c) for c, _ in FACTORS]:
        sub = table[(table["Factor"] == factor) & (table["Resolved"] >= MIN_TO_COMPARE)]
        if len(sub) < 2:
            skipped.append(factor)
            continue
        best, worst = sub.iloc[0], sub.iloc[-1]
        gap = best[f"{metric} %"] - worst[f"{metric} %"]
        weakest = confidence_label(min(best["Resolved"], worst["Resolved"]))
        found.append((gap, factor, best, worst, weakest, best["You control it"] == "Yes"))
    found.sort(key=lambda t: t[0], reverse=True)
    return found, skipped


def similar_applications(df, method, connection, score, count):
    """Loosen the match step by step until enough resolved applications are found."""
    near = (df["AI Fit Score"] - score).abs() <= SCORE_WINDOW
    m_method = df["Application Method"] == method
    m_conn = df["Connection Level"] == connection
    m_count = (df["Applicant Count"] == count) if count != "Any" else pd.Series(True, index=df.index)
    steps = [
        ("same method, connection, applicant count and a similar score", m_method & m_conn & m_count & near),
        ("same method, connection and a similar score", m_method & m_conn & near),
        ("same method and connection", m_method & m_conn),
        ("same connection level", m_conn),
        ("same application method", m_method),
        ("a similar score", near),
    ]
    if count == "Any":
        steps = steps[1:]
    for description, mask in steps:
        match = df[mask]
        if (match["Outcome"] != "Pending").sum() >= MIN_TO_COMPARE:
            return description, match
    return "all applications, since nothing closer had enough history", df


def choices(df, col, preferred):
    return [o for o in ordered(set(df[col].unique()) | set(preferred), preferred) if o != "Undefined"]


# ---------- Load ----------
raw = load_data()
missing = [c for c in REQUIRED if c not in raw.columns]
if missing:
    st.error(f"The sheet is missing these columns: {', '.join(missing)}")
    st.write("Columns found:", list(raw.columns))
    st.stop()

df = prepare(raw)
if df.empty:
    st.info("No applications found yet.")
    st.stop()

# ---------- Summary ----------
total = len(df)
resolved_n = int((df["Outcome"] != "Pending").sum())
responded_n = int(df["Outcome"].isin(["Rejected", "Interview", "Offer"]).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Applications", total)
c2.metric("Still pending", total - resolved_n)
c3.metric("Resolved", resolved_n)
c4.metric("Got a response", f"{responded_n} of {resolved_n}" if resolved_n else "n/a")

if resolved_n < SMALL_SAMPLE:
    st.info(
        f"Only {resolved_n} applications have resolved so far (goal: {SMALL_SAMPLE}+). "
        "Everything below is descriptive and exploratory, not a result."
    )

# ---------- What to do next ----------
st.header("What to do next")
st.caption(
    "Built from resolved applications only (pending ones are left out). "
    f"Confidence comes from how many resolved applications sit behind a rate: "
    + ", ".join(f"{lab} = {m}+" for m, lab in CONFIDENCE_STEPS)
    + ", fewer is 'Too few to tell'. Likely range is the 95% interval for the true rate."
)
metric = st.radio("Rank by", list(METRICS), horizontal=True,
                  help="Interview rate = interview or offer. Response rate = any reply, including rejections.")
ranking, base_rate = factor_ranking(df, metric)

if ranking.empty:
    st.info("No resolved applications yet, so there is nothing to rank.")
else:
    st.markdown(f"Overall {metric.lower()}: **{base_rate}%** across {resolved_n} resolved applications.")
    takeaways, skipped = factor_takeaways(ranking, metric)
    if takeaways:
        st.markdown("**Biggest differences so far**")
        for gap, factor, best, worst, conf, lever in takeaways:
            tag = " (you control this)" if lever else ""
            st.markdown(
                f"- **{factor}**{tag}: *{best['Option']}* {best[f'{metric} %']}% ({best['Hits']}) vs "
                f"*{worst['Option']}* {worst[f'{metric} %']}% ({worst['Hits']}), "
                f"a {gap:.0f}-point gap. Confidence: {conf}."
            )
    if skipped:
        st.caption(
            f"Not compared yet (needs 2+ options with {MIN_TO_COMPARE}+ resolved applications each): "
            + ", ".join(skipped) + "."
        )
    with st.expander("Full ranking of every option", expanded=not takeaways):
        st.dataframe(ranking, hide_index=True, width="stretch")

# ---------- Pre-apply check ----------
st.subheader("Before you apply: how did similar applications go?")
p1, p2, p3, p4 = st.columns(4)
method_choice = p1.selectbox("Application method", choices(df, "Application Method", FACTORS[1][1]))
conn_choice = p2.selectbox("Connection level", choices(df, "Connection Level", FACTORS[0][1]))
count_choice = p3.selectbox("Applicant count", ["Any"] + choices(df, "Applicant Count", FACTORS[2][1]))
score_choice = p4.number_input("AI Fit Score", min_value=0, max_value=100, value=70, step=1)
q1, q2, q3 = st.columns(3)
role_choice = q1.selectbox("Role type", ["Any"] + choices(df, "Role Type", ROLE_TYPES))
resume_choice = q2.selectbox("Resume version", ["Any"] + choices(df, "Resume Version", []))
cover_choice = q3.selectbox("Cover letter", ["Any", "Yes", "No"])

basis, similar = similar_applications(df, method_choice, conn_choice, score_choice, count_choice)
similar_resolved = similar[similar["Outcome"] != "Pending"]
n_sim, k_int, int_rate = rate_row(similar_resolved, "Interview rate")
_, k_resp, resp_rate = rate_row(similar_resolved, "Response rate")

note = f" (within {SCORE_WINDOW} points of {score_choice})" if "score" in basis else ""
st.markdown(f"Compared against applications with **{basis}**{note}.")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Similar applications", f"{n_sim} resolved", help=f"{len(similar)} including pending ones")
s2.metric("Got a response", f"{k_resp} of {n_sim} ({resp_rate}%)" if n_sim else "n/a")
s3.metric("Interview or offer", f"{k_int} of {n_sim} ({int_rate}%)" if n_sim else "n/a")
s4.metric("Confidence", confidence_label(n_sim))

overall_int = rate_row(df[df["Outcome"] != "Pending"], "Interview rate")[2]
if n_sim and overall_int is not None and confidence_label(n_sim) != "Too few to tell":
    diff = int_rate - overall_int
    if abs(diff) < 5:
        st.write(f"About the same as your overall interview rate ({overall_int}%).")
    else:
        word = "better" if diff > 0 else "worse"
        st.write(f"That is {abs(diff):.0f} points {word} than your overall interview rate ({overall_int}%).")
elif n_sim:
    st.write("Too few similar applications to say whether this beats your usual results. Treat it as a hint.")

with st.expander("Each choice on its own"):
    resolved_all = df[df["Outcome"] != "Pending"]
    picked_bucket = score_bucket(pd.Series([score_choice]))[0]
    picks = [("Application Method", method_choice), ("Connection Level", conn_choice),
             ("Score Bucket", picked_bucket)]
    for col, value in [("Applicant Count", count_choice), ("Role Type", role_choice),
                       ("Resume Version", resume_choice), ("Cover Letter", cover_choice)]:
        if value != "Any":
            picks.append((col, value))
    rows = []
    for col, value in picks:
        sub = resolved_all[resolved_all[col] == value]
        n, k_r, r_rate = rate_row(sub, "Response rate")
        _, k_i, i_rate = rate_row(sub, "Interview rate")
        rows.append({"Factor": LABELS.get(col, col), "Your choice": value, "Resolved": n,
                     "Response rate %": r_rate, "Interview rate %": i_rate,
                     "Confidence": confidence_label(n)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with st.expander(f"The {len(similar)} similar applications"):
    show = similar.sort_values("Date Applied", ascending=False).copy()
    show["Date Applied"] = show["Date Applied"].dt.date
    st.dataframe(show[["Company", "Role", "Date Applied", "Outcome", "AI Fit Score",
                       "Application Method", "Connection Level", "Applicant Count",
                       "Role Type", "Resume Version", "Cover Letter"]],
                 hide_index=True, width="stretch")

# ---------- 1. Status funnel ----------
st.subheader("Application Funnel")
status_counts = df["Status"].value_counts().reindex(FUNNEL_ORDER, fill_value=0)
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.bar(status_counts.index, status_counts.values, color="#4C7FA0")
ax1.set_ylabel("Applications")
plt.xticks(rotation=20)
st.pyplot(fig1)
plt.close(fig1)

# ---------- 2. Fit score distribution ----------
st.subheader("AI Fit Score Distribution")
scores = df["AI Fit Score"].dropna()
if len(scores) > 0:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(scores, bins=10, color="#6B9B6E", edgecolor="white")
    ax2.set_xlabel("Fit Score")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)
    plt.close(fig2)
else:
    st.info("No fit scores yet.")

# ---------- 3. Does fit score predict outcome? ----------
st.subheader("Does a higher Fit Score predict a better outcome?")
scored = df[df["Score Bucket"] != "Undefined"]
if len(scored) > 0:
    outcome_cols = ["Offer", "Interview", "Rejected", "Ghosted", "Pending"]
    counts = pd.crosstab(scored["Score Bucket"], scored["Outcome"]).reindex(columns=outcome_cols, fill_value=0)
    counts = counts.reindex([b for _, b in SCORE_BINS if b in counts.index])
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    counts.plot(
        kind="bar", stacked=True, ax=ax3,
        color=["#2E7D4F", "#4C9A6B", "#C65A52", "#8C6D62", "#B0B0B0"],
    )
    ax3.set_ylabel("Applications")
    ax3.set_xlabel("AI Fit Score bucket")
    plt.xticks(rotation=0)
    st.pyplot(fig3)
    plt.close(fig3)
else:
    st.info("Not enough scored data yet.")

# ---------- 4. What gets me seen? ----------
st.subheader("What gets my application seen?")
st.caption(
    "Response rate = share of resolved applications that got any reply (rejection, interview, or offer). "
    f"Ghosted = no reply {GHOST_DAYS_AFTER_DEADLINE} days after the deadline "
    f"(or {GHOST_DAYS_NO_DEADLINE} days after applying when no deadline is listed). "
    "Pending applications are excluded from the rates. Check the counts before the percentages. "
    "Follow-up Sent leans optimistic: you can only follow up on applications that were still open."
)

for col, preferred in FACTORS:
    st.markdown(f"**By {LABELS.get(col, col)}**")
    st.dataframe(response_table(df, col, preferred))

# ---------- 5. Recently updated ----------
st.subheader("Recently Updated")
recent = df.sort_values("Last Updated", ascending=False).head(5).copy()
recent["Last Updated"] = recent["Last Updated"].dt.date
st.dataframe(recent[["Company", "Role", "Status", "Outcome", "AI Fit Score", "Last Updated"]])
