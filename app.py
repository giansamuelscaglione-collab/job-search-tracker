import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR-K2bPgkFpcavReVoXH-K4HQWy6QxN12o6jWB5ywewq32TEo0qG9JaZpJgCMkVOiNTZB5P0Rp3UnuL/pub?gid=1301135712&single=true&output=csv"

# ---------- Settings you can change ----------
GHOST_DAYS_AFTER_DEADLINE = 14   # ghosted if still no reply this many days after the deadline
GHOST_DAYS_NO_DEADLINE = 21      # no deadline listed: ghosted this many days after applying
SMALL_SAMPLE = 20                # show a caution until this many applications have resolved

PENDING_STATUSES = {"No Response Yet", "Application Confirmed", ""}
FUNNEL_ORDER = ["No Response Yet", "Application Confirmed", "Interview Request", "Offer", "Rejected"]

REQUIRED = ["Company", "Role", "Date Applied", "Status", "AI Fit Score", "Last Updated"]
OPTIONAL = [
    "Posted Date", "Application Deadline", "Applicant Count",
    "Application Method", "Connection Level", "Furthest Stage",
]

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

    for col in ["Company", "Role", "Status", "Applicant Count",
                "Application Method", "Connection Level", "Furthest Stage"]:
        df[col] = clean_text(df[col])
    df["Status"] = df["Status"].replace({"Job Offer": "Offer"})

    for col in ["Applicant Count", "Application Method", "Connection Level"]:
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
    df["Score Bucket"] = (
        pd.cut(df["AI Fit Score"], bins=[-1, 79.99, 89.99, 100], labels=["Under 80", "80-89", "90+"])
        .astype("string").fillna("Undefined")
    )
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
    counts = counts.reindex([b for b in ["Under 80", "80-89", "90+"] if b in counts.index])
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
    "Pending applications are excluded from the rates. Check the counts before the percentages."
)

factors = [
    ("Connection Level", ["None", "Know someone there", "Referral submitted",
                          "Spoke with recruiter or hiring manager", "Undefined"]),
    ("Application Method", ["LinkedIn", "Indeed", "Company Site", "Direct Email", "School Portal", "Other", "Undefined"]),
    ("Applicant Count", ["<25", "Under 25", "25-100", "100+", ">100", "Undefined"]),
    ("Posting Age", ["0-2 days", "3-7 days", "8+ days", "Undefined"]),
    ("Score Bucket", ["Under 80", "80-89", "90+", "Undefined"]),
]
labels = {"Posting Age": "Days between posting and applying", "Score Bucket": "AI Fit Score"}
for col, preferred in factors:
    st.markdown(f"**By {labels.get(col, col)}**")
    st.dataframe(response_table(df, col, preferred))

# ---------- 5. Recently updated ----------
st.subheader("Recently Updated")
recent = df.sort_values("Last Updated", ascending=False).head(5).copy()
recent["Last Updated"] = recent["Last Updated"].dt.date
st.dataframe(recent[["Company", "Role", "Status", "Outcome", "AI Fit Score", "Last Updated"]])
