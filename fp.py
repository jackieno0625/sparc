"""
SPARC — Financial Projection Dashboard
Streamlit application for clinics and public health programs.

Clean version: duplicates removed, helpers unified, page config moved first,
callback functions lifted out of render loops, Example Scenario split into its own tab.
"""

import io
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pkgutil
import streamlit as st

# ─────────────────────────────────────────────
# Page config — MUST be the very first st call
# ─────────────────────────────────────────────
st.set_page_config(page_title="Financial Projection Dashboard", layout="wide")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CSV_FILENAME = "Compiled Fee Assessment.csv"
CSV_PATH = Path(__file__).parent / CSV_FILENAME

INSURERS = ["Uninsured", "Medicaid", "Healthy Blue", "Trillium", "Aetna", "Medicare"]

DEFAULT_GROUPS = [
    "Primary Care",
    "Adult Health",
    "STD-related",
    "BCCC-related",
    "Other",
]


# ─────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────
@st.cache_data
def load_fee_csv(path: Path) -> pd.DataFrame:
    """
    Load the bundled fee-schedule CSV.
    Tries the file next to fp.py first, then package data.
    Stops the app with a clear message if neither is found.
    """
    if path.exists():
        try:
            return pd.read_csv(path, dtype=str)
        except Exception as exc:
            st.error(f"Found {path} but could not read it as CSV: {exc}")
            st.stop()

    try:
        data = pkgutil.get_data(__name__, CSV_FILENAME)
        if data:
            return pd.read_csv(io.BytesIO(data), dtype=str)
    except Exception as exc:
        st.error(f"Found package data for {CSV_FILENAME} but could not read it: {exc}")
        st.stop()

    st.error(
        f"`{CSV_FILENAME}` was not found next to `fp.py` or as package data. "
        "Place the CSV in the same folder as `fp.py` (exact filename required), then reload."
    )
    st.stop()


reim_df: pd.DataFrame = load_fee_csv(CSV_PATH)


# ─────────────────────────────────────────────
# Fee-schedule helpers  (defined once, used everywhere)
# ─────────────────────────────────────────────
def _parse_currency(value) -> float:
    """Safely coerce a currency string or number to float."""
    try:
        return float(str(value).replace("$", "").replace(",", "").strip() or 0.0)
    except (ValueError, TypeError):
        return 0.0


def get_insurer_pay(payer_name: str, cpt_code: str, df: pd.DataFrame) -> float:
    """Return the insurer's reimbursement for a CPT code. Uninsured always returns 0."""
    if payer_name.lower() == "uninsured":
        return 0.0
    rows = df[df["CPT Code"].astype(str) == str(cpt_code)]
    if rows.empty:
        return 0.0
    row = rows.iloc[0]
    if payer_name in df.columns:
        return _parse_currency(row.get(payer_name, 0.0))
    for col in df.columns:
        if col.strip().lower() == payer_name.strip().lower():
            return _parse_currency(row.get(col, 0.0))
    return 0.0


def get_practice_fee(cpt_code: str, df: pd.DataFrame) -> float:
    """Return the practice fee for a CPT code."""
    rows = df[df["CPT Code"].astype(str) == str(cpt_code)]
    if rows.empty:
        return 0.0
    return _parse_currency(rows.iloc[0].get("Practice Fee", 0.0))


def per_patient_revenue(
    cpt_codes: list,
    payer_probs: np.ndarray,
    patient_share_frac: float,
    df: pd.DataFrame,
) -> float:
    """
    Compute expected revenue per patient given:
      - cpt_codes: list of CPT codes applied per visit
      - payer_probs: probability weights for each insurer (must sum to 1)
      - patient_share_frac: fraction of shortfall the patient pays (0-1)
      - df: fee schedule DataFrame
    """
    total = 0.0
    for payer, prob in zip(INSURERS, payer_probs):
        payer_sum = 0.0
        for cpt in cpt_codes:
            ins_pay = get_insurer_pay(payer, cpt, df)
            prac_fee = get_practice_fee(cpt, df)
            shortfall = max(0.0, prac_fee - ins_pay)
            payer_sum += ins_pay + patient_share_frac * shortfall
        total += prob * payer_sum
    return total


def build_scenario_table(
    cpt_codes: list,
    payer_probs: np.ndarray,
    population: int,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame showing total revenue at 0 / 50 / 75 / 100 % patient share."""
    rows = []
    for share in [0.0, 0.50, 0.75, 1.0]:
        rev = per_patient_revenue(cpt_codes, payer_probs, share, df) * population
        rows.append({
            "Patient Share (%)": f"{int(share * 100)}%",
            "Total Revenue ($)": f"${rev:,.2f}",
        })
    return pd.DataFrame(rows).set_index("Patient Share (%)")


# ─────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────
def safe_float(key: str, default: float = 0.0) -> float:
    try:
        return float(st.session_state.get(key, default))
    except (ValueError, TypeError):
        return default


def safe_int(key: str, default: int = 0) -> int:
    try:
        return int(st.session_state.get(key, default))
    except (ValueError, TypeError):
        return default


def group_key(gid: str, field: str) -> str:
    """Namespace a session-state key to a specific service group."""
    return f"{gid}__{field}"


# ─────────────────────────────────────────────
# Callback functions (defined at module level — never inside loops)
# ─────────────────────────────────────────────
def _add_fixed_item():
    name = st.session_state.get("fixed_new_name", "").strip()
    amt = safe_float("fixed_new_amount")
    if not name:
        st.session_state["_fixed_warning"] = "Please enter a name before adding."
        return
    st.session_state.setdefault("fixed_items", []).append({"item": name, "annual_cost": amt})
    st.session_state["fixed_new_name"] = ""
    st.session_state["fixed_new_amount"] = 0.0
    st.session_state.pop("_fixed_warning", None)


def _add_misc_item():
    name = st.session_state.get("misc_new_name", "").strip()
    amt = safe_float("misc_new_amount")
    if not name:
        st.session_state["_misc_warning"] = "Please enter a name before adding."
        return
    st.session_state.setdefault("misc_items", []).append({"item": name, "annual_cost": amt})
    st.session_state["misc_new_name"] = ""
    st.session_state["misc_new_amount"] = 0.0
    st.session_state.pop("_misc_warning", None)


def _make_group_name_callback(gid: str, name_key: str):
    """Return a callback that syncs a group's display name into session state."""
    def _cb():
        for g in st.session_state.get("service_groups", []):
            if g.get("id") == gid:
                g["name"] = st.session_state.get(name_key, "")
                break
        st.rerun()
    return _cb


# ─────────────────────────────────────────────
# CPT display list helpers
# ─────────────────────────────────────────────
def cpt_display_options(df: pd.DataFrame) -> list:
    desc = df.get("Description", pd.Series([""] * len(df))).astype(str)
    return (df["CPT Code"].astype(str) + " — " + desc).tolist()


def cpt_display_to_code(df: pd.DataFrame) -> dict:
    return {
        f"{row['CPT Code']} — {row.get('Description', '')}": row["CPT Code"]
        for _, row in df.iterrows()
    }


# ─────────────────────────────────────────────
# Logo + title
# ─────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; justify-content:center; margin-top:10px; margin-bottom:20px;">
        <img src="https://i.imgur.com/40QPfA3.png" width="250" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("Financial Projection Dashboard")

# ─────────────────────────────────────────────
# Sidebar: mode toggles
# ─────────────────────────────────────────────
st.sidebar.header("Input Modes")
pop_mode = st.sidebar.radio(
    "Patient Population Mode",
    options=["Simple", "Advanced"],
    index=0,
    help="Choose Simple for minimal inputs or Advanced for detailed population controls.",
)
cost_mode = st.sidebar.radio(
    "Total Cost Mode",
    options=["Simple", "Advanced"],
    index=0,
    help="Choose Simple for minimal inputs or Advanced for detailed cost controls.",
)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_main, tab_instructions, tab_example = st.tabs(["Main", "Instructions", "Example Scenario"])


# ─────────────────────────────────────────────
# Instructions tab  (original text preserved verbatim)
# ─────────────────────────────────────────────
with tab_instructions:
    st.subheader("Instructions")
    st.markdown(
        """
        Welcome to **SPARC**, a financial projection tool for clinics and public health programs.  
        This dashboard helps estimate reimbursement, total cost, and net profit based on user input-based patient population, service mix, and operating costs.

        ---

        ## 🔍 How SPARC Works — A Quick Overview
        You control two parts of the model:

        1. **Patient Population**  
           Describes who your patients are and what services they receive.
        2. **Total Cost**  
           Describes what it costs your clinic to provide services.

        SPARC then uses the embedded **fee schedule CSV** to calculate insurer reimbursement for each CPT code and combines that with patient cost-share settings to compute revenue and net profit.
        This fee schedule compiles the most commonly used CPT codes from publically available information from Medicaid, Medicare, Healthy Blue,
        Trillium, and Aetna under the Transparency in Coverage (TiC) federal regulation.

        ---

        ## 🧑‍⚕️ Patient Population — Simple Mode
        Use this when you want a **straightforward, quick estimate**.

        **You provide:**
        - A **single total population number**
        - An **insurance payer mix** (percentages that must sum to 100%)
        - Up to **4 CPT codes** that *every patient* in this simple population receives
        - One **global patient shortfall slider**:  
          "What % of the gap between insurer reimbursement and practice fee do patients pay?"

        **SPARC does the rest:**
        - Pulls insurer payment for each CPT from the fee schedule    
        - Applies your patient-pay %  
        - Calculates total revenue → subtracts costs → net profit  
        - Shows scenario comparisons for patient payment % at 0%, 50%, 75%, 100%

        ---

        ## 🧑‍⚕️ Patient Population — Advanced Mode
        Use this when you have **multiple service lines** or more realistic operational complexity.

        In Advanced mode, you build **Service Groups**, each representing a category such as:

        - Primary Care  
        - Adult Health  
        - Pediatrics  
        - STD Services  
        - BCCC / Breast & Cervical Cancer Control  
        - Behavioral Health  
        - Procedures  
        - "Other"

        **For each group, you can define:**
        - A **population size**
        - Up to **4 CPT codes** specific to that group  
        - An **insurance payer mix** for that group (must sum to 100%)  
        - A **group-specific patient-share slider**  
        - Groups can be **added, renamed, or removed**

        **SPARC calculates per-group:**
        - Expected per-patient revenue (payer-weighted)
        - Total revenue for the group  
        - Fiscal contribution to overall clinic revenue

        **Then it sums all groups** to produce overall clinic reimbursement.

        ---

        ## 💰 Total Cost — Simple Mode
        Use this when you know your annual cost number already.

        **You provide:**
        - One **annual total cost** (e.g., \$1,200,000)

        ---

        ## 💰 Total Cost — Advanced Mode
        Use this when your cost structure is detailed or changes frequently.

        **You can enter:**
        ### 1) Fixed Costs  
        Examples: EHR, billing software, admin subscriptions, utilities, rent, equipment.

        Add items one by one — each includes a name and annual amount.

        ### 2) Provider Payroll  
        SPARC will compute:
        
        ```annual payroll = (# providers) × (hours/week × weeks/year × hourly pay)```
        

        ### 3) Miscellaneous Costs  
        For unsorted operating costs (training, supplies, outreach, etc.)

        **SPARC then sums everything** to create the clinic's total annual cost.

        ---

        ## 📊 What Appears in "Model Outputs"
        Depending on Simple or Advanced Population mode, SPARC shows:

        - **Total reimbursement**  
        - **Total cost**  
        - **Net profit**  
        - **Per-group revenue table (Advanced Population)**  
        - **Net profit scenario table (0%, 50%, 75%, 100% patient share)**  

        Note that nothing displays until you enter valid inputs.

        ---

        ## 📁 About the Fee Schedule (CSV)
        - SPARC automatically loads the **Compiled Fee Assessment.csv** included with the app.  
        - If missing, SPARC will not execute properly.
        - The full CSV includes payer-specific reimbursement for each CPT.

        ---

        ## Need help?
        Each section includes a **? help icon** explaining how that field works.
        Please contact the developer through SPARC's website if problems persist.

        """
    )


# ─────────────────────────────────────────────
# Example Scenario tab  (original text preserved verbatim)
# ─────────────────────────────────────────────
with tab_example:
    st.subheader("Example Scenario — Complete Walkthrough")
    st.markdown(
        """
        ## 📝 Example Scenario — Complete Walkthrough

        **Scenario:**  
        A clinic wants to estimate net revenue for two service lines:  
        - **Primary Care** (Routine visits)  
        - **STD Clinic** (Testing & treatment)

        ### Step 1 — Choose modes  
        - Patient Population Mode → **Advanced**  
        - Total Cost Mode → **Advanced**

        ### Step 2 — Create groups  
        SPARC loads default groups automatically.  
        Edit them to:

        **Group 1: Primary Care**  
        - Population: **1200**  
        - CPT codes: **99213**, **36415**, **3008F**  
        - Payer mix:  
          - Medicaid 40%  
          - Healthy Blue 20%  
          - Medicare 20%  
          - Aetna 10%  
          - Uninsured 10%  
        - Patient-share slider: **20%**

        **Group 2: STD Clinic**  
        - Population: **400**  
        - CPT codes: **87491**, **87591**, **99214**  
        - Payer mix:  
          - Medicaid 60%  
          - Healthy Blue 30%  
          - Uninsured 10%  
        - Patient-share slider: **0%** (patients do not pay)

        ### Step 3 — Enter Advanced Total Costs  
        - Fixed costs:  
          - EHR: \$60,000  
          - Rent: \$120,000  
          - IT Services: \$45,000  
        - Provider payroll:  
          - 5 providers × 36 hrs/week × 48 weeks × \$60/hr = **\$518,400**  
        - Misc costs:  
          - Supplies: \$12,000  
          - Program outreach: \$8,000  

        Total Cost automatically becomes:  
        **\$763,400**

        ### Step 4 — View Model Outputs  
        SPARC shows:

        **Reimbursement by group:**  
        - Primary Care: Total \$119,836.32  
        - STD Clinic: Total \$57,350.00 

        **Combined reimbursement:**  
        \$177,186.32  

        **Net Profit:**  
        \$(177,186.32 − 763,400) = \$-586,213.68

        **Scenario Table:**  
        Shows how net profit would change if *all groups* shifted to paying 0%, 50%, 75%, or 100% of shortfalls.

        """
    )


# ─────────────────────────────────────────────
# Main tab
# ─────────────────────────────────────────────
with tab_main:
    st.markdown("Open the sidebar on the left to toggle between Simple and Advanced modeling options.")

    col1, _divider, col2 = st.columns([0.6, 0.02, 0.38])

    with _divider:
        st.markdown(
            "<div style='border-left:1px solid #ccc; height:100%; margin:auto;'></div>",
            unsafe_allow_html=True,
        )

    # ──────────────────────────────────────────
    # LEFT COLUMN: Patient Population
    # ──────────────────────────────────────────
    with col1:
        st.subheader("Create a Patient Population")
        st.write("Mode:", f"**{pop_mode}**")

        # ── Simple population ──────────────────
        if pop_mode == "Simple":
            st.markdown("#### Patient Population Size")
            st.session_state.setdefault("simple_population", 0)
            st.number_input(
                "Enter patient population:",
                min_value=0,
                step=1,
                key="simple_population",
                help="Type the total number of patients you want to model.",
            )
            st.write("You entered:", safe_int("simple_population"))

            st.markdown("#### Insurer Mix")
            for name in INSURERS:
                st.session_state.setdefault(f"pct_{name}", 0.0)
            for name in INSURERS:
                st.number_input(
                    f"{name} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    key=f"pct_{name}",
                    help="Enter insurer percentages for the population. They must total 100%.",
                )

            pct_values = np.array([safe_float(f"pct_{n}") for n in INSURERS])
            total_pct = float(np.round(pct_values.sum(), 6))
            st.markdown(f"**Total: {total_pct:.2f}%**")
            if abs(total_pct - 100.0) > 1e-6:
                st.error("Insurer percentages MUST add up to 100%. Please adjust the values above.")

            st.markdown("#### Select CPT Codes")
            cpt_options = cpt_display_options(reim_df)
            st.session_state.setdefault("selected_cpts", [])
            selected = st.multiselect(
                "Select up to 4 CPT codes (these will be applied to every patient)",
                options=cpt_options,
                key="selected_cpts",
                help="Search CPTs by code or description. Pick up to 4.",
            )
            if len(selected) > 4:
                st.warning("Please select at most 4 CPT codes. Only the first 4 will be used.")

            if len(selected) == 0:
                st.info("Select 1–4 CPT codes to enable reimbursement calculations.")
            else:
                st.markdown("#### Reimbursement via Patients")
                st.session_state.setdefault("pct_patient_share", 0)
                st.slider(
                    "Patient pays what percent of the shortfall after insurance reimburses? (X%)",
                    min_value=0,
                    max_value=100,
                    step=1,
                    key="pct_patient_share",
                    help="If insurer pays less than the practice fee, patient covers X% of that gap.",
                )

        # ── Advanced population ────────────────
        else:
            if "service_groups" not in st.session_state:
                st.session_state["service_groups"] = [
                    {"id": str(uuid.uuid4()), "name": name} for name in DEFAULT_GROUPS
                ]

            cols = st.columns([1, 1])
            with cols[0]:
                st.markdown("**Manage groups**")
                if st.button("Add new empty group"):
                    st.session_state["service_groups"].append(
                        {"id": str(uuid.uuid4()), "name": "New Group"}
                    )
            with cols[1]:
                st.markdown("**Actions**")
                if st.button("Reset to defaults"):
                    st.session_state["service_groups"] = [
                        {"id": str(uuid.uuid4()), "name": name} for name in DEFAULT_GROUPS
                    ]
                    for k in list(st.session_state.keys()):
                        if "__" in k:
                            del st.session_state[k]

            cpt_options = cpt_display_options(reim_df)
            display_map = cpt_display_to_code(reim_df)

            for grp in list(st.session_state["service_groups"]):
                gid = grp["id"]
                with st.expander(grp.get("name", "Group"), expanded=False):

                    # Group name
                    nk = group_key(gid, "name")
                    st.session_state.setdefault(nk, grp.get("name", ""))
                    st.text_input(
                        "Group name",
                        key=nk,
                        on_change=_make_group_name_callback(gid, nk),
                    )

                    # Population
                    pk = group_key(gid, "population")
                    st.session_state.setdefault(pk, 0)
                    st.number_input(
                        "Population (integer)",
                        min_value=0,
                        step=1,
                        key=pk,
                        help="Number of patients in this service group.",
                    )

                    # CPT codes
                    sk = group_key(gid, "selected_cpts")
                    st.session_state.setdefault(sk, [])
                    st.multiselect(
                        "Select up to 4 CPT codes for this group",
                        options=cpt_options,
                        key=sk,
                        help="These CPTs will be applied to visits in this group (per-visit).",
                    )

                    # Patient-share slider
                    shk = group_key(gid, "patient_share_pct")
                    st.session_state.setdefault(shk, 0)
                    st.slider(
                        "Patient pays what % of shortfall (group-level)",
                        min_value=0,
                        max_value=100,
                        key=shk,
                        help="This percent is applied to each CPT shortfall for this group's patients.",
                    )

                    # Payer mix
                    st.markdown("Payer mix for this group (must sum to 100%)")
                    payer_vals = []
                    for payer in INSURERS:
                        kk = group_key(gid, f"pct_{payer}")
                        st.session_state.setdefault(kk, 0.0)
                        st.number_input(
                            f"{payer} (%)",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.5,
                            key=kk,
                        )
                        payer_vals.append(safe_float(kk))

                    payer_total = sum(payer_vals)
                    if payer_total <= 0:
                        st.warning("Payer mix for this group sums to 0 — enter percentages before computing.")
                    elif abs(payer_total - 100.0) > 1e-6:
                        st.warning(f"Payer mix sums to {payer_total:.2f}%. Please make it sum to 100%.")

                    # Remove group
                    if st.button("Remove this group", key=group_key(gid, "remove")):
                        st.session_state["service_groups"] = [
                            g for g in st.session_state["service_groups"] if g["id"] != gid
                        ]
                        for k in list(st.session_state.keys()):
                            if k.startswith(f"{gid}__"):
                                del st.session_state[k]
                        st.rerun()

    # ──────────────────────────────────────────
    # RIGHT COLUMN: Total Cost
    # ──────────────────────────────────────────
    with col2:
        st.subheader("Total Cost")
        st.write("Mode:", f"**{cost_mode}**")

        if cost_mode == "Simple":
            st.markdown("### Simple Total Cost Input")
            st.number_input(
                "Enter total cost ($):",
                min_value=0.0,
                step=1000.0,
                key="simple_net_cost",
                help="This represents all costs combined — overhead + variable + fixed.",
            )
            st.markdown(f"## Total Cost = **${safe_float('simple_net_cost'):,.2f}**")

        else:
            st.markdown("### Advanced Cost Inputs")

            # ── Fixed costs ────────────────────
            st.markdown("#### Fixed costs (subscriptions, maintenance, IT, rent, etc.)")
            st.session_state.setdefault("fixed_items", [])
            st.session_state.setdefault("fixed_new_name", "")
            st.session_state.setdefault("fixed_new_amount", 0.0)

            st.number_input(
                "New fixed cost annual amount ($)",
                min_value=0.0,
                step=100.0,
                key="fixed_new_amount",
            )
            st.text_input("New fixed cost name", key="fixed_new_name")
            st.button("Add fixed cost item", on_click=_add_fixed_item)
            if st.session_state.get("_fixed_warning"):
                st.warning(st.session_state["_fixed_warning"])

            fixed_total = 0.0
            to_remove_fixed = None
            if st.session_state["fixed_items"]:
                st.write("Current fixed cost items:")
                for idx, entry in enumerate(list(st.session_state["fixed_items"])):
                    c1, c2, c3 = st.columns([4, 1, 1])
                    c1.markdown(f"**{entry['item']}**")
                    c2.markdown(f"${entry['annual_cost']:,.2f}")
                    if c3.button("Remove", key=f"remove_fixed_{idx}"):
                        to_remove_fixed = idx
                    fixed_total += entry["annual_cost"]
                if to_remove_fixed is not None:
                    st.session_state["fixed_items"].pop(to_remove_fixed)
            else:
                st.info("No fixed cost items added yet.")
            st.write(f"Fixed costs total: **${fixed_total:,.2f}**")
            st.markdown("---")

            # ── Provider payroll ───────────────
            st.markdown("#### Provider payroll")
            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1:
                st.number_input("Number of providers", min_value=0, value=0, step=1, key="adv_num_providers")
            with pc2:
                st.number_input("Hours/provider/week", min_value=0.0, value=0.0, step=1.0, key="adv_hours_week")
            with pc3:
                st.number_input("Paid work weeks/year", min_value=0, value=0, step=1, key="adv_weeks_year")
            with pc4:
                st.number_input("Hourly pay ($)", min_value=0.0, value=0.0, step=1.0, key="adv_hourly_pay")

            annual_per_provider = safe_float("adv_hours_week") * safe_float("adv_weeks_year") * safe_float("adv_hourly_pay")
            provider_total = safe_int("adv_num_providers") * annual_per_provider
            st.write(f"Provider payroll total: **${provider_total:,.2f}** (${annual_per_provider:,.2f} per provider / year)")
            st.markdown("---")

            # ── Miscellaneous costs ────────────
            st.markdown("#### Miscellaneous costs")
            st.session_state.setdefault("misc_items", [])
            st.session_state.setdefault("misc_new_name", "")
            st.session_state.setdefault("misc_new_amount", 0.0)

            st.text_input("New miscellaneous cost name", key="misc_new_name")
            st.number_input("New misc cost annual amount ($)", min_value=0.0, value=0.0, step=100.0, key="misc_new_amount")
            st.button("Add misc cost item", on_click=_add_misc_item)
            if st.session_state.get("_misc_warning"):
                st.warning(st.session_state["_misc_warning"])

            misc_total = 0.0
            to_remove_misc = None
            if st.session_state["misc_items"]:
                st.write("Current miscellaneous cost items:")
                for idx, entry in enumerate(list(st.session_state["misc_items"])):
                    c1, c2, c3 = st.columns([4, 1, 1])
                    c1.markdown(f"**{entry['item']}**")
                    c2.markdown(f"${entry['annual_cost']:,.2f}")
                    if c3.button("Remove", key=f"remove_misc_{idx}"):
                        to_remove_misc = idx
                    misc_total += entry["annual_cost"]
                if to_remove_misc is not None:
                    st.session_state["misc_items"].pop(to_remove_misc)
            else:
                st.info("No miscellaneous cost items added yet.")
            st.write(f"Misc costs total: **${misc_total:,.2f}**")
            st.markdown("---")

            # ── Advanced cost total ────────────
            net_cost_advanced = fixed_total + provider_total + misc_total
            st.markdown(f"## Total Cost = **${net_cost_advanced:,.2f}**")
            st.session_state["net_cost_advanced"] = float(net_cost_advanced)

    # ──────────────────────────────────────────
    # MODEL OUTPUTS
    # ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("# Model Outputs")

    # Resolve total cost
    if cost_mode == "Advanced":
        total_cost = float(st.session_state.get("net_cost_advanced", 0.0))
    else:
        total_cost = safe_float("simple_net_cost")

    display_map = cpt_display_to_code(reim_df)
    grand_revenue = 0.0
    grand_population = 0

    # ── Simple population outputs ──────────────
    if pop_mode == "Simple":
        raw_displays = st.session_state.get("selected_cpts", [])[:4]
        selected_cpts = [display_map[d] for d in raw_displays if d in display_map]

        if not selected_cpts:
            st.info("No CPTs selected yet — select up to 4 CPT codes in the Patient Population panel to see final outputs.")
        else:
            payer_pcts = np.array([safe_float(f"pct_{p}") for p in INSURERS])
            if payer_pcts.sum() == 0:
                st.error("Payer distribution sums to 0. Please enter insurer percentages in the Patient Population panel.")
            else:
                payer_probs = payer_pcts / payer_pcts.sum()
                population = safe_int("simple_population")
                patient_share = safe_float("pct_patient_share") / 100.0

                grand_revenue = per_patient_revenue(selected_cpts, payer_probs, patient_share, reim_df) * population
                grand_population = population

                st.write(f"- Patient shortfall share (slider): **{patient_share * 100:.0f}%**")
                st.write(f"- Population: **{population:,d}**")
                st.write(f"- Total Reimbursement: **${grand_revenue:,.2f}**")
                st.write(f"- Total Cost: **${total_cost:,.2f}**")
                st.markdown(f"## Final Net profit: **${grand_revenue - total_cost:,.2f}**")

                st.markdown("#### Quick scenario comparison (patient pays X% of shortfall)")
                st.table(build_scenario_table(selected_cpts, payer_probs, population, reim_df))

    # ── Advanced population outputs ────────────
    else:
        groups = st.session_state.get("service_groups", [])
        if not groups:
            st.info("No service groups configured in Advanced Patient Population. Add groups in the Advanced panel.")
        else:
            group_rows = []
            for grp in groups:
                gid = grp["id"]
                gname = st.session_state.get(group_key(gid, "name"), grp.get("name", "Group"))
                pop_val = safe_int(group_key(gid, "population"))
                sel_displays = st.session_state.get(group_key(gid, "selected_cpts"), [])
                sel_cpts = [display_map[d] for d in sel_displays if d in display_map]
                payer_pct_arr = np.array([safe_float(group_key(gid, f"pct_{p}")) for p in INSURERS])
                share_frac = safe_float(group_key(gid, "patient_share_pct")) / 100.0

                if pop_val > 0 and sel_cpts and payer_pct_arr.sum() > 0:
                    probs = payer_pct_arr / payer_pct_arr.sum()
                    grp_revenue = per_patient_revenue(sel_cpts, probs, share_frac, reim_df) * pop_val
                else:
                    grp_revenue = 0.0

                group_rows.append({
                    "name": gname,
                    "population": pop_val,
                    "total_revenue": grp_revenue,
                })
                grand_revenue += grp_revenue
                grand_population += pop_val

            if group_rows:
                gr_df = pd.DataFrame(group_rows)
                gr_df_display = gr_df.copy()
                gr_df_display["Total revenue ($)"] = gr_df_display["total_revenue"].map(lambda x: f"${x:,.2f}")
                gr_df_display = gr_df_display.rename(columns={"name": "Group", "population": "Population"})
                st.markdown("### Revenue by group")
                st.dataframe(
                    gr_df_display[["Group", "Population", "Total revenue ($)"]],
                    use_container_width=True,
                    height=260,
                )

                fig = px.bar(
                    gr_df,
                    x="name",
                    y="total_revenue",
                    labels={"name": "Group", "total_revenue": "Total Revenue ($)"},
                    title="Total Revenue by Group",
                )
                fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            st.write(f"- Total population (all groups): **{grand_population:,d}**")
            st.write(f"- Grand total reimbursement (all groups): **${grand_revenue:,.2f}**")
            st.write(f"- Total Cost (clinic): **${total_cost:,.2f}**")
            st.markdown(f"## Final Net profit: **${grand_revenue - total_cost:,.2f}**")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("SPARC financial projections may not be accurate.")
