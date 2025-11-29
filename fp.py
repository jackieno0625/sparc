import streamlit as st
import numpy as np
from pathlib import Path
import pandas as pd
import streamlit as st
import io
import uuid
import plotly.express as px
import pkgutil

# -------------------------
# CSV Loader
# -------------------------


CSV_FILENAME = "Compiled Fee Assessment.csv"
CSV_PATH = Path(__file__).parent / CSV_FILENAME

@st.cache_data
def load_fee_csv_or_stop(path: Path):
    """
    Attempt to load the bundled CSV from disk next to fp.py or from package data.
    If neither is available, display an error and stop the app immediately.
    Returns: pandas.DataFrame
    """
    # 1) Try file next to the script (works on Streamlit Cloud & local)
    try:
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            return df
    except Exception as e:
        # if reading fails, show the exception below (fall through)
        err_msg = f"Found {path} but failed to read it as CSV: {e}"
        st.error(err_msg)
        st.stop()

    # 2) Try package data (useful if you bundle the CSV into a package)
    try:
        data = pkgutil.get_data(__name__, CSV_FILENAME)
        if data:
            df = pd.read_csv(io.BytesIO(data), dtype=str)
            return df
    except Exception as e:
        err_msg = f"Found package data for {CSV_FILENAME} but failed to read it as CSV: {e}"
        st.error(err_msg)
        st.stop()

    # 3) Not found anywhere — stop (no fallback sample, no uploader)
    st.error(
        f"Required file `{CSV_FILENAME}` was not found next to `fp.py` or as package data. "
        "Place the CSV file in the same folder as `fp.py` (exact filename required), then reload the app."
    )
    st.stop()

# Actually load (this will either return a DataFrame or stop the app)
reim_df = load_fee_csv_or_stop(CSV_PATH)

# Optional confirmation message (only appears when CSV successfully loaded)
st.success(f"Fee schedule loaded from: {CSV_PATH.name}")
# -------------------------

    
st.set_page_config(page_title="Financial Projection Dashboard", layout="wide")
st.markdown(
    """
    <div style="display:flex; justify-content:center; margin-top:10px; margin-bottom:20px;">
        <img src="https://i.imgur.com/40QPfA3.png" width="250" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)
st.title("Financial Projection Dashboard")

# -----------------------
# Sidebar: Simple/Advanced mode toggles for user inputs
# -----------------------
st.sidebar.header("Input Modes")

# Patient Population mode toggle
pop_mode = st.sidebar.radio("Patient Population Mode", options=["Simple", "Advanced"], index=0, help="Choose Simple for minimal inputs or Advanced for detailed population controls.")

# Total Cost mode toggle
cost_mode = st.sidebar.radio("Total Cost Mode", options=["Simple", "Advanced"], index=0, help="Choose Simple for minimal inputs or Advanced for detailed cost controls.")

# -----------------------
# Tabs at the top
# -----------------------
tab_main, tab_instructions = st.tabs(["Main", "Instructions"])

# -----------------------
# Instructions Tab
# -----------------------
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
          “What % of the gap between insurer reimbursement and practice fee do patients pay?”

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
        - “Other”

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

        **SPARC then sums everything** to create the clinic’s total annual cost.

        ---

        ## 📊 What Appears in “Model Outputs”
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


        ---

        ## Need help?
        Each section includes a **? help icon** explaining how that field works.
        Please contact the developer through SPARC's website it problems persist.

        """
    )


# -----------------------
# Main Tab
# -----------------------
with tab_main:
    st.markdown("Open the sidebar on the left to toggle between Simple and Advanced modeling options.")
    # Two-column layout with a thin divider
    col1, divider, col2 = st.columns([0.6, 0.02, 0.38])

    with divider:
        st.markdown("<div style='border-left: 1px solid #cccccc; height: 100%; margin: auto;'></div>", unsafe_allow_html=True)

    # -----------------------
    # Left Column: Patient Population (vertical inputs, not crowded)
    # -----------------------
    with col1:
        st.subheader("Create a Patient Population")
        st.write("Mode:", f"**{pop_mode}**")

        if pop_mode == "Simple":
            st.markdown("#### Patient Population Size")

            st.session_state.setdefault("simple_population", 0)
            simple_population = st.number_input(
                "Enter patient population:",
                min_value=0,
                step=1,
                key="simple_population",
                help="Type the total number of patients you want to model."
            )

            st.write("You entered:", simple_population)

            st.markdown("#### Insurer Mix")
            insurers = ["Uninsured", "Medicaid", "Healthy Blue", "Trillium", "Aetna", "Medicare"]

            # safe initialization — run BEFORE creating the widgets:
            for name in insurers:
                key = f"pct_{name}"
                st.session_state.setdefault(key, 0.0)

            # create widgets — DO NOT pass `value=...`, let the widget read initial value from session_state via `key`
            for name in insurers:
                key = f"pct_{name}"
                st.number_input(
                    f"{name} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    key=key,
                    help="Enter insurer percentages for the population. They must total 100%."
                )

            # Compute and show total
            pct_values = np.array([st.session_state[f"pct_{name}"] for name in insurers], dtype=float)
            total_pct = float(np.round(pct_values.sum(), 6))
            st.markdown(f"**Total: {total_pct:.2f}%**")
            if abs(total_pct - 100.0) > 1e-6:
                st.error("Insurer percentages MUST add up to 100%. Please adjust the values above.")

            # -------------------------
            # CPT selection + patient-share slider (drop this into your Simple population block)
            # -------------------------
            st.markdown("#### Select CPT Codes")
            # Build display strings for CPTs and descriptions (safe if Description missing)
            cpt_display = (reim_df["CPT Code"].astype(str) 
                           + " — " 
                           + reim_df.get("Description", "").astype(str))
            # Use multiselect (user should pick up to 4)
            # ensure session key exists before creating the multiselect
            st.session_state.setdefault("selected_cpts", [])

            selected = st.multiselect(
                "Select up to 4 CPT codes (these will be applied to every patient)",
                options=cpt_display.tolist(),
                key="selected_cpts",
                help="Search CPTs by code or description. Pick up to 4."
            )


            # Enforce maximum of 4 choices (inform the user)
            if len(selected) > 4:
                st.warning("Please select at most 4 CPT codes. Only the first 4 will be used.")
                selected = selected[:4]

            if len(selected) == 0:
                st.info("Select 1–4 CPT codes to enable reimbursement calculations.")
            else:
                # Map display string back to CPT code
                display_to_cpt = {
                    f"{row['CPT Code']} — {row.get('Description','')}": row["CPT Code"]
                    for _, row in reim_df.iterrows()
                }
                selected_cpts = [display_to_cpt[s] for s in selected]
                
                # Slider: what % of the shortfall patients pay (0-100%)
                st.markdown("#### Reimbursement via Patients")

                # ensure key exists before widget
                st.session_state.setdefault("pct_patient_share", 0)

                pct_patient_share = st.slider(
                    "Patient pays what percent of the shortfall after insurance reimburses? (X%)",
                    min_value=0,
                    max_value=100,
                    step=1,
                    key="pct_patient_share",
                    help="If insurer pays less than the practice fee, patient covers X% of that gap."
                ) / 100.0


                # Build payer list consistent with insurer mix inputs
                # Make sure this list matches the names you used for payer inputs earlier
                payers = insurers  # e.g., ["Uninsured","Medicaid","Healthy Blue", ...]
                # Build payer probabilities from session state (user inputs)
                payer_pcts = np.array([float(st.session_state.get(f"pct_{p}", 0.0)) for p in payers], dtype=float)
                if payer_pcts.sum() == 0:
                    st.error("Payer distribution sums to 0. Please enter insurer percentages.")
                else:
                    payer_probs = payer_pcts / payer_pcts.sum()  # normalized

                    # helper: get insurer payment for a cpt and payer (coerce to float)
                    def get_insurer_pay(payer_name, cpt_code):
                        row = reim_df[reim_df["CPT Code"].astype(str) == str(cpt_code)]
                        if row.empty:
                            return 0.0
                        row = row.iloc[0]
                        if payer_name.lower() == "uninsured":
                            return 0.0  # insurer pays nothing for uninsured
                        # try exact column name then case-insensitive match
                        if payer_name in reim_df.columns:
                            val = row.get(payer_name, 0.0)
                        else:
                            val = 0.0
                            for c in reim_df.columns:
                                if c.strip().lower() == payer_name.strip().lower():
                                    val = row.get(c, 0.0)
                                    break
                        # coerce to float safely (strip $/commas if needed)
                        try:
                            return float(str(val).replace("$", "").replace(",", "").strip() or 0.0)
                        except:
                            return 0.0

                    # get practice fee for CPT
                    def get_practice_fee(cpt_code):
                        row = reim_df[reim_df["CPT Code"].astype(str) == str(cpt_code)]
                        if row.empty:
                            return 0.0
                        val = row.iloc[0].get("Practice Fee", 0.0)
                        try:
                            return float(str(val).replace("$", "").replace(",", "").strip() or 0.0)
                        except:
                            return 0.0

                    # Compute expected per-patient revenue (averaged across payer mix)
                    per_patient_revenue = 0.0
                    for p_idx, payer in enumerate(payers):
                        prob = payer_probs[p_idx]
                        # revenue for a patient of this payer (sum across selected CPTs)
                        payer_revenue_sum = 0.0
                        for cpt in selected_cpts:
                            insurer_pay = get_insurer_pay(payer, cpt)
                            practice_fee = get_practice_fee(cpt)
                            shortfall = max(0.0, practice_fee - insurer_pay)
                            # patient pays pct_patient_share of the shortfall
                            patient_pay = pct_patient_share * shortfall
                            # total revenue from this CPT for this payer = insurer_pay + patient_pay
                            # for uninsured insurer_pay=0, patient_pay becomes pct*practice_fee (consistent)
                            payer_revenue_sum += (insurer_pay + patient_pay)
                        per_patient_revenue += prob * payer_revenue_sum

                    # Totals across population
                    population = int(simple_population) if "simple_population" in locals() or "simple_population" in globals() else int(st.session_state.get("simple_population", 0))
                    # prefer local variable if present (from the UI)
                    try:
                        population = int(simple_population)
                    except:
                        population = int(st.session_state.get("simple_population", 0) or 0)

                    total_revenue = population * per_patient_revenue

                    # Determine total cost to subtract (respect current cost_mode)
                    if st.session_state.get("net_cost_advanced", None) is not None and st.session_state.get("net_cost_advanced", 0.0) > 0 and st.session_state.get("cost_mode", cost_mode) == "Advanced":
                        net_cost_value = float(st.session_state.get("net_cost_advanced", 0.0))
                    else:
                        # use simple total cost if available in session_state
                        net_cost_value = float(st.session_state.get("simple_net_cost", 0.0))

                    net_profit = total_revenue - net_cost_value

                    # Display results
                    st.markdown(f"## Estimated Net Reimbursement: **${net_profit:,.2f}**")

                    # -------------------------
                    # Quick scenario comparison (only total revenue)
                    # -------------------------
                    st.markdown("#### Quick scenario comparison (patient pays X% of shortfall)")

                    scenarios = []
                    for s in [0.0, 0.5, 0.75, 1.0]:
                        # compute per-patient revenue for scenario s
                        pp_rev = 0.0
                        for p_idx, payer in enumerate(payers):
                            prob = payer_probs[p_idx]
                            payer_sum = 0.0
                            for cpt in selected_cpts:
                                insurer_pay = get_insurer_pay(payer, cpt)
                                practice_fee = get_practice_fee(cpt)
                                shortfall = max(0.0, practice_fee - insurer_pay)
                                patient_pay = s * shortfall
                                payer_sum += insurer_pay + patient_pay
                            pp_rev += prob * payer_sum

                        total_rev_s = population * pp_rev

                        scenarios.append({
                            "patient_share": f"{int(s*100)}%",
                            "total_revenue": f"${total_rev_s:,.2f}"
                        })

                    # Convert to DataFrame and set index
                    sc_df = pd.DataFrame(scenarios).set_index("patient_share")

                    # Rename index (because patient_share is the index, not a column)
                    sc_df.index.name = "Patient Share (%)"

                    # Rename data columns if desired
                    sc_df = sc_df.rename(columns={
                        "total_revenue": "Total Revenue ($)"
                    })

                    st.table(sc_df)

        else:
            # -----------------------
            # Advanced: Service Group Manager (per-group populations, CPTs, payer mix, per-group patient-share)
            # -----------------------

            # default payer list (matches your UI)
            INSURERS = ["Uninsured", "Medicaid", "Healthy Blue", "Trillium", "Aetna", "Medicare"]

            # initialize groups container in session_state
            if "service_groups" not in st.session_state:
                # each group: {"id":id, "name":str}
                st.session_state["service_groups"] = [
                    {"id": str(uuid.uuid4()), "name": "Primary Care"},
                    {"id": str(uuid.uuid4()), "name": "Adult Health"},
                    {"id": str(uuid.uuid4()), "name": "STD-related"},
                    {"id": str(uuid.uuid4()), "name": "BCCC-related"},
                    {"id": str(uuid.uuid4()), "name": "Other"},
                ]

            # helper to create keys per group
            def kp(gid, key):
                return f"{gid}__{key}"

            # functions to read fee schedule safely
            def _insurer_pay_for_cpt(reim_df_local, payer_name, cpt_code):
                row = reim_df_local[reim_df_local["CPT Code"].astype(str) == str(cpt_code)]
                if row.empty:
                    return 0.0
                row = row.iloc[0]
                if payer_name.lower() == "uninsured":
                    return 0.0
                if payer_name in reim_df_local.columns:
                    val = row.get(payer_name, 0.0)
                else:
                    val = 0.0
                    for c in reim_df_local.columns:
                        if c.strip().lower() == payer_name.strip().lower():
                            val = row.get(c, 0.0)
                            break
                try:
                    return float(str(val).replace("$","").replace(",","").strip() or 0.0)
                except:
                    return 0.0

            def _practice_fee_for_cpt(reim_df_local, cpt_code):
                row = reim_df_local[reim_df_local["CPT Code"].astype(str) == str(cpt_code)]
                if row.empty:
                    return 0.0
                val = row.iloc[0].get("Practice Fee", 0.0)
                try:
                    return float(str(val).replace("$","").replace(",","").strip() or 0.0)
                except:
                    return 0.0

            # function: expected per-patient revenue given selected CPTs, payer probs, and patient-share (fraction)
            def compute_per_patient_expected(selected_cpts, payer_probs, patient_share_frac, insurers_list, reim_df_local):
                per_patient_rev = 0.0
                for p_idx, payer in enumerate(insurers_list):
                    prob = payer_probs[p_idx]
                    payer_sum = 0.0
                    for cpt in selected_cpts:
                        insurer_pay = _insurer_pay_for_cpt(reim_df_local, payer, cpt)
                        practice_fee = _practice_fee_for_cpt(reim_df_local, cpt)
                        shortfall = max(0.0, practice_fee - insurer_pay)
                        patient_pay = patient_share_frac * shortfall
                        payer_sum += (insurer_pay + patient_pay)
                    per_patient_rev += prob * payer_sum
                return per_patient_rev

            # UI: group controls (collapsible)
            st.header("Advanced: Service Groups (per-group populations, CPTs, payer mix, patient-share)")

            cols = st.columns([1, 1])
            with cols[0]:
                st.markdown("**Manage groups**")
                if st.button("Add new empty group"):
                    new_id = str(uuid.uuid4())
                    st.session_state["service_groups"].append({"id": new_id, "name": f"New Group"})
            with cols[1]:
                st.markdown("**Actions**")
                if st.button("Reset to defaults"):
                    st.session_state["service_groups"] = [
                        {"id": str(uuid.uuid4()), "name": "Primary Care"},
                        {"id": str(uuid.uuid4()), "name": "Adult Health"},
                        {"id": str(uuid.uuid4()), "name": "STD-related"},
                        {"id": str(uuid.uuid4()), "name": "BCCC-related"},
                        {"id": str(uuid.uuid4()), "name": "Other"},
                    ]
                    # clear group-specific keys for cleanliness (optional)
                    for k in list(st.session_state.keys()):
                        if "__" in k:
                            st.session_state.pop(k, None)

            # iterate groups and render an expander for each
            group_summaries = []
            for grp in list(st.session_state["service_groups"]):  # iterate over copy to allow removals
                gid = grp["id"]
                exp_title = f"{grp.get('name', 'Group')}"

                with st.expander(exp_title, expanded=False):
                    # --- Group name field (updates expander label immediately, safe pattern) ---
                    name_key = kp(gid, "name")

                    # initialize session_state for this name key before creating the widget
                    if name_key not in st.session_state:
                        st.session_state[name_key] = grp.get("name", "")

                    # callback that runs when user edits the text input
                    def _on_group_name_change(gid=gid, name_key=name_key):
                        # update the corresponding group entry in the session-state list
                        for g in st.session_state.get("service_groups", []):
                            if g.get("id") == gid:
                                g["name"] = st.session_state.get(name_key, "")
                                break
                        # re-run so UI (expander titles etc.) refresh immediately
                        st.rerun()

                    # create text input bound to the key and callback
                    st.text_input("Group name", key=name_key, on_change=_on_group_name_change)
                    # No manual assignment to st.session_state[name_key] here — the widget manages it.



                    # population input (keyed)
                    pop_key = kp(gid, "population")
                    if pop_key not in st.session_state:
                        st.session_state[pop_key] = 0
                    st.number_input("Population (integer)", min_value=0, step=1, value=int(st.session_state[pop_key]), key=pop_key, help="Number of patients in this service group")

                    # CPT multiselect for this group (up to 4)
                    cpt_display = (reim_df["CPT Code"].astype(str) + " — " + reim_df.get("Description", "").astype(str)).tolist()
                    sel_key = kp(gid, "selected_cpts")
                    if sel_key not in st.session_state:
                        st.session_state[sel_key] = []
                    st.multiselect("Select up to 4 CPT codes for this group", options=cpt_display, default=st.session_state[sel_key], key=sel_key, help="These CPTs will be applied to visits in this group (per-visit).")

                    # group-level patient-share slider (0-100)
                    share_key = kp(gid, "patient_share_pct")
                    if share_key not in st.session_state:
                        st.session_state[share_key] = 0
                    st.slider("Patient pays what % of shortfall (group-level)", min_value=0, max_value=100, value=int(st.session_state[share_key]), key=share_key, help="This percent is applied to each CPT shortfall for this group's patients.")

                    # payer mix inputs for this group (vertical)
                    st.markdown("Payer mix for this group (must sum to 100%)")
                    payer_vals = []
                    for payer in INSURERS:
                        k = kp(gid, f"pct_{payer}")
                        # initialize if missing
                        if k not in st.session_state:
                            st.session_state[k] = 0.0
                        st.number_input(f"{payer} (%)", min_value=0.0, max_value=100.0, step=0.5, value=float(st.session_state[k]), key=k)
                        payer_vals.append(float(st.session_state[k]))
                    total_pct = sum(payer_vals)
                    if total_pct <= 0:
                        st.warning("Payer mix for this group sums to 0 — enter percentages before computing.")
                    elif abs(total_pct - 100.0) > 1e-6:
                        st.warning(f"Payer mix sums to {total_pct:.2f}%. Please make it sum to 100%.")

                    # remove group button
                    if st.button("Remove this group", key=kp(gid, "remove")):
                        # delete group and any session_state keys for tidiness
                        st.session_state["service_groups"] = [g for g in st.session_state["service_groups"] if g["id"] != gid]
                        # clean up keys
                        for k in list(st.session_state.keys()):
                            if k.startswith(gid + "__"):
                                st.session_state.pop(k, None)
                        st.rerun()


                    # compute group-level summary into group_summaries to show below
                    # but only compute if payer mix > 0 and CPTs selected
                    sel_displays = st.session_state.get(sel_key, [])
                    sel_cpts = []
                    display_to_cpt_map = {f"{row['CPT Code']} — {row.get('Description','')}": row["CPT Code"] for _, row in reim_df.iterrows()}
                    for d in sel_displays:
                        if d in display_to_cpt_map:
                            sel_cpts.append(display_to_cpt_map[d])
                    population_val = int(st.session_state.get(pop_key, 0) or 0)
                    patient_share_frac = float(st.session_state.get(share_key, 0) or 0) / 100.0

                    payer_pct_array = np.array([float(st.session_state.get(kp(gid, f"pct_{p}"), 0.0) or 0.0) for p in INSURERS], dtype=float)
                    if payer_pct_array.sum() > 0 and population_val > 0 and len(sel_cpts) > 0:
                        payer_probs_local = payer_pct_array / payer_pct_array.sum()
                        per_patient = compute_per_patient_expected(sel_cpts, payer_probs_local, patient_share_frac, INSURERS, reim_df)
                        total_revenue_grp = per_patient * population_val
                    else:
                        per_patient = 0.0
                        total_revenue_grp = 0.0

                    group_summaries.append({
                        "id": gid,
                        "name": grp.get("name", ""),
                        "population": population_val,
                        "per_patient_rev": per_patient,
                        "total_revenue": total_revenue_grp
                    })



    # -----------------------
    # Right Column: Total Cost / Reimbursement (stacked vertically)
    # -----------------------
    with col2:
        st.subheader("Total Cost")
        st.write("Mode:", f"**{cost_mode}**")

        if cost_mode == "Simple":
            st.markdown("### Simple Total Cost Input")
            simple_net_cost = st.number_input(
                "Enter total cost ($):",
                min_value=0.0,
                step=1000.0,
                value=0.0,
                key="simple_net_cost",            
                help="This represents all costs combined — overhead + variable + fixed."
            )

            st.markdown(f"## Total Cost = **${simple_net_cost:,.2f}**")

        else:
            # ---- Advanced Cost Inputs (fixed: no experimental_rerun, safe removals) ----
            st.markdown("### Advanced Cost Inputs")

            # ---- FIXED COSTS ----
            st.markdown("#### Fixed costs (subscriptions, maintenance, IT, rent, etc.)")

            if "fixed_items" not in st.session_state:
                st.session_state["fixed_items"] = []  # list of {"item": str, "annual_cost": float}

            # Prepare new-item keys
            if "fixed_new_name" not in st.session_state:
                st.session_state["fixed_new_name"] = ""
            if "fixed_new_amount" not in st.session_state:
                st.session_state["fixed_new_amount"] = 0.0

            # Callback to add a fixed item (safe)
            def add_fixed_item():
                name = st.session_state.get("fixed_new_name", "").strip()
                amt = float(st.session_state.get("fixed_new_amount", 0.0) or 0.0)
                if name == "":
                    # can't show st.warning here inside callback reliably; set a flag instead
                    st.session_state["_add_fixed_warning"] = "Please enter a name for the fixed cost before adding."
                    return
                st.session_state["fixed_items"].append({"item": name, "annual_cost": amt})
                st.session_state["fixed_new_name"] = ""
                st.session_state["fixed_new_amount"] = 0.0
                st.session_state.pop("_add_fixed_warning", None)

            # Widgets for adding a fixed-cost item
            st.text_input("New fixed cost name", key="fixed_new_name")
            st.number_input("New fixed cost annual amount ($)", min_value=0.0, value=0.0, step=100.0, key="fixed_new_amount")
            st.button("Add fixed cost item", on_click=add_fixed_item)

            # Show any add warning set by callback
            if st.session_state.get("_add_fixed_warning"):
                st.warning(st.session_state["_add_fixed_warning"])

            # Display current fixed-cost items and allow safe removal
            fixed_total = 0.0
            to_remove_fixed = None
            if st.session_state["fixed_items"]:
                st.write("Current fixed cost items:")
                # iterate and render; capture index to remove after loop if pressed
                for idx, entry in enumerate(list(st.session_state["fixed_items"])):
                    cols = st.columns([4, 1, 1])
                    cols[0].markdown(f"**{entry['item']}**")
                    cols[1].markdown(f"${entry['annual_cost']:,.2f}")
                    if cols[2].button("Remove", key=f"remove_fixed_{idx}"):
                        to_remove_fixed = idx
                    fixed_total += entry["annual_cost"]
                # remove after rendering loop
                if to_remove_fixed is not None:
                    st.session_state["fixed_items"].pop(to_remove_fixed)
            else:
                st.info("No fixed cost items added yet.")

            st.write(f"Fixed costs total: **${fixed_total:,.2f}**")
            st.markdown("---")

            # ---- PROVIDER PAYROLL ----
            st.markdown("#### Provider payroll")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                num_providers = st.number_input("Number of providers", min_value=0, value=0, step=1, key="adv_num_providers")
            with col_b:
                hours_per_week = st.number_input("Hours/provider/week", min_value=0.0, value=0.0, step=1.0, key="adv_hours_week")
            with col_c:
                weeks_per_year = st.number_input("Paid work weeks/year", min_value=0, value=0, step=1, key="adv_weeks_year")
            with col_d:
                hourly_pay = st.number_input("Hourly pay ($)", min_value=0.0, value=0.0, step=1.0, key="adv_hourly_pay")

            annual_per_provider = hours_per_week * weeks_per_year * hourly_pay
            provider_total = num_providers * annual_per_provider
            st.write(f"Provider payroll total: {provider_total:,.2f} (${annual_per_provider:,.2f} per provider / year)")
            st.markdown("---")

            # ---- MISC COSTS (no override) ----
            st.markdown("#### Miscellaneous costs")

            if "misc_items" not in st.session_state:
                st.session_state["misc_items"] = []

            # Prepare new-item keys
            if "misc_new_name" not in st.session_state:
                st.session_state["misc_new_name"] = ""
            if "misc_new_amount" not in st.session_state:
                st.session_state["misc_new_amount"] = 0.0

            # Callback to add a misc item
            def add_misc_item():
                name = st.session_state.get("misc_new_name", "").strip()
                amt = float(st.session_state.get("misc_new_amount", 0.0) or 0.0)
                if name == "":
                    st.session_state["_add_misc_warning"] = "Please enter a name for the misc cost before adding."
                    return
                st.session_state["misc_items"].append({"item": name, "annual_cost": amt})
                st.session_state["misc_new_name"] = ""
                st.session_state["misc_new_amount"] = 0.0
                st.session_state.pop("_add_misc_warning", None)

            st.text_input("New miscellaneous cost name", key="misc_new_name")
            st.number_input("New misc cost annual amount ($)", min_value=0.0, value=0.0, step=100.0, key="misc_new_amount")
            st.button("Add misc cost item", on_click=add_misc_item)

            if st.session_state.get("_add_misc_warning"):
                st.warning(st.session_state["_add_misc_warning"])

            misc_total = 0.0
            to_remove_misc = None
            if st.session_state["misc_items"]:
                st.write("Current miscellaneous cost items:")
                for idx, entry in enumerate(list(st.session_state["misc_items"])):
                    cols = st.columns([4, 1, 1])
                    cols[0].markdown(f"**{entry['item']}**")
                    cols[1].markdown(f"${entry['annual_cost']:,.2f}")
                    if cols[2].button("Remove", key=f"remove_misc_{idx}"):
                        to_remove_misc = idx
                    misc_total += entry['annual_cost']
                if to_remove_misc is not None:
                    st.session_state["misc_items"].pop(to_remove_misc)
            else:
                st.info("No miscellaneous cost items added yet.")

            st.write(f"Misc costs total: **${misc_total:,.2f}**")
            st.markdown("---")

            # ---- FINAL TOTAL COST AGGREGATION ----
            net_cost_advanced = fixed_total + provider_total + misc_total
            st.markdown(f"## Total Cost = **${net_cost_advanced:,.2f}**")
            # keep a canonical stored value for downstream reads
            st.session_state["net_cost_advanced"] = float(net_cost_advanced)


    # -----------------------
    # Final Outputs (supports Simple/Advanced population AND Simple/Advanced cost)
    # Replace your existing Final Outputs section with this block.
    # -----------------------
    st.markdown("---")
    st.markdown("# Model Outputs")

    # --- helpers (safe conversions) ---
    def _to_float_safe(x, default=0.0):
        try:
            return float(x)
        except:
            return default

    def _get_insurer_pay(payer_name, cpt_code, df):
        row = df[df["CPT Code"].astype(str) == str(cpt_code)]
        if row.empty:
            return 0.0
        row = row.iloc[0]
        if payer_name.lower() == "uninsured":
            return 0.0
        if payer_name in df.columns:
            val = row.get(payer_name, 0.0)
        else:
            val = 0.0
            for c in df.columns:
                if c.strip().lower() == payer_name.strip().lower():
                    val = row.get(c, 0.0)
                    break
        try:
            return float(str(val).replace("$","").replace(",","").strip() or 0.0)
        except:
            return 0.0

    def _get_practice_fee(cpt_code, df):
        row = df[df["CPT Code"].astype(str) == str(cpt_code)]
        if row.empty:
            return 0.0
        val = row.iloc[0].get("Practice Fee", 0.0)
        try:
            return float(str(val).replace("$","").replace(",","").strip() or 0.0)
        except:
            return 0.0

    def compute_per_patient_rev_for_cpts(cpt_codes, payer_probs_local, patient_share_frac, insurers_local, df):
        per_patient_rev = 0.0
        for p_idx, payer in enumerate(insurers_local):
            prob = payer_probs_local[p_idx]
            payer_sum = 0.0
            for cpt in cpt_codes:
                insurer_pay = _get_insurer_pay(payer, cpt, df)
                practice_fee = _get_practice_fee(cpt, df)
                shortfall = max(0.0, practice_fee - insurer_pay)
                patient_pay = patient_share_frac * shortfall
                payer_sum += (insurer_pay + patient_pay)
            per_patient_rev += prob * payer_sum
        return per_patient_rev

    # --- Determine unified clinic total cost (support both Simple and Advanced cost modes) ---
    def compute_total_cost_from_state():
        """Compute advanced total cost from session_state items (fallback)."""
        fixed_total = sum(_to_float_safe(i.get("annual_cost", 0.0), 0.0) for i in st.session_state.get("fixed_items", []))
        misc_total = sum(_to_float_safe(i.get("annual_cost", 0.0), 0.0) for i in st.session_state.get("misc_items", []))
        num_providers = int(_to_float_safe(st.session_state.get("adv_num_providers", 0), 0))
        hours_per_week = _to_float_safe(st.session_state.get("adv_hours_week", 0.0), 0.0)
        weeks_per_year = int(_to_float_safe(st.session_state.get("adv_weeks_year", 0), 0))
        hourly_pay = _to_float_safe(st.session_state.get("adv_hourly_pay", 0.0), 0.0)
        provider_total = num_providers * (hours_per_week * weeks_per_year * hourly_pay)
        return fixed_total + misc_total + provider_total

    if cost_mode == "Advanced":
        total_cost_value = st.session_state.get("net_cost_advanced", None)
        if total_cost_value is None:
            # compute fallback and store it for later reuse
            total_cost_value = compute_total_cost_from_state()
            st.session_state["net_cost_advanced"] = float(total_cost_value)
    else:
        # Simple cost mode: prefer explicit widget key "simple_net_cost"
        total_cost_value = _to_float_safe(st.session_state.get("simple_net_cost", 0.0), 0.0)

    # --- canonical insurers list used in payer widgets ---
    INSURERS = ["Uninsured", "Medicaid", "Healthy Blue", "Trillium", "Aetna", "Medicare"]

    # --- Compute total revenue depending on population mode ---
    grand_total_revenue = 0.0
    population_total = 0

    if pop_mode == "Simple":
        # Simple population path uses global CPT selection + global slider
        selected_display = st.session_state.get("selected_cpts", [])  # display strings like "12345 — Desc"
        slider_percent = float(st.session_state.get("pct_patient_share", 0.0))  # 0-100
        display_to_cpt = {f"{row['CPT Code']} — {row.get('Description','')}": row["CPT Code"] for _, row in reim_df.iterrows()}
        selected_cpts = [display_to_cpt[s] for s in selected_display if s in display_to_cpt]

        if not selected_cpts:
            st.info("No CPTs selected yet — select up to 4 CPT codes in the Patient Population panel to see final outputs.")
        else:
            payer_pcts = np.array([_to_float_safe(st.session_state.get(f"pct_{p}", 0.0), 0.0) for p in INSURERS], dtype=float)
            if payer_pcts.sum() == 0:
                st.error("Payer distribution sums to 0. Please enter insurer percentages in the Patient Population panel.")
            else:
                payer_probs = payer_pcts / payer_pcts.sum()
                population = int(_to_float_safe(st.session_state.get("simple_population", 0), 0))
                population_total = population
                slider_frac = slider_percent / 100.0

                per_patient_rev_now = compute_per_patient_rev_for_cpts(selected_cpts, payer_probs, slider_frac, INSURERS, reim_df)
                grand_total_revenue = per_patient_rev_now * population

                st.write(f"- Patient shortfall share (slider): **{slider_frac*100:.0f}%**")
                st.write(f"- Population: **{population:,d}**")
                st.write(f"- Total Reimbursement: **${grand_total_revenue:,.2f}**")
                st.write(f"- Total Cost: **${total_cost_value:,.2f}**")
                st.markdown(f"## Final Net profit: **${(grand_total_revenue - total_cost_value):,.2f}**")

    else:
        # Advanced population path: iterate groups stored in session_state
        groups = st.session_state.get("service_groups", [])
        if not groups:
            st.info("No service groups configured in Advanced Patient Population. Add groups in the Advanced panel.")
        else:
            group_rows = []
            for grp in groups:
                gid = grp.get("id")
                gname = st.session_state.get(f"{gid}__name", grp.get("name","Group"))
                pop_val = int(_to_float_safe(st.session_state.get(f"{gid}__population", 0), 0))
                sel_displays = st.session_state.get(f"{gid}__selected_cpts", []) or []
                display_to_cpt = {f"{row['CPT Code']} — {row.get('Description','')}": row["CPT Code"] for _, row in reim_df.iterrows()}
                sel_cpts = [display_to_cpt[d] for d in sel_displays if d in display_to_cpt]
                payer_pct_array = np.array([_to_float_safe(st.session_state.get(f"{gid}__pct_{p}", 0.0), 0.0) for p in INSURERS], dtype=float)
                group_share_pct = float(_to_float_safe(st.session_state.get(f"{gid}__patient_share_pct", 0), 0)) / 100.0

                if pop_val <= 0 or len(sel_cpts) == 0 or payer_pct_array.sum() == 0:
                    per_patient = 0.0
                    total_revenue_grp = 0.0
                else:
                    payer_probs_local = payer_pct_array / payer_pct_array.sum()
                    per_patient = compute_per_patient_rev_for_cpts(sel_cpts, payer_probs_local, group_share_pct, INSURERS, reim_df)
                    total_revenue_grp = per_patient * pop_val

                group_rows.append({
                    "name": gname,
                    "population": pop_val,
                    "per_patient": per_patient,
                    "total_revenue": total_revenue_grp
                })

                grand_total_revenue += total_revenue_grp
                population_total += pop_val

            # present group summary
            if group_rows:
                gr_df = pd.DataFrame(group_rows)
                gr_df_display = gr_df.copy()
                gr_df_display["Total revenue ($)"] = gr_df_display["total_revenue"].map(lambda x: f"${x:,.2f}")
                gr_df_display = gr_df_display.rename(columns={"name":"Group", "population":"Population"})
                st.markdown("### Revenue by group")
                st.dataframe(gr_df_display[["Group","Population","Total revenue ($)"]], width=True)

                # small bar chart if plotly available
                try:
                    import plotly.express as px
                    fig = px.bar(pd.DataFrame(group_rows), x="name", y="total_revenue",
                                 labels={"name":"Group","total_revenue":"Total Revenue ($)"},
                                 title="Total Revenue by Group")
                    fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
                    st.plotly_chart(fig, width=True)
                except Exception:
                    pass

            st.write(f"- Total population (all groups): **{population_total:,d}**")
            st.write(f"- Grand total reimbursement (all groups): **${grand_total_revenue:,.2f}**")
            st.write(f"- Total Cost (clinic): **${total_cost_value:,.2f}**")
            st.markdown(f"## Final Net profit: **${(grand_total_revenue - total_cost_value):,.2f}**")





# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("SPARC financial projections may not be accurate.")
