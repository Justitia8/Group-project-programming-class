#AIused

# -*- coding: utf-8 -*-
"""
Creates an optimal three-round speed-networking schedule
and writes the result to  event_assignment_result.xlsx

The MILP model:
  • Exactly one company per student & round
  • 3–7 students per company & round
  • No repeat visits
  • Objective = maximise weighted preference points
"""

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp   # Mixed-Integer Linear Programming


# -------------------------------------------------------------------------
# 1.  File paths
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent               # folder where the script lives
INPUT_FILE  = BASE_DIR / "event_input.xlsx"              # raw data: companies & students
OUTPUT_FILE = BASE_DIR / "event_assignment_result.xlsx"  # final schedule


# -------------------------------------------------------------------------
# 2.  Read Excel data
# -------------------------------------------------------------------------
xls = pd.ExcelFile(INPUT_FILE)         # open workbook once

# First sheet (index 0) contains the company list
companies_df = xls.parse(0)
# Second sheet (index 1) contains student names + preference columns
students_df  = xls.parse(1)

# Clean whitespace / NaNs and convert to plain Python lists
company_names = (
    companies_df.iloc[:, 0]            # take the first column
               .dropna()               # drop empty cells
               .str.strip()            # trim whitespace
               .to_list()
)
student_names = (
    students_df.iloc[:, 0]             # first column = student names
               .astype(str)            # ensure string dtype
               .str.strip()
               .to_list()
)

# Immutable ranges for easy iteration (0 … n-1)
COMPANY_IDS = range(len(company_names))
STUDENT_IDS = range(len(student_names))
ROUNDS      = range(3)                  # exactly three networking rounds

# Fast lookup  “company name  →  internal index”
company_to_idx = {name: idx for idx, name in enumerate(company_names)}


# -------------------------------------------------------------------------
# 2a.  Build the preference matrix
# -------------------------------------------------------------------------
# Identify all columns that look like “prio1”, “Prio_2”, …  (case-insensitive)
prio_cols = [col for col in students_df.columns if "prio" in col.lower()]

# Sort them numerically so “prio10” doesn’t come before “prio2”
prio_cols.sort(key=lambda col: int("".join(filter(str.isdigit, col)) or 99))

# Points awarded per rank: 1st wish → 5 pts, 2nd wish → 4 pts, …, 5th wish → 1 point
points_for_rank = {rank: 6 - rank for rank in range(1, 6)}

# preference_matrix[s, c] = points a student s assigns to company c
preference_matrix = np.zeros(
    (len(student_names), len(company_names)),
    dtype=int
)

# Mapping for later colour-coding:
#   student name → { company name: rank (1–5) }
student_rank_map = {}

# Iterate row by row through the students sheet
for s_idx, row in students_df.iterrows():
    student = student_names[s_idx]
    rank_dict = {}                     # temporary mapping for this student

    # Check every “prio#” column
    for rank, col in enumerate(prio_cols, start=1):
        company = str(row[col]).strip()    # read cell as string + trim
        # Ignore blanks / NaNs
        if company and company.lower() != "nan":
            c_idx = company_to_idx[company]
            # Fill matrix with the weight (5 … 1)
            preference_matrix[s_idx, c_idx] = points_for_rank[rank]
            # Remember the rank for colouring
            rank_dict[company] = rank

    student_rank_map[student] = rank_dict   # may be empty if no wishes given


# -------------------------------------------------------------------------
# 3.  MILP model
# -------------------------------------------------------------------------
solver = pywraplp.Solver.CreateSolver("CBC")  # open-source MILP engine

# Decision variables:
# x[s, c, r] == 1  ⇔  student s meets company c in round r
x = {
    (s, c, r): solver.BoolVar(f"x_{s}_{c}_{r}")
    for s in STUDENT_IDS
    for c in COMPANY_IDS
    for r in ROUNDS
}

# ---------- Constraints --------------------------------------------------
# (1)  Each student is scheduled with exactly ONE company in every round
for s in STUDENT_IDS:
    for r in ROUNDS:
        solver.Add(sum(x[s, c, r] for c in COMPANY_IDS) == 1)

# (2)  Capacity: 3–7 students per company & round
for c in COMPANY_IDS:
    for r in ROUNDS:
        load = sum(x[s, c, r] for s in STUDENT_IDS)
        solver.Add(load >= 3)
        solver.Add(load <= 7)

# (3)  No student visits the same company twice across rounds
for s in STUDENT_IDS:
    for c in COMPANY_IDS:
        solver.Add(sum(x[s, c, r] for r in ROUNDS) <= 1)

# ---------- Objective ----------------------------------------------------
# Maximise the total weighted preferences fulfilled
solver.Maximize(
    sum(
        preference_matrix[s, c] * x[s, c, r]
        for s in STUDENT_IDS
        for c in COMPANY_IDS
        for r in ROUNDS
    )
)

# ---------- Solve --------------------------------------------------------
# Show a quick “before” snapshot: size of the mathematical program
print(
    f"Optimising …  variables = {solver.NumVariables():,}, "   # nr decision vars
    f"constraints = {solver.NumConstraints():,}"               # nr equalities + bounds
)

status = solver.Solve()    # launch CBC solver (may take a few seconds)

# Summarise the outcome: whether the solver hit optimum and the final score
print(
    "Result:",
    "OPTIMAL" if status == solver.OPTIMAL else "FEASIBLE",     # solution status
    f"(objective = {solver.Objective().Value():,.0f})",        # total preference points
)


# -------------------------------------------------------------------------
# 4.  Build the result tables
# -------------------------------------------------------------------------
# DataFrame: index = student names, columns = “Round 1” … “Round 3”
schedule_df = pd.DataFrame(
    index=student_names,
    columns=[f"Round {r + 1}" for r in ROUNDS],
)

# Fill the table with the company chosen by the solver
for s in STUDENT_IDS:
    for r in ROUNDS:
        # Exactly ONE company has x == 1 for (student s, round r).
        # Pull its index and turn that into a name.
        c_idx = next(
            c for c in COMPANY_IDS
            if x[s, c, r].solution_value() > 0.5
        )
        schedule_df.iat[s, r] = company_names[c_idx]

# Derive a company-centric view:
company_load_df = (
    schedule_df
        # reshape: one row per (student, round), value = company
        .melt(
            ignore_index=False,
            var_name="Round",
            value_name="Company"
        )
        # count rows per (company, round) combination
        .groupby(["Company", "Round"])
        .size()
        # move ‘Round’ values into columns “Round 1”, “Round 2”, …
        .unstack(fill_value=0)
        # keep original company order from input file
        .reindex(company_names)
)
company_load_df.index.name = "Company"


# -------------------------------------------------------------------------
# 5.  Write Excel file with colour coding
# -------------------------------------------------------------------------
with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:

    # Export raw data - one sheet per table
    schedule_df.to_excel(writer, sheet_name="Schedule_Students")
    company_load_df.to_excel(writer, sheet_name="Load_Companies")

    # Worksheets abholen
    wb          = writer.book
    ws_sched    = writer.sheets["Schedule_Students"]   # sheet 1
    ws_load     = writer.sheets["Load_Companies"]      # sheet 2

    # Fixe Spaltenbreiten setzen 
    ws_sched.set_column(0, len(schedule_df.columns), 40)      # column width sheet 1
    ws_load.set_column(0, len(company_load_df.columns), 15)   # column width sheet 2
    ws_load.set_column(0, 0, 40)                              # column width sheet 2 (only column 1)


    rank_to_format = {
        # “0” means the student hadn’t ranked this company at all.
        0: wb.add_format({"bg_color": "#FFFFFF"}),  # white = no wish
        1: wb.add_format({"bg_color": "#92D050"}),  # green  = 1st wish
        2: wb.add_format({"bg_color": "#F4B084"}),  # orange = 2nd wish
        3: wb.add_format({"bg_color": "#FFF2CC"}),  # yellow = 3rd wish
        4: wb.add_format({"bg_color": "#9DC3E6"}),  # blue   = 4th wish
        5: wb.add_format({"bg_color": "#FF9999"}),  # red    = 5th wish
    }

    # Excel coordinate offsets:
    row_offset = 1    # header row “Round 1/2/3” is row 0
    col_offset = 1    # index column (student names) is column 0

    # Apply colour per cell
    for row_nr, student in enumerate(schedule_df.index, start=row_offset):
        # Map of {company → rank (1–5)} for this particular student
        wishes = student_rank_map.get(student, {})
        for col_nr, round_lbl in enumerate(schedule_df.columns, start=col_offset):
            company = schedule_df.at[student, round_lbl]
            # rank = 0 if company not on the student’s wish list
            rank = wishes.get(company, 0)
            # Write text + background format into the cell
            ws_sched.write(row_nr, col_nr, company, rank_to_format[rank])

# Console confirmation that the file exists and is complete
print(f"Schedule saved to {OUTPUT_FILE}")
