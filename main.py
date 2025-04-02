import gurobipy as gp
from gurobipy import Model, GRB, quicksum
from alandata import load_data
import pandas as pd
from itertools import product
from result_processing import extract_gurobi_values
from sankey_visualization import create_sankey


def rename_columns(col):
    if 'no qual' in col:
        return col.replace('no qual', '0')
    elif 'type 1 qual' in col:
        return col.replace('type 1 qual', '1')
    elif 'type 2 qual' in col:
        return col.replace('type 2 qual', '2')
    elif 'type 3 qual' in col:
        return col.replace('type 3 qual', '3')
    return col

data = load_data()

initial_crew = data["initial_crew"]
initial_crew_type_qualification = data["initial_crew_type_qualification"]
crew_demand = data["crew_demand"]
grounded_aircraft_cost = data["grounded_aircraft_cost"]
grounded_aircraft_cost[0] = 0.0
crew_leaving = data["crew_leaving"]
simulator_ability = data["simulator_ability"]
training = data["training"]
training_types = data["training_types"]


"""
Sets
"""
C = set(initial_crew['Rating'])
rho = {'FO', 'C'}
A = {'Boeing', 'Airbus'}
S = set(simulator_ability['Week'])

    #for p in rho
    #for a in A
    #for q in Qqual

W = range(1, 53)  
W_start = range(1)
W_end = range(max(W)+1, max(W)+2)
W_all = range(max(W)+2)
W_with_end = range(1, max(W)+2)

Q = range(4)
Qunqual = range(1)
Qqual = range(1, 4)
Qext = range(4, 5)

Qall = range(5)
Q_trainee = {0, 4}

tau = set(training_types['Training Type'])
tau_internal = {1, 2, 3}
tau_external = {4, 5}
G = range(14)

"""
Parameters
"""

# availability of crew by type and category
c_0 = {
    (row['Rating'].split()[0], row['Rating'].split()[1]): row['Total']
    for _, row in initial_crew.iterrows()
}

# number of qualified instructors needed for training type t at week w for qualification q
k = {
    (row['Week'], row['Training Type'], row['Qualification']): row['Capacity Needed']
    for _, row in training.iterrows()
    if not pd.isna(row['Capacity Needed'])  # Only include if capacity is not NaN
}

# demand for crew of aircraft type a at week w
Delta = {
    (row['Week'], row['Aircraft']): row['Demand']
    for _, row in crew_demand.iterrows()
}

# number of crew departures at week w by type
d = {
    (row['Week'], *col.split()): row[col]
    for _, row in crew_leaving.iterrows()
    for col in row.index[1:]  # Skip the first index which is the week number
}

# total number of simulators available at week w
xi = {row['Week']: row['Available Simulators'] for _, row in simulator_ability.iterrows()};
sigma = {}
for _, row in training.iterrows():
    t = row['Training Type']
    w = row['Week']
    sigma[(w, t)] = row['Simulator Needed'] if pd.notna(row['Simulator Needed']) else 0


# duration of training t
l_t = {
    row['Training Type']: row['Week']
    for _, row in training.iterrows() if not pd.isna(row['Week'])
}

# kappa_w_tau: Capacity gained during training type tau at week w
kappa = {
    (row['Week'], row['Training Type']): 1 if not pd.isna(row['Capacity Gained']) else 0
    for _, row in training.iterrows()
}

# zeta_w_g: Cost of having g aircraft grounded in week w
zeta_w_g = {
    (row['Week'], g): row[g]
    for _, row in grounded_aircraft_cost.iterrows()
    for g in grounded_aircraft_cost.columns if isinstance(g, int) and g >= 1
}

#training types to start and end types
beta_start = {(row['Training Type'], row['Start Crew Type'].split()[1], row['Start Crew Type'].split()[0]): 1
              for _, row in training_types.iterrows()}

beta_end = {(row['Training Type'], row['End Crew Type'].split()[1], row['End Crew Type'].split()[0]): 1
            for _, row in training_types.iterrows()}

lambda_ = {row['Training Type']: row['Max Students'] for _, row in training.iterrows()};

crew_leaving = crew_leaving.rename(columns=rename_columns)

crew_leaving_dict = {}
for i, row in crew_leaving.iterrows():
    week = int(row['Week'])
    for col in crew_leaving.columns[1:]:
        numleaving = row[col]
        if pd.notna(numleaving):
            parts = col.split()
            brand = parts[0]
            rank = parts[1]
            qual = parts[int(2)]
            
            key = (week, rank, brand, qual)
            crew_leaving_dict[key] = int(numleaving)

c_52 = {
    ('Airbus', 'FO' ): 86,
    ('Airbus', 'C'): 94
}

k = {}
for _, row in initial_crew_type_qualification.iterrows():
    a, p = row['Rating'].split()
    k[(a, p, int(row['Qualification']))] = row['Number Qualified'] if pd.notna(row['Number Qualified']) else 0

l_t = training.groupby('Training Type')['Week'].max().to_dict()

instructor_need = {
    (row['Training Type'], row['Week'], row['Qualification']): row['Capacity Needed']
    for _, row in training.iterrows()
}


"""
Decision Variables
"""
model = Model("Crew Optimization")

u = model.addVars(W, tau, vtype=GRB.INTEGER, name="u")
x = model.addVars(W, tau, vtype=GRB.INTEGER, name="x")

# Auxiliary Decision Variables
m = model.addVars(W, rho, A, Q, vtype=GRB.INTEGER, name="m")
n = model.addVars(W, rho, A, Q, vtype=GRB.INTEGER, name="n")
f = model.addVars(W, rho, A, Qqual, vtype=GRB.INTEGER, name="f") # number of instructors fulfilled for training
psi = model.addVars(W, rho, A, vtype=GRB.INTEGER, name="psi")
tilde_psi = model.addVars(W, A, vtype=GRB.INTEGER, name="tilde_psi")
c = model.addVars(W_all, rho, A, Q, vtype=GRB.INTEGER, name="c")
tilde_c = model.addVars(W_all, rho, A, Q, vtype=GRB.INTEGER, name="tilde_c")
z = model.addVars(W, G, vtype=GRB.BINARY, name="z")
h = model.addVars(W_with_end, rho, A, Qall, vtype=GRB.INTEGER, name="h")

# constraint 1 - 4: pilot conversion after training
model.addConstrs(
    (c[w, p, a, q] == c[w-1, p, a, q] + 
     sum(beta_end.get((t, p, a), 0) * x[w-l_t[t], t] if w - l_t[t] >= 1 else 0 for t in tau)
         - h[w, p, a, q] 
    for p in rho for a in A for w in W_with_end for q in Qunqual), name = "training_conversion_1"
    )

model.addConstrs(
    (c[w, p, a, q] == c[w-1, p, a, q] - h[w, p, a, q]
     for p in rho for a in A for w in W_with_end for q in Qqual), name = "training_conversion_2"
    )

model.addConstrs(
    (tilde_c[w, p, a, q] == c[w, p, a, q]
     + gp.quicksum(beta_end.get((t, p, a), 0) * kappa.get((w-i+1, t), 0) * x[i, t]
       for t in tau for i in range(w - l_t[t]+1 if w - l_t[t] >= 0 else 1, w + 1))
       - crew_leaving_dict.get((w, p, a, q), 0)
     for p in rho for a in A for w in W for q in Qunqual),
    name="temporary_pilots_update_1"
)
model.addConstrs((tilde_c[w, p, a, q] == c[w, p, a, q] -  - crew_leaving_dict.get((w, p, a, q), 0) for p in rho for a in A for w in W for q in Qqual), name = "temporary_pilots_update_2")

# constraints: Training instructor (type p, aircraft a, qualification q) need
model.addConstrs(
    (n[w, p, a, q] == sum(beta_end.get((t, p, a), 0) * instructor_need.get((t,w-i+1, q), 0) * u[i, t]
                         for t in tau for i in range(w - l_t[t]+1 if w - l_t[t] >= 0 else 1, w + 1))
                         for q in Qqual for p in rho for a in A for w in W),
    name="instructor_need"
)

model.addConstrs(
    (m[w, p, a, q] == c[w, p, a, q] - f[w, p, a, q]
     for p in rho for a in A for w in W for q in Qqual), name="flying_pilots_qualified")
model.addConstrs((m[w, p, a, q] == tilde_c[w, p, a, q]
                 for p in rho for a in A for w in W for q in Qunqual),
                   name="flying_pilots_unqualified")

model.addConstrs(
    (gp.quicksum(f[w, p, a, q] for p in rho for q in Qqual) == gp.quicksum(n[w, p, a, q] for p in rho for q in Qqual) for w in W for a in A),
    name = "relaxed_training_fulfillment_1"
)
model.addConstrs((gp.quicksum(f[w, 'C', a, q] for q in range(max(Qqual), i, -1)) >= gp.quicksum(n[w, 'C', a, q] for q in range(max(Qqual), i, -1)) for w in W for a in A for i in Q), name = 'relaxed_training_fulfillment_2')

model.addConstrs((gp.quicksum(f[w, p, a, q] for q in range(max(Qqual), i, -1) for p in rho) >= gp.quicksum(n[w, p, a, q] for q in range(max(Qqual), i, -1) for p in rho) for w in W for a in A for i in Q), name = 'relaxed_training_fulfillment_3')

model.addConstrs(((u[w, t] - 1) * lambda_[t] + 1 <= x[w, t] for t in tau for w in W), name="training_capacity_lower_bound")
model.addConstrs((x[w, t] <= u[w, t] * lambda_[t] for t in tau for w in W), name="training_capacity_upper_bound")

model.addConstrs((gp.quicksum(h[w, p, a, q] for q in Qunqual) == gp.quicksum(beta_start.get((t, p, a), 0) * x[w, t] for t in tau_internal) for p in rho for a in A for w in W), name="total_trainees")
model.addConstrs((gp.quicksum(h[w, p, a, q] for q in Qqual) == 0 for p in rho for a in A for w in W), name="total_trainees_qualified")
model.addConstrs((gp.quicksum(h[w, p, a, q] for q in Qext) == gp.quicksum(beta_start.get((t, p, a), 0) * x[w, t] for t in tau_external) for p in rho for a in A for w in W), name="total_trainees_external")

model.addConstrs(
    (gp.quicksum(sigma.get((w-i+1, t), 0) * u[i, t]
     for t in tau for i in range(w - l_t[t]+1 if w - l_t[t] >= 0 else 1, w + 1)) <= xi.get(w, 0) for w in W), name="simulator_availability")

# constraints: unmet demand and grounded planes
model.addConstrs(
    (psi[w, p, a] >= Delta.get((w, a)) - gp.quicksum(m[w, p, a, q] for q in Q)
     for p in rho for a in A for w in W), name="unmet_pilot_demand")
model.addConstrs(
    (tilde_psi[w, a] >= psi[w, p, a]
     for p in rho for a in A for w in W), name="unmet_demand")
model.addConstrs((sum(z[w, g] for g in G) == 1 for w in W), name="grounding_decision")
model.addConstrs(
    (gp.quicksum(tilde_psi[w, a] for a in A) == gp.quicksum(g * z[w, g] for g in G) for w in W), name="unmet_demand_equals_grounded")

model.addConstrs((sum(u[w, t] for w in range(max(W) - l_t[t] + 1, max(W)+1)) == 0 for t in tau), name="no_training_at_end")

model.addConstr((gp.quicksum(c[max(W)+1, 'FO', 'Airbus', q] for q in Q) >= c_52[('Airbus', 'FO')]), name="eoy_fo_requirement")
model.addConstr((gp.quicksum(c[max(W)+1, 'C', 'Airbus', q] for q in Q) >= c_52[('Airbus', 'C')]), name="eoy_captain_requirement")

model.addConstrs((c[w, p, a, q] == c_0[(a, p)] - sum(k.get((a, p, q_qual), 0) for q_qual in Qqual)  for p in rho for a in A for w in W_start for q in Qunqual), name="initial_crew_qualify_1")
model.addConstrs(
   (c[w, p, a, q] == k.get((a, p, q), 0) for p in rho for a in A for w in W_start for q in Qqual), name="initial_crew_qualify_2"
)
# model.addConstrs((u[w, 5] == 0 for w in W), name="no_training_type_5")
# model.addConstrs((u[w, 4] == 0 for w in W), name="no_training_type_4")

model.update()

model.setObjective(
    quicksum(zeta_w_g.get((w, g), 0) * z[w, g] for w in W for g in G) + 1.5e5 * quicksum(x[w, 4] + x[w, 5] for w in W),
    GRB.MINIMIZE
)

model.optimize()

for w in W:
    for a in A:
        if tilde_psi[w, a].X >= 1e-6:
            print(f"Unmet demand for {a} in week {w}: {tilde_psi[w, a].X}")
    for t in tau:
        if x[w, t].X >= 1e-6:
            print(f"x[{w}, {t}]: {x[w, t].X}")
            print(f"u[{w}, {t}]: {u[w, t].X}")
    # for p in rho:
    #     for a in A:
    #         for q in Qqual:
    #                 print(f"c[{w}, {p}, {a}, {q}]: {c[w, p, a, q].X}")

model.write('model.lp')
# model.computeIIS()
# model.write('infeasibility.ilp')


output_file = "results_output/pilot_output.xlsx"
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

# Iterate over (p, a) combinations to create separate sheets
for (p, a) in product(rho, A):
    # Create an empty DataFrame with weeks as columns
    df = pd.DataFrame(columns=[f"Week {w}" for w in W])

    # Iterate over q values and include multiple variables (c, n, h)
    row_data = []

    # Iterate over q values and include multiple variables (c, n, h)
    for q in Q:
        for var_name, var_dict in [("c", c), ("tilde_c", tilde_c), ("m", m), ("h", h), ("f", f), ("n", n)]:
            row_label = f"{var_name}_{{{p},{a},{q}}}"
            row_values = [
                var_dict[w, p, a, q].X if (w, p, a, q) in var_dict else 0
                for w in W_with_end
            ]
            row_data.append([row_label] + row_values)

        # Insert an empty row after each `q` group
        row_data.append([""] + [""] * len(W))

    for q in Qext:
        for var_name, var_dict in [("c", c), ("h", h)]:
            row_label = f"{var_name}_{{{p},{a},{q}}}"
            row_values = [
                var_dict[w, p, a, q].X if (w, p, a, q) in var_dict else 0
                for w in W_with_end
            ]
            row_data.append([row_label] + row_values)

        # Insert an empty row after each `q` group
        row_data.append([""] + [""] * len(W))


    # Save this DataFrame to a separate sheet named after (p, a)
    df = pd.DataFrame(row_data, columns=["Variable"] + [f"Week {w}" for w in W_with_end])
    sheet_name = f"{p}_{a}"
    df.to_excel(writer, sheet_name=sheet_name)

# Save the Excel file
writer.close()


txt_filename = "results_output/training_output.txt"
with open(txt_filename, "w") as f:
    for w in W:
        for t in tau:
            if x[w, t].X >= 1e-6:
                f.write(f"x[{w}, {t}] {x[w, t].X}\n")
                f.write(f"u[{w}, {t}] {u[w, t].X}\n")
