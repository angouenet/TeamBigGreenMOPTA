import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from datetime import datetime, timedelta
from alandata import load_data

st.set_page_config(layout="wide")
st.header("Optimization Results")

alandata = load_data()
crew_demand = alandata["crew_demand"]

folder_path = "results_output/"
baseline_file = "baseline_pilot_output.xlsx"

pilot_files = [
    f for f in os.listdir(folder_path) 
    if f.startswith("pilot_output_") and f.endswith(".xlsx")]

#Sorted

def sort_key(filename):
    parts = filename.replace("pilot_output_", "").replace(".xlsx", "").split("_")
    return (float(parts[0]), float(parts[1]))

pilot_files_sorted = sorted(pilot_files, key=sort_key)

excel_files = [baseline_file] + pilot_files_sorted

display_names = ["Baseline"]
for file in pilot_files_sorted:
    parts = file.replace("pilot_output_", "").replace(".xlsx", "").split("_")
    grounding_coeff, hire_cost = parts[0], parts[1]
    display_names.append(f"Training (Grounding Coefficient={grounding_coeff}, External Cost={float(hire_cost):,.0f})")

excel_file_paths = [os.path.join(folder_path, file) for file in excel_files]

file_map = dict(zip(display_names, excel_file_paths))
selected_display_name = st.selectbox("Select the Excel File", display_names)
selected_file = file_map[selected_display_name]

xls = pd.ExcelFile(selected_file)

optimization_results = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

name_mapping = {
    'FO_Boeing': 'Boeing First Officer',
    'FO_Airbus': 'Airbus First Officer',
    'C_Boeing': 'Boeing Captain',
    'C_Airbus': 'Airbus Captain'}

pretty_results = {name_mapping[key]: df for key, df in optimization_results.items()}

def rename_qualification(var_name):
    """Convert variable names to qualification labels"""
    if var_name.endswith(',0}'):
        return 'Qualification 0'
    elif var_name.endswith(',1}'):
        return 'Qualification 1'
    elif var_name.endswith(',2}'):
        return 'Qualification 2'
    elif var_name.endswith(',3}'):
        return 'Qualification 3'
    return var_name

def process_dataframe(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    ### This is where to change if we change the naming convention of different stuff ###
    filtered_df = df[df["Variable"].fillna("").str.startswith("m_")].copy()
    filtered_df['Variable'] = filtered_df['Variable'].apply(rename_qualification)
    #if len(filtered_df) >= 4:
    #    filtered_df.iloc[0, 1:] = filtered_df.iloc[0, 1:] - filtered_df.iloc[1:4, 1:].sum(axis=0)
    week_columns = [col for col in filtered_df.columns if col.startswith('Week')]
    filtered_df = filtered_df[~(filtered_df[week_columns].eq(0).all(axis=1))]
    return filtered_df

def create_stacked_qual_chart(df, title):

    melted_df = df.melt(id_vars=['Variable'], var_name='Week', value_name='Count')

    weeks = sorted(melted_df['Week'].unique(), key=lambda x: int(x.split()[-1]))
    quals = ["Qualification 0", "Qualification 1", "Qualification 2", "Qualification 3"] # Omit "External Hire" - It's in training gantt
    
    fig = go.Figure()
    
    for qual in quals:
        qual_data = melted_df[melted_df['Variable'] == qual]
        fig.add_trace(go.Bar(
            x=weeks,
            y=[qual_data[qual_data['Week'] == week]['Count'].sum() for week in weeks],
            name=qual,
            hoverinfo='y+name'))
    
    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title='Week',
        yaxis_title='Number of Crew',
        legend_title='Qualification Level',
        height=600)
    return fig

def create_total_vs_demand_chart(allocation_df, demand_df, title):
    alloc_renamed = allocation_df.rename(columns={
        col: col.replace('Week ', '') for col in allocation_df.columns if col.startswith('Week ')})
    total_alloc = alloc_renamed.drop(columns=['Variable']).sum().reset_index()
    total_alloc.columns = ['Week', 'Total Allocation']
    
    demand_melted = demand_df.copy()
    demand_melted['Week'] = demand_melted['Week'].astype(str)
    
    combined = pd.merge(total_alloc, demand_melted, on='Week')
    combined = combined.melt(id_vars=['Week'], 
                           value_vars=['Total Allocation', 'Demand'],
                           var_name='Type',
                           value_name='Count')
    
    fig = px.bar(combined,
                x='Week',
                y='Count',
                color='Type',
                title=title,
                barmode='group',
                color_discrete_map={
                    'Total Allocation': '#636EFA',
                    'Demand': '#FF5733'})
    fig.update_layout(
        xaxis_title='Week',
        yaxis_title='Number of Crew',
        legend_title='Comparison',
        height=600)
    return fig

demand_wide = crew_demand.pivot(index='Week', columns='Aircraft', values='Demand').reset_index()
demand_wide.columns = ['Week', 'Airbus_Demand', 'Boeing_Demand']
boeing_demand = demand_wide[['Week', 'Boeing_Demand']].rename(columns={'Boeing_Demand': 'Demand'})
airbus_demand = demand_wide[['Week', 'Airbus_Demand']].rename(columns={'Airbus_Demand': 'Demand'})

### FO ###

st.header("First Officer Analysis")
fo_tab1, fo_tab2, fo_tab3, fo_tab4 = st.tabs([
    "Boeing Qualifications", 
    "Airbus Qualifications",
    "Boeing vs Demand",
    "Airbus vs Demand"])

with fo_tab1:
    processed_df = process_dataframe(pretty_results['Boeing First Officer'])
    st.dataframe(processed_df)
    fig = create_stacked_qual_chart(processed_df, "Boeing FO: Cohort Eligible to Fly per Week")
    st.plotly_chart(fig, use_container_width=True)

with fo_tab2:
    processed_df = process_dataframe(pretty_results['Airbus First Officer'])
    st.dataframe(processed_df)
    fig = create_stacked_qual_chart(processed_df, "Airbus FO: Cohort Eligible to Fly per Week")
    st.plotly_chart(fig, use_container_width=True)

with fo_tab3:
    processed_df = process_dataframe(pretty_results['Boeing First Officer'])
    st.dataframe(processed_df)
    fig = create_total_vs_demand_chart(processed_df, boeing_demand, "Boeing FO: Total Allocation vs Demand")
    st.plotly_chart(fig, use_container_width=True)

with fo_tab4:
    processed_df = process_dataframe(pretty_results['Airbus First Officer'])
    st.dataframe(processed_df)
    fig = create_total_vs_demand_chart(processed_df, airbus_demand, "Airbus FO: Total Allocation vs Demand")
    st.plotly_chart(fig, use_container_width=True)

### Captain ###
st.header("Captain Analysis")
cap_tab1, cap_tab2, cap_tab3, cap_tab4 = st.tabs([
    "Boeing Qualifications", 
    "Airbus Qualifications",
    "Boeing vs Demand",
    "Airbus vs Demand"])

with cap_tab1:
    processed_df = process_dataframe(pretty_results['Boeing Captain'])
    st.dataframe(processed_df)
    fig = create_stacked_qual_chart(processed_df, "Boeing Captain: Cohort Eligible to Fly per Week")
    st.plotly_chart(fig, use_container_width=True)

with cap_tab2:
    processed_df = process_dataframe(pretty_results['Airbus Captain'])
    st.dataframe(processed_df)
    fig = create_stacked_qual_chart(processed_df, "Airbus Captain: Cohort Eligible to Fly per Week")
    st.plotly_chart(fig, use_container_width=True)

with cap_tab3:
    processed_df = process_dataframe(pretty_results['Boeing Captain'])
    st.dataframe(processed_df)
    fig = create_total_vs_demand_chart(processed_df, boeing_demand, "Boeing Captain: Total Allocation vs Demand")
    st.plotly_chart(fig, use_container_width=True)

with cap_tab4:
    processed_df = process_dataframe(pretty_results['Airbus Captain'])
    st.dataframe(processed_df)
    fig = create_total_vs_demand_chart(processed_df, airbus_demand, "Airbus Captain: Total Allocation vs Demand")
    st.plotly_chart(fig, use_container_width=True)


### Grounded Aircraft ###



def create_grounded_chart(allocation_dfs, demand_df):
    shortages = []
    
    for aircraft_type, df in allocation_dfs.items():
        alloc = df.rename(columns={
            col: col.replace('Week ', '') for col in df.columns if col.startswith('Week ')})
        
        total_alloc = alloc.drop(columns=['Variable']).sum().reset_index()
        total_alloc.columns = ['Week', 'Total_Allocation']
        total_alloc['Week'] = total_alloc['Week'].astype(int)
        
        demand_col = f'{aircraft_type}_Demand'
        weekly_demand = demand_df[['Week', demand_col]].rename(columns={demand_col: 'Demand'})
        
        merged = pd.merge(total_alloc, weekly_demand, on='Week')
        merged['Shortage'] = merged['Demand'] - merged['Total_Allocation']
        merged['Aircraft'] = aircraft_type
        
        merged = merged[merged['Shortage'] > 0]
        shortages.append(merged)
    
    if not shortages:
        return None
    
    combined = pd.concat(shortages)
    
    fig = px.bar(combined,
                 x='Week',
                 y='Shortage',
                 color='Aircraft',
                 title='Grounded Aircraft by Week',
                 labels={'Shortages': 'Number of Grounded Aircraft'},
                 color_discrete_map={
                     'Boeing': '#636EFA',
                     'Airbus': '#00CC96'})
    
    fig.update_layout(
        xaxis_title='Week',
        yaxis_title='Number of Grounded Aircraft',
        legend_title='Aircraft Type',
        hovermode='x unified',
        height=500,
        title_font=dict(size=20),
        font=dict(size=14),
        legend=dict(
            title_font=dict(size=16),
            font=dict(size=14)),
        xaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=14)),
        yaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=14)))
    
    return fig

st.header("Grounded Aircraft Analysis")

allocation_dfs = {
    'Boeing': process_dataframe(pretty_results['Boeing First Officer']),
    'Boeing': process_dataframe(pretty_results['Boeing Captain']),
    'Airbus': process_dataframe(pretty_results['Airbus Captain']),
    'Airbus': process_dataframe(pretty_results['Airbus First Officer'])}

grounded_fig = create_grounded_chart(allocation_dfs, demand_wide)
st.markdown("### Hover for more information!")
st.plotly_chart(grounded_fig, use_container_width=True)

### Training Data ###
training = alandata["training"]

grouped_training = training.groupby("Training Type")["Week"].max().reset_index()
grouped_training.rename(columns={"Week": "Duration"}, inplace=True)

category_training = alandata["training_types"]

st.header("Training Data")

def parse_training_data(file_path):
    trainings = []  # U
    trainees = []   # X
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip() 
            if line.startswith('u['):
                # Use regex to extract week, training type, and number of trainings
                match = re.match(r'u\[(\d+),\s*(\d+)\]\s+([\d.]+)', line)
                if match:
                    week = int(match.group(1))
                    training_type = int(match.group(2))
                    num_trainings = float(match.group(3))
                    trainings.append({
                        'Week': week,
                        'Training Type': training_type,
                        'Num Trainings': num_trainings})
            elif line.startswith('x['):
                match = re.match(r'x\[(\d+),\s*(\d+)\]\s+([\d.]+)', line)
                if match:
                    week = int(match.group(1))
                    training_type = int(match.group(2))
                    num_trainees = float(match.group(3))
                    trainees.append({
                        'Week': week,
                        'Training Type': training_type,
                        'Num Trainees': num_trainees})
    return pd.DataFrame(trainings), pd.DataFrame(trainees)

folder_path = "results_output/"

training_files = []
display_names = []

baseline_file = "baseline_training_output.txt"
training_files.append(baseline_file)
display_names.append("Baseline Training Output")

files_with_params = []
for file in os.listdir(folder_path):
    if file.startswith("training_output_") and file.endswith(".txt"):
        params = file.replace("training_output_", "").replace(".txt", "").split("_")
        if len(params) == 2: 
            grounding_coeff, hire_cost = float(params[0]), float(params[1])
            files_with_params.append((grounding_coeff, hire_cost, file))

files_with_params.sort(key=lambda x: (x[0], x[1]))

for grounding_coeff, hire_cost, file in files_with_params:
    training_files.append(file)
    display_names.append(f"Training (Grounding Coefficient={grounding_coeff}, External Cost={hire_cost:,.0f})")

txt_map = dict(zip(display_names, [os.path.join(folder_path, f) for f in training_files]))

if "training_output.txt" in os.listdir(folder_path):
    old_training_entry = {"Old Training Output": os.path.join(folder_path, "training_output.txt")}
    txt_map = {**old_training_entry, **txt_map}

txt_display_name = st.selectbox("Select the TXT File", list(txt_map.keys()))
selected_txt_file = txt_map[txt_display_name]

training_df, trainee_df = parse_training_data(selected_txt_file)

#print("Trainings DataFrame:")
#print(training_df)
#print("\nTrainees DataFrame:")
#print(trainee_df)

st.subheader("Training Schedule")

schedule_df = training_df.merge(trainee_df, on=['Week', 'Training Type'], how='left') \
                          .merge(grouped_training, on='Training Type', how='left') \
                          .merge(category_training, on='Training Type', how='left') 

schedule_df['Transition'] = schedule_df['Start Crew Type'] + " → " + schedule_df['End Crew Type']

schedule_df = schedule_df[[
    'Week', 'Training Type', 'Duration', 'Num Trainings', 'Num Trainees', 'Transition']]

st.dataframe(schedule_df.sort_values('Week'))

### GANTT ###

start_date_2024 = datetime(2024, 1, 1)

def calculate_dates(row):
    start_week = row['Week'] - 1
    duration_weeks = row['Duration']
    start_date = start_date_2024 + timedelta(weeks=start_week)
    end_date = start_date + timedelta(weeks=duration_weeks) - timedelta(days=1)
    return pd.Series([start_date, end_date])

schedule_df[['Start Date', 'End Date']] = schedule_df.apply(calculate_dates, axis=1)

custom_order = [
    'Boeing FO → Airbus FO',
    'Boeing C → Airbus C', 
    'Boeing FO → Boeing C',
    'External Boeing FO → Boeing FO']

def assign_vertical_positions(df):
    df = df.sort_values(['Start Date', 'sort_key'])
    positions = []
    active_trainings = []  # List of end dates
    
    for _, row in df.iterrows():
        start = row['Start Date']
        end = row['End Date']
        
        # Find the first available position where there's no overlap
        pos = 0
        while True:
            # Check if this position is available
            available = True
            for existing_end, existing_pos in active_trainings:
                if existing_pos == pos and existing_end >= start:
                    available = False
                    break
            
            if available:
                break
            pos += 5
        
        positions.append(pos)
        
        # Add to active trainings and clean up finished ones
        active_trainings.append((end, pos))
        active_trainings = [(e, p) for e, p in active_trainings if e >= start]
    
    return positions

priority_map = {t: i for i, t in enumerate(custom_order)}
schedule_df['sort_key'] = schedule_df['Transition'].map(priority_map)
schedule_df = schedule_df.sort_values('sort_key')
schedule_df['Vertical Position'] = assign_vertical_positions(schedule_df)

max_positions = schedule_df.groupby('Transition')['Vertical Position'].max()
schedule_df['Y Value'] = (
    schedule_df['sort_key'] * 1 + 
    schedule_df['Vertical Position'] * 0.15)

schedule_df['Start Week'] = schedule_df['Week']
schedule_df['End Week'] = schedule_df['Week'] + schedule_df['Duration'] - 1

fig = px.timeline(
    schedule_df,
    x_start="Start Date",
    x_end="End Date",
    y="Y Value",
    color="Transition",
    color_discrete_map={t: px.colors.qualitative.Plotly[i] for i, t in enumerate(custom_order)},
    hover_name="Transition",
    hover_data={
        "Start Week": True,
        "End Week": True,
        "Duration": True,
        "Num Trainees": True,
        "Start Date": False,
        "End Date": False,
        "Y Value": False
    },
    #text="Num Trainees", # just for the pic
    title="Training Schedule Projected Onto 2024")

fig.update_yaxes(
    autorange="reversed",
    #tickvals=list(range(len(custom_order))),
    #ticktext=custom_order,
    showgrid=True,
    showticklabels=False,  # This ensures the transition labels are shown
    # Remove numeric labels by hiding side ticks
    side='right',        # Moves the labels to the right side (optional)
    showline=False,      # Hides the axis line
    ticks=''            # Hides the tick marks
)
fig.update_layout(
    height=600,
    xaxis_title="Timeline",
    yaxis_title='Training Transition',
    showlegend=True,
    margin=dict(l=100, r=50, b=100, t=100),
    plot_bgcolor='white')

fig.update_traces(width=0.1,
                  #textposition = 'inside',
                  #textfont=dict(size = 12, color = 'white')
)

st.plotly_chart(fig, use_container_width=True)
