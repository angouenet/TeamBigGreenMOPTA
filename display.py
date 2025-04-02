import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime, timedelta
from alandata import load_data

st.set_page_config(layout="wide")

alandata = load_data()
crew_demand = alandata["crew_demand"]

folder_path = "results_output/"
excel_files = [
    "pilot_output.xlsx",
    "baseline_pilot_output.xlsx",
    "optimization_results2.xlsx",
    "optimization_results3.xlsx"
]

excel_file_paths = [folder_path + file for file in excel_files]
display_names = ["Old Pilot Output", "Basline Pilot Output", "Optimization Result 3", "Optimization Result 4"]

file_map = dict(zip(display_names, excel_file_paths))
selected_display_name = st.selectbox("Select the Excel File", display_names)

selected_file = file_map[selected_display_name]

xls = pd.ExcelFile(selected_file)

optimization_results = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

name_mapping = {
    'FO_Boeing': 'Boeing First Officer',
    'FO_Airbus': 'Airbus First Officer',
    'C_Boeing': 'Boeing Captain',
    'C_Airbus': 'Airbus Captain'
}

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
            hoverinfo='y+name'
        ))
    
    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title='Week',
        yaxis_title='Number of Crew',
        legend_title='Qualification Level',
        height=600
    )
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

st.header("First Officer Data")
fo_tab1, fo_tab2, fo_tab3, fo_tab4 = st.tabs([
    "Boeing Qualifications", 
    "Airbus Qualifications",
    "Boeing vs Demand",
    "Airbus vs Demand"
])

with fo_tab1:
    processed_df = process_dataframe(pretty_results['Boeing First Officer'])
    st.dataframe(processed_df)
    fig = create_stacked_qual_chart(processed_df, "Boeing FO: Qualification Over Time")
    st.plotly_chart(fig, use_container_width=True)

with fo_tab2:
    processed_df = process_dataframe(pretty_results['Airbus First Officer'])
    st.dataframe(processed_df)
    fig = create_stacked_qual_chart(processed_df, "Airbus FO: Qualification Over Time")
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
st.header("Captain Data")
cap_tab1, cap_tab2, cap_tab3, cap_tab4 = st.tabs([
    "Boeing Qualifications", 
    "Airbus Qualifications",
    "Boeing vs Demand",
    "Airbus vs Demand"
])

with cap_tab1:
    processed_df = process_dataframe(pretty_results['Boeing Captain'])
    st.dataframe(processed_df)
    fig = create_stacked_qual_chart(processed_df, "Boeing Captain: Qualification Over Time")
    st.plotly_chart(fig, use_container_width=True)

with cap_tab2:
    processed_df = process_dataframe(pretty_results['Airbus Captain'])
    st.dataframe(processed_df)
    fig = create_stacked_qual_chart(processed_df, "Airbus Captain: Qualification Over Time")
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
            col: col.replace('Week ', '') for col in df.columns if col.startswith('Week ')
        })
        
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
                     'Airbus': '#00CC96'
                 })
    
    fig.update_layout(
        xaxis_title='Week',
        yaxis_title='Number of Grounded Aircraft',
        legend_title='Aircraft Type',
        hovermode='x unified',
        height=500
    )
    
    return fig

st.header("Grounded Aircraft Analysis")

allocation_dfs = {
    'Boeing': process_dataframe(pretty_results['Boeing First Officer']),
    'Boeing': process_dataframe(pretty_results['Boeing Captain']),
    'Airbus': process_dataframe(pretty_results['Airbus Captain']),
    'Airbus': process_dataframe(pretty_results['Airbus First Officer'])
}

grounded_fig = create_grounded_chart(allocation_dfs, demand_wide)
st.plotly_chart(grounded_fig, use_container_width=True)

### Training Data ###
training = alandata["training"]

grouped_training = training.groupby("Training Type")["Week"].max().reset_index()
grouped_training.rename(columns={"Week": "Duration"}, inplace=True)

category_training = alandata["training_types"]

st.header("Training Data Tables")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Training Duration")
    st.dataframe(grouped_training)
with col2:
    st.subheader("Training Types")
    st.dataframe(category_training)

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
                        'Num Trainings': num_trainings
                    })
            elif line.startswith('x['):
                match = re.match(r'x\[(\d+),\s*(\d+)\]\s+([\d.]+)', line)
                if match:
                    week = int(match.group(1))
                    training_type = int(match.group(2))
                    num_trainees = float(match.group(3))
                    trainees.append({
                        'Week': week,
                        'Training Type': training_type,
                        'Num Trainees': num_trainees
                    })
    return pd.DataFrame(trainings), pd.DataFrame(trainees)


folder_path = "results_output/"
txt_files = [
    "training_output.txt",
    "baseline_training_output.txt",
    "optimization_results2.xlsx",
    "optimization_results3.xlsx"
]

txt_file_paths = [folder_path + file for file in txt_files]
txt_names = ["Old Training Output", "Basline Training Output", "Optimization Result 3", "Optimization Result 4"]

txt_map = dict(zip(txt_names, txt_file_paths))
txt_display_name = st.selectbox("Select the TXT File", txt_names)
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
    'Week', 'Training Type', 'Duration', 'Num Trainings', 'Num Trainees', 'Transition'
]]

st.dataframe(schedule_df.sort_values('Week'))

### GANTT ###

# Calculate dates (your existing code)
start_date_2024 = datetime(2024, 1, 1)

def calculate_dates(row):
    start_week = row['Week'] - 1
    duration_weeks = row['Duration']
    start_date = start_date_2024 + timedelta(weeks=start_week)
    end_date = start_date + timedelta(weeks=duration_weeks) - timedelta(days=1)
    return pd.Series([start_date, end_date])

schedule_df[['Start Date', 'End Date']] = schedule_df.apply(calculate_dates, axis=1)

# Custom ordering
custom_order = [
    'Boeing FO → Airbus FO',
    'Boeing C → Airbus C', 
    'Boeing FO → Boeing C',
    'External Boeing FO → Boeing FO'
]

# Vertical positioning algorithm (optimized)
def assign_vertical_positions(df):
    df = df.sort_values(['sort_key', 'Start Date'])
    positions = []
    active_trainings = {}  # {transition: [end_dates]}
    
    for _, row in df.iterrows():
        start = row['Start Date']
        end = row['End Date']
        transition = row['Transition']
        
        if transition not in active_trainings:
            active_trainings[transition] = []
        
        # Find first available position (0 if no overlap)
        pos = 0
        for existing_end in sorted(active_trainings[transition]):
            if existing_end < start:
                break
            pos += 1
        
        positions.append(pos)
        active_trainings[transition].append(end)
        
        # Clean up finished trainings
        active_trainings[transition] = [e for e in active_trainings[transition] if e >= start]
    
    return positions

# Apply positioning
priority_map = {t: i for i, t in enumerate(custom_order)}
schedule_df['sort_key'] = schedule_df['Transition'].map(priority_map)
schedule_df = schedule_df.sort_values('sort_key')
schedule_df['Vertical Position'] = assign_vertical_positions(schedule_df)

# Dynamic Y-axis spacing
max_positions = schedule_df.groupby('Transition')['Vertical Position'].max()
schedule_df['Y Value'] = (
    schedule_df['sort_key'] +  # Base position
    schedule_df['Vertical Position'] * 0.5  # Small offset only when needed
)

schedule_df['Start Week'] = schedule_df['Week']
schedule_df['End Week'] = schedule_df['Week'] + schedule_df['Duration'] - 1

# Visualization
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
        "Num Trainees": True
    },

    title="Training Schedule Projected Onto 2024"
)

# Y-axis configuration
fig.update_yaxes(
    autorange="reversed",
    tickvals=list(range(len(custom_order))),
    ticktext=custom_order,
    showgrid=True
)

# Layout adjustments
fig.update_layout(
    height=600,
    xaxis_title="Timeline",
    yaxis_title="Training Transition",
    showlegend=True,
    margin=dict(l=100, r=50, b=100, t=100),
    plot_bgcolor='white'
)

# Consistent bar width
fig.update_traces(width=0.25)

st.plotly_chart(fig, use_container_width=True)