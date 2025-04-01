import pandas as pd

def load_data():
    file_path = 'CrewTrainingData.xlsx'
    
    initial_crew = pd.read_excel(file_path, sheet_name='Initial Crew')
    
    # Load data from various sheets in the Excel file
    initial_crew = pd.read_excel(file_path, sheet_name="Initial Crew")
    initial_crew_type_qualification = pd.read_excel(file_path, sheet_name='Initial Crew Type Qualification')
    crew_demand = pd.read_excel(file_path, sheet_name='Crew Demand')
    grounded_aircraft_cost = pd.read_excel(file_path, sheet_name='Grounded Aircraft Cost')
    crew_leaving = pd.read_excel(file_path, sheet_name='Crew Leaving')
    airbus_crew_eoy_requirement = pd.read_excel(file_path, sheet_name='Airbus Crew EOY Requirement')
    training_types = pd.read_excel(file_path, sheet_name='Training Types')
    training = pd.read_excel(file_path, sheet_name='Training')
    simulator_ability = pd.read_excel(file_path, sheet_name='Simulator Availability')
    
    
    training = training.rename(columns={"Week of Training": "Week"})
    
    grounded_aircraft_cost = grounded_aircraft_cost.rename(columns={"Unnamed: 0": "Week"})
    crew_leaving = crew_leaving.rename(columns={"Unnamed: 0": "Week"})
    simulator_ability = simulator_ability.rename(columns={"Unnamed: 0": "Week"})
    
    crew_demand["Week"] = crew_demand["Week"].str.extract(r'(\d+)').astype(int)
    grounded_aircraft_cost["Week"] = grounded_aircraft_cost["Week"].str.extract(r'(\d+)').astype(int)
    crew_leaving["Week"] = crew_leaving["Week"].str.extract(r'(\d+)').astype(int)
    simulator_ability["Week"] = simulator_ability["Week"].str.extract(r'(\d+)').astype(int)
    
    return {
        "initial_crew": initial_crew,
        "initial_crew_type_qualification": initial_crew_type_qualification,
        "crew_demand": crew_demand,
        "grounded_aircraft_cost": grounded_aircraft_cost,
        "crew_leaving": crew_leaving,
        "airbus_crew_eoy_requirement": airbus_crew_eoy_requirement,
        "training_types": training_types,
        "training": training,
        "simulator_ability": simulator_ability
    }

