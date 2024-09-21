import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Patch


def draw_wavy_break(ax, position, wavy_width, wavy_height):
    x = np.linspace(-wavy_width, wavy_width, 100)
    y = np.sin(3*x/wavy_width) * wavy_height + position
    ax.plot(x, y, transform=ax.get_yaxis_transform(), color='k', clip_on=False)

# Convert data to pandas DataFrames
# tasks_df = pd.DataFrame(tasks_data.items(), columns=['Task', 'Count']).sort_values(by='Count', ascending=False)
# objects_df = pd.DataFrame(objects_data.items(), columns=['Object', 'Count']).sort_values(by='Count', ascending=False)
# spatial_df = pd.DataFrame(spatial_data.items(), columns=['Spatial Relation', 'Count']).sort_values(by='Count', ascending=False)
# spatial_df = spatial_df[spatial_df['Spatial Relation'] != 'None']

# bridge_tasks_df = pd.DataFrame(bridge_tasks_data.items(), columns=['Task', 'Count']).sort_values(by='Count', ascending=False)
# bridge_objects_df = pd.DataFrame(bridge_objects_data.items(), columns=['Object', 'Count']).sort_values(by='Count', ascending=False)
# bridge_spatial_df = pd.DataFrame(bridge_spatial_data.items(), columns=['Spatial Relation', 'Count']).sort_values(by='Count', ascending=False)
# bridge_spatial_df = bridge_spatial_df[bridge_spatial_df['Spatial Relation'] != 'None']

# Calculate maximum y-value for each category
# max_tasks_count = max(tasks_df['Count'].max(), bridge_tasks_df['Count'].max())
# max_objects_count = max(objects_df['Count'].max(), bridge_objects_df['Count'].max())
# max_spatial_count = max(spatial_df['Count'].max(), bridge_spatial_df['Count'].max())
def plot_learning_rate():
    # Load data from JSON files
    # Paths to the CSV files
    csv_1_path = os.path.join(data_path, 'csv_1.csv')
    csv_2_path = os.path.join(data_path, 'csv_2.csv')
    csv_3_path = os.path.join(data_path, 'csv_3.csv')

    # Load the CSV files into pandas dataframes
    df1 = pd.read_csv(csv_1_path)
    df2 = pd.read_csv(csv_2_path)
    df3 = pd.read_csv(csv_3_path)

    # Rename the learning rate columns to a unified name "learning_rate"
    df1 = df1.rename(columns={'simpler_siglip_finetune - learning rate': 'learning_rate'})
    df2 = df2.rename(columns={'simpler_siglip_finetune_1 - learning rate': 'learning_rate'})
    df3 = df3.rename(columns={'simpler_siglip_finetune_2 - learning rate': 'learning_rate'})

    # Find the maximum step in each DataFrame
    max_step_df1 = df1['Step'].max()
    max_step_df2 = df2['Step'].max()

    # Adjust the steps in df2 and df3
    df2['Step'] += max_step_df1
    df3['Step'] += (max_step_df1 + max_step_df2)

    combined_df = pd.concat([df1[['Step', 'learning_rate']], 
                             df2[['Step', 'learning_rate']], 
                             df3[['Step', 'learning_rate']]], ignore_index=True)

    # Verify the column names of the combined DataFrame
    print(combined_df.columns)

    fontsize = 20

    # Plot the unified learning rates over steps
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(combined_df['Step'], combined_df['learning_rate'], label='Learning Rate', color='b', alpha=0.7)

    # Customize the plot
    ax.set_title('Learning Rate over Steps', fontsize=fontsize)
    ax.set_xlabel('Step', fontsize=fontsize)
    ax.set_ylabel('Learning Rate', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize - 4)
    ax.tick_params(axis='x', labelsize=fontsize - 4)
    ax.legend(fontsize=fontsize)
    ax.set_xlim(0, 320000)
    ax.set_ylim(0, 0.00011)
    # fig.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/learning_rate.pdf')

def plot_simpler_results(file_openvla, file_medit_low, file_medit_high, file_octo_base, file_rt1_x, name, save_name, fontsize):
    
    df_openvla = pd.read_csv(file_openvla)
    df_medit_low = pd.read_csv(file_medit_low)
    df_medit_high = pd.read_csv(file_medit_high)
    df_octo_base = pd.read_csv(file_octo_base)
    df_rt1_x = pd.read_csv(file_rt1_x)

    # Strip whitespace from 'Type' values in each DataFrame
    df_openvla['Type'] = df_openvla['Type'].str.strip()
    df_medit_low['Type'] = df_medit_low['Type'].str.strip()
    df_medit_high['Type'] = df_medit_high['Type'].str.strip()
    df_octo_base['Type'] = df_octo_base['Type'].str.strip()
    df_rt1_x['Type'] = df_rt1_x['Type'].str.strip()

    # Prepare data for plotting, converting values to percentages
    layouts = df_openvla['Layout']
    values_openvla = df_openvla['Value'] * 100  # Convert to percentages
    values_medit_low = df_medit_low['Value'] * 100  # Convert to percentages
    values_medit_high = df_medit_high['Value'] * 100  # Convert to percentages
    values_octo_base = df_octo_base['Value'] * 100  # Convert to percentages
    values_rt1_x = df_rt1_x['Value'] * 100  # Convert to percentages

    # Determine the indices for each type to apply background colors
    sim_variant_indices = [i for i, t in enumerate(df_openvla['Type']) if t == 'sim variant']
    visual_matching_indices = [i for i, t in enumerate(df_openvla['Type']) if t in ['visual matching', 'sim visual matching']]

    # Create the plot with background colors for types
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define the width of each bar and spacing
    bar_width = 0.15
    x = np.arange(len(layouts))

    # Plot each set of bars with new datasets
    bars_medit_low = ax.bar(x - 2 * bar_width, values_medit_low, bar_width, label='MeDIt-Low', color='blue')
    bars_medit_high = ax.bar(x - bar_width, values_medit_high, bar_width, label='MeDIt-High', color='orange')
    bars_octo_base =   ax.bar(x, values_octo_base, bar_width, label='Octo Base', color='purple')
    bars_rt1_x =      ax.bar(x + bar_width, values_rt1_x, bar_width, label='RT-1-X', color='red')
    bars_openvla =   ax.bar(x + 2 * bar_width, values_openvla, bar_width, label='OpenVLA', color='green')

    # Add background color bands for different types
    for index in sim_variant_indices:
        ax.axvspan(index - 0.5, index + 0.5, color='lightgrey', alpha=0.5)

    for index in visual_matching_indices:
        ax.axvspan(index - 0.5, index + 0.5, color='lightblue', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Task Description', fontsize=fontsize)
    ax.set_ylabel('Success Rate (%)', fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(layouts, rotation=45, ha="right")

    # Add a legend for colors
    legend1 = ax.legend(title='Model', loc='upper left')

    # Create custom handles for background colors
    background_handles = [
        Patch(facecolor='silver', edgecolor='none', label='Variant Aggregation'),
        Patch(facecolor='skyblue', edgecolor='none', label='Visual Matching')
    ]

    legend2 = ax.legend(handles=background_handles, title='Type', loc='upper right')
    ax.add_artist(legend1)

    # Add grid for better readability
    # ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for each bar
    for bars in [bars_medit_low, bars_medit_high, bars_octo_base, bars_openvla, bars_rt1_x]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.0f}%', ha='center', va='bottom', fontsize=8)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/' + save_name)

def plot_coke_can_and_near():
    # Load the CSV files into pandas DataFrames
    file_openvla =  os.path.join(data_path, 'pick_coke_can_openvla.csv')
    file_medit_low =  os.path.join(data_path, 'pick_coke_can_medit_low.csv')
    file_medit_high =  os.path.join(data_path, 'pick_coke_can_medit_high.csv')
    file_octo_base =  os.path.join(data_path, 'pick_coke_can_Octo_Base.csv')
    file_rt1_x =  os.path.join(data_path, 'pick_coke_can_RT-1-X.csv')
    name = 'Pick Coke Can & Move Near'
    save_name = 'coke_can_and_near.pdf'
    fontsize = 20
    plot_simpler_results(file_openvla, file_medit_low, file_medit_high, file_octo_base, file_rt1_x, name, save_name, fontsize)

def plot_drawer():
    # Load the CSV files into pandas DataFrames
    file_openvla =  os.path.join(data_path, 'drawer_openvla.csv')
    file_medit_low =  os.path.join(data_path, 'drawer_medit_low.csv')
    file_medit_high =  os.path.join(data_path, 'drawer_medit_high.csv')
    file_octo_base =  os.path.join(data_path, 'drawer_Octo_Base.csv')
    file_rt1_x =  os.path.join(data_path, 'drawer_RT-1-X.csv')
    name = 'Drawer Tasks'
    save_name = 'drawer.pdf'
    fontsize = 20
    plot_simpler_results(file_openvla, file_medit_low, file_medit_high, file_octo_base, file_rt1_x, name, save_name, fontsize)

def plot_bridge():
    # Load the CSV files into pandas DataFrames
    file_openvla =  os.path.join(data_path, 'bridge_openvla.csv')
    file_medit_low =  os.path.join(data_path, 'bridge_medit_low.csv')
    file_medit_high =  os.path.join(data_path, 'bridge_medit_high.csv')
    file_octo_base =  os.path.join(data_path, 'bridge_Octo_Base.csv')
    file_rt1_x =  os.path.join(data_path, 'bridge_RT-1-X.csv')
    name = 'Bridge Tasks'
    save_name = 'bridge.pdf'
    fontsize = 20
    plot_simpler_results(file_openvla, file_medit_low, file_medit_high, file_octo_base, file_rt1_x, name, save_name, fontsize)

data_path = "/home/marcelr/rlds_dataset_builder/data"
# plot_learning_rate()
plot_coke_can_and_near()
plot_drawer()
plot_bridge()
print('Done')