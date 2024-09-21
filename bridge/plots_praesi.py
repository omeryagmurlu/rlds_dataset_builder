import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Patch

def plot_avg_success_rate(file_oxe, save_name, fontsize):
    # Load the CSV file
    data = pd.read_csv(file_oxe)

    # Set the Name column as the index
    data.set_index('Name', inplace=True)
    # Transpose the data to swap "Name" and "Task"
    data_transposed = data.transpose()

    # Define colors for each bar
    colors = {
        'MDT-original': 'black',
        'MDT-baseline': 'green',
        'MeDIt-SigLIP-frozen': 'blue',
        'MeDIt-SigLIP-finetune': 'red'
    }
    # Apply colors to the bars
    color_list = [colors.get(name, 'gray') for name in data.index]

    # Create the bar plot for # Datasets per Robot Embodiment
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data with custom colors
    data_transposed.plot(kind='bar', ax=ax, color=color_list)

    ax.set_yscale('linear')
    ax.tick_params(axis='y', labelsize=fontsize - 4)
    ax.tick_params(axis='x', labelsize=fontsize - 4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set the labels for the axes and the title
    ax.set_ylabel('Avg. Success Rate %', fontsize=fontsize)
    ax.set_title('Average Success Rates Over All Tasks', fontsize=fontsize)

    # Rotate the x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/real_kitchen_results_' + save_name)

def plot_avg_horizon_length(file_oxe, save_name, fontsize):
    # Load the CSV file
    data = pd.read_csv(file_oxe)

    # Set the Name column as the index
    data.set_index('Name', inplace=True)
    # Transpose the data to swap "Name" and "Task"
    data_transposed = data.transpose()

    # Define colors for each bar
    colors = {
        'MDT-original': 'black',
        'MDT-baseline': 'green',
        'MeDIt-SigLIP-frozen': 'blue',
        'MeDIt-SigLIP-finetune': 'red'
    }
    # Apply colors to the bars
    color_list = [colors.get(name, 'gray') for name in data.index]

    # Create the bar plot for # Datasets per Robot Embodiment
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data with custom colors
    data_transposed.plot(kind='bar', ax=ax, color=color_list)

    ax.set_yscale('linear')
    ax.tick_params(axis='y', labelsize=fontsize - 4)
    ax.tick_params(axis='x', labelsize=fontsize - 4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set the labels for the axes and the title
    ax.set_ylabel('Avg. Horizon Length', fontsize=fontsize)
    ax.set_title('Average Horizon Length Per Tasks', fontsize=fontsize)

    # Rotate the x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.set_ylim(5)
    
    # Move the legend to the upper left, outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(0.8, 1.12), title='Model', fontsize=fontsize - 6)
    # ax.legend(loc='upper right', title='Model', fontsize=fontsize - 8.5)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/real_kitchen_results_' + save_name)

def plot_simpler_results(file_openvla, file_medit_low, file_medit_high, file_octo_base, file_rt1_x, name, save_name, fontsize):
    
    df_openvla = pd.read_csv(file_openvla)
    df_medit_low = pd.read_csv(file_medit_low)
    df_medit_high = pd.read_csv(file_medit_high)
    df_octo_base = pd.read_csv(file_octo_base)
    df_rt1_x = pd.read_csv(file_rt1_x)

    # Prepare data for plotting, converting values to percentages
    layouts = df_openvla['Layout']
    values_openvla = df_openvla['Value'] * 100  # Convert to percentages
    values_medit_low = df_medit_low['Value'] * 100  # Convert to percentages
    values_medit_high = df_medit_high['Value'] * 100  # Convert to percentages
    values_octo_base = df_octo_base['Value'] * 100  # Convert to percentages
    values_rt1_x = df_rt1_x['Value'] * 100  # Convert to percentages

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

    # Add labels and title
    ax.set_xlabel('Task Description', fontsize=fontsize)
    ax.set_ylabel('Success Rate (%)', fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize - 4)
    ax.tick_params(axis='x', labelsize=fontsize - 4)
    ax.set_xticks(x)
    ax.set_xticklabels(layouts, rotation=45, ha="right")

    # Add a legend for colors
    legend1 = ax.legend(title='Model', fontsize=fontsize - 4)

    # ax.add_artist(legend1)

    # Add grid for better readability
    # ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for each bar
    for bars in [bars_medit_low, bars_medit_high, bars_octo_base, bars_openvla, bars_rt1_x]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.0f}%', ha='center', va='bottom', fontsize=12)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/' + save_name)


def plot_data():
    # Load the CSV files into pandas DataFrames
    file_oxe =  os.path.join(data_path, 'results_avg.csv')
    file_oxe_trimmed =  os.path.join(data_path, 'results_single_task.csv')
    moved_cam =  os.path.join(data_path, 'results_moved_camera.csv')
    save_name = 'avg_success.pdf'
    save_name_trimmed = 'avg_horizon_length.pdf'
    save_name_moved_cam = 'avg_horizon_length_moved_cam.pdf'
    fontsize = 20

    plot_avg_success_rate(file_oxe, save_name, fontsize)
    plot_avg_horizon_length(file_oxe_trimmed, save_name_trimmed, fontsize)
    plot_avg_horizon_length(moved_cam, save_name_moved_cam, fontsize)

def plot_bridge():
    # Load the CSV files into pandas DataFrames
    file_openvla =  os.path.join(data_path, 'praesi/bridge_openvla.csv')
    file_medit_low =  os.path.join(data_path, 'praesi/bridge_medit_low.csv')
    file_medit_high =  os.path.join(data_path, 'praesi/bridge_medit_high.csv')
    file_octo_base =  os.path.join(data_path, 'praesi/bridge_Octo_Base.csv')
    file_rt1_x =  os.path.join(data_path, 'praesi/bridge_RT-1-X.csv')
    name = 'Bridge Tasks'
    save_name = 'praesi/bridge.pdf'
    fontsize = 20
    plot_simpler_results(file_openvla, file_medit_low, file_medit_high, file_octo_base, file_rt1_x, name, save_name, fontsize)

def plot_fractal():
    # Load the CSV files into pandas DataFrames
    file_openvla =  os.path.join(data_path, 'praesi/fractal_openvla.csv')
    file_medit_low =  os.path.join(data_path, 'praesi/fractal_medit_low.csv')
    file_medit_high =  os.path.join(data_path, 'praesi/fractal_medit_high.csv')
    file_octo_base =  os.path.join(data_path, 'praesi/fractal_Octo_Base.csv')
    file_rt1_x =  os.path.join(data_path, 'praesi/fractal_RT-1-X.csv')
    name = 'Fractal Tasks'
    save_name = 'praesi/fractal.pdf'
    fontsize = 20
    plot_simpler_results(file_openvla, file_medit_low, file_medit_high, file_octo_base, file_rt1_x, name, save_name, fontsize)

data_path = "/home/marcelr/rlds_dataset_builder/data"

plot_data()
# plot_bridge()
# plot_fractal()
print('Done')