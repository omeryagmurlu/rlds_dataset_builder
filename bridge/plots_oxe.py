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

def plot_first_version(file_oxe, save_name, fontsize):
    # Load the CSV file
    data = pd.read_csv(file_oxe)

    # Preprocess the data
    # Remove commas and convert the 'Trajectories' column to integers
    data['Trajectories'] = data['Trajectories'].str.replace(',', '').astype(int)

    # Aggregate data by robot type
    datasets_per_robot = data['Robot'].value_counts().sort_values(ascending=False)
    scenes_per_robot = data.groupby('Robot')['Scenes'].sum().sort_values(ascending=False)
    trajectories_per_robot = data.groupby('Robot')['Trajectories'].sum().sort_values(ascending=False)

    # Create the bar plot for # Datasets per Robot Embodiment
    plt.figure(figsize=(10, 6))
    datasets_per_robot.plot(kind='bar', color='navy')
    plt.yscale('log')
    plt.xlabel('Robot Embodiment')
    plt.ylabel('# Datasets')
    plt.title('(a) # Datasets per Robot Embodiment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/1_bar_plot' + save_name)

    # Create the pie chart for # Scenes per Embodiment
    plt.figure(figsize=(8, 8))
    plt.pie(scenes_per_robot, labels=scenes_per_robot.index, autopct='%1.1f%%', startangle=140)
    plt.title('(b) # Scenes per Embodiment')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/1_scenes_robot' + save_name)

    # Create the pie chart for # Trajectories per Embodiment
    plt.figure(figsize=(8, 8))
    plt.pie(trajectories_per_robot, labels=trajectories_per_robot.index, autopct='%1.1f%%', startangle=140)
    plt.title('(c) # Trajectories per Embodiment')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/1_trajectories_robot' + save_name)

def plot_second_version(file_oxe, save_name, fontsize, trimmed):
    # Load the CSV file
    data = pd.read_csv(file_oxe)

    # Preprocess the data
    # Remove commas and convert the 'Trajectories' column to integers
    data['Trajectories'] = data['Trajectories'].str.replace(',', '').astype(int)

    # Aggregate data by robot type
    datasets_per_robot = data['Robot'].value_counts().sort_values(ascending=False)
    scenes_per_robot = data.groupby('Robot')['Scenes'].sum().sort_values(ascending=False)
    trajectories_per_robot = data.groupby('Robot')['Trajectories'].sum().sort_values(ascending=False)

    datasets_counts_norm = datasets_per_robot / datasets_per_robot.max()
    datasets_colors = plt.cm.Blues(0.5 + 0.5 * datasets_counts_norm)

    # Create the bar plot for # Datasets per Robot Embodiment
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 4]})

    ax.bar(datasets_per_robot.index, datasets_per_robot.values, color=datasets_colors)
    ax2.bar(datasets_per_robot.index, datasets_per_robot.values, color=datasets_colors)

    ax.set_ylim(7, 30)
    ax2.set_ylim(0, 6)

    draw_wavy_break(ax, position=7, wavy_width=0.03, wavy_height=3.5)
    draw_wavy_break(ax2, position=6, wavy_width=0.03, wavy_height=0.25)

    ax.tick_params(labelbottom=False)
    ax.tick_params(bottom=False)
    ax.set_yscale('linear')
    ax2.set_yscale('linear')
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ax2.set_xlabel('Robot Embodiment', fontsize=fontsize)
    # ax2.set_ylabel('# Datasets', fontsize=fontsize)
    if trimmed:
        ax.set_title('(a) # Datasets per Robot Embodiment Trimmed', fontsize=fontsize)
    else:
        ax.set_title('(a) # Datasets per Robot Embodiment', fontsize=fontsize)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/2_bar_plot' + save_name)

    # Create the pie chart for # Scenes per Embodiment with legend
    plt.figure(figsize=(10, 10))
    wedges, texts, autotexts = plt.pie(
        # scenes_per_robot, autopct='%1.1f%%', startangle=140, wedgeprops=dict(edgecolor='k')
        scenes_per_robot, autopct=naming_scenes, startangle=140, wedgeprops=dict(edgecolor='k')
    )
    if trimmed:
        plt.title('(b) # Scenes per Embodiment Trimmed', fontsize=fontsize)
    else:
        plt.title('(b) # Scenes per Embodiment', fontsize=fontsize)
    plt.legend(wedges, scenes_per_robot.index, title="Robots", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/2_scenes_robot' + save_name)

    # Create the pie chart for # Trajectories per Embodiment with legend
    plt.figure(figsize=(10, 10))
    wedges, texts, autotexts = plt.pie(
        # trajectories_per_robot, autopct='%1.1f%%', startangle=140, wedgeprops=dict(edgecolor='k')
        trajectories_per_robot, autopct=naming_traj, startangle=140, wedgeprops=dict(edgecolor='k')
    )
    if trimmed:
        plt.title('(c) # Scenes per Embodiment Trimmed', fontsize=fontsize)
    else:
        plt.title('(c) # Trajectories per Embodiment', fontsize=fontsize)
    plt.legend(wedges, trajectories_per_robot.index, title="Robots", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/marcelr/rlds_dataset_builder/data/2_trajectories_robot' + save_name)

def naming_traj(x):
    if x > 1.0:
        return ('%1.1f%%' % x)
    else:
        return ''
    
def naming_scenes(x):
    if x > 0.8:
        return ('%1.1f%%' % x)
    else:
        return ''

def plot_data():
    # Load the CSV files into pandas DataFrames
    file_oxe =  os.path.join(data_path, 'open_x_new_values.csv')
    file_oxe_trimmed =  os.path.join(data_path, 'open_x_new_values_trimmed.csv')
    save_name = 'oxe_plots.pdf'
    save_name_trimmed = 'oxe_plots_trimmed.pdf'
    fontsize = 20
    # plot_first_version(file_openvla, save_name, fontsize)
    plot_second_version(file_oxe, save_name, fontsize, False)
    plot_second_version(file_oxe_trimmed, save_name_trimmed, fontsize, True)


data_path = "/home/marcelr/rlds_dataset_builder/data"
# plot_learning_rate()
plot_data()
print('Done')