#%%
# Original code for connectivity plot from https://mne.tools/mne-connectivity/dev/auto_examples/mne_inverse_label_connectivity.html#sphx-glr-auto-examples-mne-inverse-label-connectivity-py
# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Nicolas P. Rougier (graph code borrowed from his matplotlib gallery)
#
# License: BSD (3-clause)
# Modified by: Riccardo Leone <riccardoleone1991@gmail.com>


import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.viz import circular_layout

from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle

import pandas as pd

dict_labels_without_first_lobe_order = {"occ": ['cuneus', 'lateraloccipital', 'pericalcarine','lingual'],
                                   "temp": ["temporalpole", "parahippocampal", 'bankssts', 'transversetemporal', 'superiortemporal'],
                                    "ins": ['insula'],
                                    "par" : ["postcentral", 'superiorparietal', 'inferiorparietal', 'supramarginal', 'paracentral', "precuneus", "posteriorcingulate", "isthmuscingulate"],
                                    "front": ["precentral", "superiorfrontal", 'caudalmiddlefrontal', 'rostralmiddlefrontal', 'lateralorbitofrontal', 'medialorbitofrontal', 'parsopercularis', 'parstriangularis', 'parsorbitalis', 'frontalpole', "caudalanteriorcingulate", "rostralanteriorcingulate"],}


labels = [lab for lab in dict_labels_without_first_lobe_order.values()]
labels_without_first_lobe_order = [label for sublist in labels for label in sublist]

first_labels = ["entorhinal", "fusiform", "inferiortemporal", "middletemporal"]

df = pd.read_csv("average_disconnectivity_matrix.csv", index_col=0)
new_index = df.index.str.lower().copy()
new_index = [idx.split("_")[-1] + "-" + idx.split("_")[-2] for idx in new_index]
df.index = new_index
df.columns = new_index

data_path = sample.data_path()
subjects_dir = data_path / "subjects"
# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot("sample", parc="aparc", subjects_dir=subjects_dir)

# First, we reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]
lh_labels = [name for name in label_names if name.endswith("lh")]
# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]
first_labels_lh = [name + "-lh" for name in first_labels]
lh_labels_without_first = [lab for lab in lh_labels if lab not in first_labels_lh]
lh_labels_without_first = [region + '-lh' for region in labels_without_first_lobe_order]
lh_labels = lh_labels_without_first + first_labels_lh

# For the right hemi
rh_labels = [label[:-2] + "rh" for label in lh_labels]
first_labels_rh = [name + "-rh" for name in first_labels]
rh_labels_without_first = [region + '-rh' for region in labels_without_first_lobe_order]
rh_labels = rh_labels_without_first + first_labels_rh

red_shades = [
    (1.0, 0.5, 0.5, 1.0), 
    (1.0, 0.4, 0.4, 1.0), 
    (0.9, 0.3, 0.3, 1.0), 
    (0.8, 0.1, 0.1, 1.0),     
]

label_colors = [(0,0,0,0) for i in range(68)]
# Combine LH and RH labels to apply same colors to both sides
first_labels_both = first_labels_lh + first_labels_rh
not_first_label_both = lh_labels_without_first + rh_labels_without_first

# Apply green shades to both hemispheres
for i, label in enumerate(first_labels_both):
    if label in label_names:
        idx = label_names.index(label)
        # Repeat the green shades (wrap around if needed)
        shade = red_shades[i % len(red_shades)]
        label_colors[idx] = shade


# Save the plot order and create a circular layout
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_angles = circular_layout(
    label_names, node_order, start_pos=90, group_boundaries=[0, 4, 16, 24, 25, 30, 
                                                             len(label_names) / 2,  
                                                             len(label_names)-30,
                                                             len(label_names)-25,
                                                             len(label_names)-24,
                                                             len(label_names)-16,
                                                             len(label_names)-4])

df_reordered = df.loc[label_names, label_names]

# Assign muted gray-blue/greenish shades to non-first labels
for i, label in enumerate(label_names):
    if label not in first_labels_both:
        label_both = label[:-3]

        if label_both in dict_labels_without_first_lobe_order["front"]:
            muted_color = (0.8, 0.5, 0.2, 1.0)
        elif label_both in dict_labels_without_first_lobe_order["par"]:
            muted_color = (0.6, 0.5, 0.2, 1.0)
        elif label_both in dict_labels_without_first_lobe_order["ins"]:
            muted_color = (0.4, 0.5, 0.1, 1.0)
        elif label_both in dict_labels_without_first_lobe_order["temp"]:            
            muted_color = (0.2, 0.5, 0.3, 1.0)
        elif label_both in dict_labels_without_first_lobe_order["occ"]:            
            muted_color = (0., 0.5, 0.5, 1.0)
        
        label_colors[i] = muted_color

df_filtered_connections = df_reordered.copy()
not_first_label_both = lh_labels_without_first + rh_labels_without_first
df_filtered_connections.loc[not_first_label_both, not_first_label_both] = 0
import seaborn as sns

sns.set_context("paper", font_scale=0.8)
# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
fig, ax = plt.subplots(figsize=(7, 7), facecolor="white", subplot_kw=dict(polar=True))
plot_connectivity_circle(
    df_filtered_connections.values,
    label_names,
    facecolor="white",
    textcolor="black",
    n_lines=100,
    vmin=0,
    vmax=60,
    node_edgecolor="white",
    colormap="Blues",
    node_angles=node_angles,
    node_colors=label_colors,
    ax=ax,
    colorbar_pos = (.8, .5)
)
fig.tight_layout()
fig.show()
fig.savefig("/mnt/c/Users/leo_r/Desktop/cat_whim_connectivity_circle.svg", dpi=300, facecolor="white")
fig.savefig("/mnt/c/Users/leo_r/Desktop/cat_whim_connectivity_circle.png", dpi=300, facecolor="white")

#%%
def get_most_disconnected_region_with_target(target):
    idx = np.argmax(df_filtered_connections[target])
    val = df_filtered_connections[target].max()
    print("-------------------------------------------")
    print("Most disconnected region with", target, ":")
    print(df_filtered_connections.index[idx], f"{val:.2f}")

get_most_disconnected_region_with_target("entorhinal-lh")
get_most_disconnected_region_with_target("fusiform-lh")
get_most_disconnected_region_with_target("middletemporal-lh")
get_most_disconnected_region_with_target("inferiortemporal-lh")

get_most_disconnected_region_with_target("entorhinal-rh")
get_most_disconnected_region_with_target("fusiform-rh")
get_most_disconnected_region_with_target("middletemporal-rh")
get_most_disconnected_region_with_target("inferiortemporal-rh")

# Collect all first_labels with hemisphere suffixes
first_labels_full = [label + hemi for label in first_labels for hemi in ["-lh", "-rh"]]

# Accumulate disconnection vectors from each seed region
disconnection_vectors = []
for seed in first_labels_full:
    if seed in df_filtered_connections.index:
        disconnection_vectors.append(df_filtered_connections.loc[seed])

# Compute the average disconnection profile across all first_labels
average_disconnection = pd.concat(disconnection_vectors, axis=1).mean(axis=1)

# Sum average disconnection strength per lobe
lobe_disconnectivity_avg = {lobe: 0.0 for lobe in dict_labels_without_first_lobe_order}

for lobe, regions in dict_labels_without_first_lobe_order.items():
    for region in regions:
        for hemi in ['-lh', '-rh']:
            full_region_name = region + hemi
            if full_region_name in average_disconnection.index:
                lobe_disconnectivity_avg[lobe] += average_disconnection[full_region_name]

# Compute total disconnection across all lobes
total_disconnection = sum(lobe_disconnectivity_avg.values())

# Compute percentage per lobe
lobe_percentages = {lobe: (val / total_disconnection) * 100 for lobe, val in lobe_disconnectivity_avg.items()}

df = pd.DataFrame(lobe_percentages, index=[0]).T.reset_index()
df.columns = ["lobe", "value"]
df_2 = pd.concat([df, df], axis=0).reset_index(drop=True)
df_2.loc[:4, "lobe"] = "R_" + df_2.loc[:4, "lobe"]
df_2.loc[5:, "lobe"] = "L_" + df_2.loc[5:, "lobe"]


df_2.to_csv("/mnt/c/Users/leo_r/Desktop/cat_whim_df_lobe_percentages.csv")






