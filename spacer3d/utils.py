import os
import numpy as np
import cupy as cp
import seaborn as sns
import pandas as pd
from glob import glob
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from collections import defaultdict

def calculate_pi(data, p=0.95):
    """
    Calculate the 95% prediction interval for the given data.

    Parameters:
    data (array-like): A list or array of numerical data points.

    Returns:
    dict: Lower and upper bounds of the 95% prediction interval.
    """
    # Calculate mean and standard error of the mean
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # Number of observations
    n = len(data)

    bounds = {}
    for conf in [0.95, 0.99, 0.999]:
        t_score = stats.t.ppf((1 + conf) / 2., n-1)
        margin_of_error = t_score * std * np.sqrt(1 + 1/n)
        bounds[str(conf)] = {
            "lower": mean - margin_of_error,
            "upper": mean + margin_of_error,
        }

    return bounds

def calculate_percentile_range(data):
    """
    Calculate the 95%, 99%, 99.9%, and 99.99% percentile intervals for the given data.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'Radius (r)', 'K(r) value', and possibly 
                         'simulation number' columns.

    Returns:
    dict: Lower and upper bounds of the percentile intervals for each radius.
    """
    radii = data[data["Line"] == 1]["Radius (r)"].tolist()
    
    bounds = {}
    for radius in radii:
        subset = data[data["Radius (r)"] == radius]["K(r)"].tolist()

        # Define the confidence levels and their corresponding percentile values
        percentiles = {
            0.95: (2.5, 97.5),
            0.99: (0.5, 99.5),
            0.999: (0.05, 99.95),
            0.9999: (0.005, 99.995)
        }

        bounds[radius] = {}
        for conf, (lower_perc, upper_perc) in percentiles.items():
            lower_bound = np.percentile(subset, lower_perc)
            upper_bound = np.percentile(subset, upper_perc)
            bounds[radius][str(conf)] = {
                "lower": lower_bound,
                "upper": upper_bound,
            }
    
    return bounds

def get_equivalent_color(cf, alpha, cb=(255, 255, 255)):
    """
    Calculate the RGB values of an opaque color in decimal format (0-1) that visually matches a semi-transparent color
    when placed over a specified background color. This function automatically handles both regular RGB (0-255) and
    decimal RGB (0-1) formats for the input colors.

    Parameters:
    - cf (tuple): The foreground color in RGB format, either in regular (0-255) or decimal (0-1).
    - alpha (float): The alpha transparency level of the foreground color, ranging from 0 (completely transparent)
      to 1 (completely opaque).
    - cb (tuple, optional): The background color in RGB format, either in regular (0-255) or decimal (0-1). Defaults to white (255, 255, 255).

    Returns:
    - list of float: The RGB values of the equivalent opaque color in decimal format that visually matches the semi-transparent
      foreground color when viewed against the background color.

    Example:
    >>> get_equivalent_color((0, 0, 255), 0.5)
    [0.5, 0.5, 1.0]
    """
    # Check if the input colors are in the regular RGB (0-255) format and convert them to decimal format if necessary
    cf_decimal = [f / 255. if max(cf) > 1 else f for f in cf]
    cb_decimal = [b / 255. if max(cb) > 1 else b for b in cb]

    # Calculate the equivalent color in decimal RGB format
    result_decimal = [alpha * f + (1 - alpha) * b for f, b in zip(cf_decimal, cb_decimal)]

    return result_decimal
    
def pi_range_to_df(data: dict):
    """
    Converts a nested dictionary containing radius, confidence intervals ('pi'), and their corresponding
    lower and upper bounds into a pandas DataFrame.

    Parameters:
    - data (dict): A nested dictionary where the first level keys are radii, the second level keys are
      confidence intervals (e.g., '0.95'), and the values are dictionaries with 'lower' and 'upper' bounds.

    Returns:
    - pandas.DataFrame: A DataFrame with columns for 'radius', 'pi' (confidence interval), 'lower', and 'upper' bounds.

    Example:
    >>> data = {1: {'0.95': {'lower': 0, 'upper': 10}, '0.99': {'lower': -5, 'upper': 15}}}
    >>> pi_range_to_df(data)
    # Returns a DataFrame with each row representing a combination of radius, pi, lower, and upper values.
    """
    # Flatten the dictionary into a list of rows
    rows = []
    for radius, pi in data.items():
        for pi, bounds in pi.items():
            rows.append({
                "radius": radius,
                "pi": pi,
                "lower": bounds["lower"],
                "upper": bounds["upper"],
            })

    # Convert the list of rows into a DataFrame
    return pd.DataFrame(rows)

def plot_interval(df, color, ax=None, pi="0.95"):
    """
    Plots confidence intervals (lower and upper bounds) for a specified 'pi' value from a DataFrame
    onto a given axis ('ax'), including filling the area between these bounds.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to plot, expected to have columns
      'radius', 'pi', 'lower', and 'upper'.
    - color (str): The color to use for both lines and the fill between them.
    - ax (matplotlib.axes.Axes): The Matplotlib axis object where the plot will be drawn.
    - pi (str, optional): The confidence interval level to plot (e.g., '0.95'). Defaults to '0.95'.

    Returns:
    - None: This function plots on the provided Matplotlib axis and does not return a value.

    Example:
    >>> plot_interval(df, 'blue', ax, pi="0.99")
    # This will plot the lower and upper bounds for the '0.99' confidence interval on the provided axis.
    """
    for col_name in ["lower", "upper"]:
        sns.lineplot(
            data=df[df["pi"] == pi],
            x="radius",
            y=col_name,
            color=get_equivalent_color(color, 0.4),
            linewidth=0.5,
            ax=ax,
        )
    
    # Now, fill between the lines
    # Extracting the 'radius', 'lower', and 'upper' values for the "0.95" confidence interval
    radii = df[df["pi"] == pi]["radius"]
    lower = df[df["pi"] == pi]["lower"]
    upper = df[df["pi"] == pi]["upper"]

    # Filling between the 'lower' and 'upper' lines on the second subplot
    if ax is None:
        plt.fill_between(radii, lower, upper, color=get_equivalent_color(color, 0.2))  # Adjust the color and alpha as needed
    else:
        ax.fill_between(radii, lower, upper, color=get_equivalent_color(color, 0.2))  # Adjust the color and alpha as needed
    label = float(pi) * 100
    label = int(label) if label.is_integer() else label
    return mpatches.Patch(color=get_equivalent_color(color, 0.2), label=f"{label}% Interval")

def plot_background_intervals(df, color, ax=None):
    max_pi = str(df["pi"].astype(float).max())

    # Extracting the 'radius', 'lower', and 'upper' values for the specified confidence interval
    radii = df[df["pi"] == max_pi]["radius"]
    lower = df[df["pi"] == max_pi]["lower"]
    upper = df[df["pi"] == max_pi]["upper"]

    # Get current y-axis limits
    if ax is not None:
        ymin, ymax = ax.get_ylim()

        # Filling from the bottom of the graph to the bottom-most line
        ax.fill_between(radii, ymin, lower, color=get_equivalent_color(color, 0.2))

        # Filling from the top of the graph to the topmost line
        ax.fill_between(radii, upper, ymax, color=get_equivalent_color(color, 0.2))
    else:
        ymin, ymax = plt.ylim()

        # Filling from the bottom of the graph to the bottom-most line
        plt.fill_between(radii, ymin, lower, color=get_equivalent_color(color, 0.2))

        # Filling from the top of the graph to the topmost line
        plt.fill_between(radii, upper, ymax, color=get_equivalent_color(color, 0.2))

    return mpatches.Patch(color=get_equivalent_color(color, 0.2), label=f">99.99% Interval")

def normalize(rstats, rand_rstats):
    def min_max(data, min_val, max_val):
        """
        Normalize the data to have a lower bound of -1 and upper bound of 1
        with respect to the 95% interval min and max values
    
        Parameters:
        data (array-like): A list or array of numerical data points.
        min_val (float): min value
        max_val (float): max value
    
        Returns:
        array: Normalized data with bounds [-1, 1].
        """
        normalized_data = 2 * ((data - min_val) / (max_val - min_val)) - 1
        return normalized_data

    normalized_K = []
    for radius in rstats["Radius (r)"]:
        subset = rand_rstats[rand_rstats["Radius (r)"] == radius]["K(r)"].tolist()
        bounds = calculate_pi(subset)
        lower_bound_95, upper_bound_95 = bounds["0.95"]["lower"], bounds["0.95"]["upper"]
        if abs(lower_bound_95 - upper_bound_95) < 1:
            val = 0.
        else:
            val = min_max(rstats[rstats["Radius (r)"] == radius]["K(r)"].tolist()[0], lower_bound_95, upper_bound_95)
        normalized_K.append(val)
    return normalized_K

def normalize_w_intervals(rstats, rand_rstats):
    def min_max(data, min_val, max_val):
        """
        Normalize the data to have a lower bound of -1 and upper bound of 1
        with respect to the 95% interval min and max values.

        Parameters:
        data (float): A numerical data point.
        min_val (float): min value of the 95% interval.
        max_val (float): max value of the 95% interval.

        Returns:
        float: Normalized data with bounds [-1, 1].
        """
        # Avoid division by zero
        if max_val == min_val:
            return 0
        return 2 * ((data - min_val) / (max_val - min_val)) - 1

    # Lists to store data before concatenation
    normalized_k_rows = []
    intervals_rows = []

    # Calculate percentile ranges for random simulations
    percentile_ranges = calculate_percentile_range(rand_rstats)

    for radius in rstats["Radius (r)"].unique():
        bounds_95 = percentile_ranges[radius]["0.95"]
        
        # Normalize the K function value for this radius
        K_value = rstats[rstats["Radius (r)"] == radius]["K(r)"].iloc[0]
        normalized_K = min_max(K_value, bounds_95["lower"], bounds_95["upper"])

        # Normalize the theoretical K function for this radius
        theoretical_K = rstats[rstats["Radius (r)"] == radius]["theoretical_K"].iloc[0]
        normalized_theoretical_K = min_max(theoretical_K, bounds_95["lower"], bounds_95["upper"])

        # Append to the list for normalized K values
        normalized_k_rows.append({"radius": radius, "normalized_K": normalized_K, "theoretical_K": normalized_theoretical_K})

        # Iterate through each confidence level to populate intervals_df
        for conf in ["0.95", "0.99", "0.999", "0.9999"]:
            bounds = percentile_ranges[radius][conf]
            lower = min_max(bounds["lower"], bounds_95["lower"], bounds_95["upper"])
            upper = min_max(bounds["upper"], bounds_95["lower"], bounds_95["upper"])
            
            # Append to the list for interval bounds
            intervals_rows.append({
                "radius": radius,
                "pi": conf,
                "lower": lower,
                "upper": upper
            })
    
    # Concatenate lists into DataFrames
    normalized_k_df = pd.concat([pd.DataFrame([row]) for row in normalized_k_rows], ignore_index=True)
    intervals_df = pd.concat([pd.DataFrame([row]) for row in intervals_rows], ignore_index=True)
    
    return normalized_k_df, intervals_df

def get_interval_pairs(df):
    interval_dfs = {}
    
    # Confidence levels to create separate DataFrames for
    conf_levels = ["95", "99", "999", "9999"]
    
    for conf in conf_levels:
        # Selecting columns for the current confidence level
        cols = ["radius", f"lower_{conf}", f"upper_{conf}"]
        interval_dfs[conf] = df[cols].copy()
        interval_dfs[conf].columns = ["radius", "lower", "upper"]

    return interval_dfs

def _plot_all_intervals(pi_df, palette, ax=None):
    patches = []
    for i, pi in enumerate(["0.9999", "0.999", "0.99", "0.95"]):
        patch = plot_interval(pi_df, color=palette[i+1], ax=ax, pi=pi)
        patches.append(patch)
    patch = plot_background_intervals(pi_df, color=palette[0], ax=ax)
    patches.insert(0, patch)
    return patches

def _plot_normalized_graph(rstats, rand_rstats, palette, ax=None):
    normalized_k_df, intervals_df = normalize_w_intervals(rstats, rand_rstats)
    sns.lineplot(data=normalized_k_df, x="radius", y="theoretical_K", ax=ax, label=r"Theoretical $\mathit{K}$ Function", color="#888", linewidth=2, linestyle="dotted", zorder=98)
    sns.lineplot(data=normalized_k_df, x="radius", y="normalized_K", ax=ax, alpha=1, zorder=99, label=r"Observed $\mathit{K}$ Function")
    patches = _plot_all_intervals(intervals_df, palette=palette, ax=ax)
    return patches
    
    # paired_intervals = get_interval_pairs(normalized_df)

def _config_legend(patches, fontsize=None, ax=None):
    patch_labels = [p.get_label() for p in patches]

    if ax is not None:
        # Retrieve existing handles and labels from the axis
        line_handles, line_labels = ax.get_legend_handles_labels()
        ax.legend(handles=line_handles+patches, labels=line_labels+patch_labels, loc="upper left", fontsize=fontsize)
    else:
        # Retrieve existing handles and labels from the current axes
        line_handles, line_labels = plt.gca().get_legend_handles_labels()
        # Configure the legend for the current axes
        plt.legend(handles=line_handles+patches, labels=line_labels+patch_labels, loc="upper left", fontsize=fontsize)


def plot_ripley(df, rand_df=None, mode="3D", norm=False, save=False, output_folder="./ripley_plots", output_filename="ripley_figure.svg"):
    if norm and rand_df is None:
        raise ValueError("rand_df parameter required to plot normalized Ripley's K Function")
    
    plt.figure(figsize=(5,5))
    plt.xlabel("Radius")
    palette = sns.color_palette("rocket_r")

    if mode == "2D":
        df['theoretical_K'] = (np.pi * df['Radius (r)']**2) + df["K(r)"].min()
    else:
        df['theoretical_K'] = ((4/3) * np.pi * df['Radius (r)']**3) + df["K(r)"].min()

    if norm:
        plt.ylabel(r"$\mathit{K}_{\mathrm{Norm}}(r)$")
        patches = _plot_normalized_graph(df, rand_rstats=rand_df, palette=palette)
        _config_legend(patches)
    else:
        plt.ylabel(r"$\mathit{K}$(r)")
        sns.lineplot(data=df, x="Radius (r)", y="K(r)", alpha=1, label=r"Observed $\mathit{K}$ Function", zorder=99)
        sns.lineplot(data=df, x="Radius (r)", y="theoretical_K", label=r"Theoretical $\mathit{K}$ Function", color="#888", linewidth=2, linestyle="dotted", zorder=98)

        if rand_df is not None:
            pi_df = pi_range_to_df(calculate_percentile_range(rand_df))
            patches = _plot_all_intervals(pi_df, palette=palette)
            _config_legend(patches)

    if save:
        create_directory(output_folder)
        plt.savefig(os.path.join(output_folder, output_filename))


def plot_process(rstats_path, save=False, output_folder="./ripley_plots"):
    # palette = sns.color_palette("colorblind")
    palette = sns.color_palette("rocket_r")
    # palette = sns.color_palette("husl", 4)

    def get_rstats_files(path):
        return glob(f"{path}/*.csv")

    def _plot(rstats, rand_rstats, ax):
        a = sns.lineplot(data=rand_rstats, x="Radius (r)", y="K(r)", hue="Line", ax=ax[0], alpha=1)
        a.get_legend().remove()
        ax[0].set(xlabel=None, ylabel=r"$\mathit{K}$(r)")

        pi_df = pi_range_to_df(calculate_percentile_range(rand_rstats))
        patches = _plot_all_intervals(pi_df, palette=palette, ax=ax[1])
        _config_legend(patches, fontsize=12, ax=ax[1])
        ax[1].set(xlabel=None, ylabel=None)

        # Calculate theoretical K values and add to rstats
        rstats['theoretical_K'] = ((4/3) * np.pi * rstats['Radius (r)']**3) + rstats["K(r)"].min()

        sns.lineplot(data=rstats, x="Radius (r)", y="theoretical_K", ax=ax[2], label=r"Theoretical $\mathit{K}$ Function", color="#888", linewidth=2, linestyle="dotted", zorder=98)
        sns.lineplot(data=rstats, x="Radius (r)", y="K(r)", ax=ax[2], alpha=1, zorder=99, label=r"Observed $\mathit{K}$ Function")
        patches = _plot_all_intervals(pi_df, palette=palette, ax=ax[2])

        _config_legend(patches, fontsize=12, ax=ax[2])        
        ax[2].set(xlabel=None, ylabel=None)

        # draw_normalized_graph(rstats, rand_rstats, ax=ax[3])
        patches = _plot_normalized_graph(rstats, rand_rstats, palette=palette, ax=ax[3])
        _config_legend(patches, fontsize=12, ax=ax[3])
        ax[3].set(xlabel=None, ylabel=r"$\mathit{K}$$_{Norm}$")
        
    rstats_files = get_rstats_files(rstats_path)
    print("Loaded:", rstats_files)
    
    # filter our monte carlo results
    rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
    u_rstats_files = [f for f in rstats_files if "univariate" in f]
    m_rstats_files = [f for f in rstats_files if "multivariate" in f]

    # Update the font size for the plot
    plt.rcParams.update({"font.size": 16})
    
    # Create a subplot with 3 rows and 1 column
    f, axes = plt.subplots(len(u_rstats_files), 4, sharex=True, figsize=(19,6*len(u_rstats_files)))

    f.tight_layout(pad=1.4)
    f.supxlabel("Radius (μm)", y=0.00)
    f.subplots_adjust(bottom=0.12)
    
    for i, filename in enumerate(u_rstats_files):
        fullpath = os.path.join(rstats_path, f"{filename}.csv")
        
        prefix, _, date, id, mode, label, _ = filename.split("_")
        
        # Construct the path for the random CSV file
        rand_fullpath = os.path.join(rstats_path, f"FV10__{date}_{id}_random_{mode}_{label}_rstats.csv")
    
        # Load the CSV file and random CSV file into DataFrames
        rstats = pd.read_csv(fullpath)
        rand_rstats = pd.read_csv(rand_fullpath)

        for ax in axes[i]:
            ax.tick_params(axis="both", labelsize="16")

        _plot(rstats, rand_rstats, ax=axes[i])

        if save:
            create_directory(output_folder)
            plt.savefig(os.path.join(output_folder, f"{filename}_process.svg"))
        break
        
def plot_individuals(rstats_path, save=False, output_folder="./ripley_results/"):
    palette = sns.color_palette("rocket_r")

    def get_rstats_files(path):
        return glob(f"{path}/*.csv")

    def group_items(l, n=4):
        """Groups every 'n' items in the list into sublists."""
        return [l[i:i + n] for i in range(0, len(l), n)]

    def rearrange_sublists(lst, order):
        """Rearranges elements within each sublist of a list based on a specified order."""
        return [[sublist[i] for i in order] for sublist in lst]
        
    rstats_files = get_rstats_files(rstats_path)
    print("Loaded:", rstats_files)

    # filter our monte carlo results
    rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
    u_rstats_files = [f for f in rstats_files if "univariate" in f]
    m_rstats_files = [f for f in rstats_files if "multivariate" in f]

    grouped_u_rstats_files = rearrange_sublists(group_items(sorted(u_rstats_files)), [2,1,0,3])
    # Update the font size for the plot
    plt.rcParams.update({"font.size": 20})
    
    # Create a subplot with 3 rows and 1 column
    f, axes = plt.subplots(4,4, sharex=True, figsize=(15,15))
    f.tight_layout(pad=-0.35)

    f.supxlabel("Radius (μm)", x=0.51, y=-0.001)
    f.supylabel("K$_{Norm}$", x=-0.001)
    f.subplots_adjust(bottom=0.06, left=0.06)
    
    # Loop over each univariate filename
    for i, group in enumerate(grouped_u_rstats_files):
        for j, filename in enumerate(group):
            # Construct the path for the CSV file
            fullpath = os.path.join(rstats_path, f"{filename}.csv")
            
            prefix, _, date, id, mode, label, _ = filename.split("_")
            
            # Construct the path for the random CSV file
            rand_fullpath = os.path.join(rstats_path, f"FV10__{date}_{id}_random_{mode}_{label}_rstats.csv")
    
            # Load the CSV file and random CSV file into DataFrames
            rstats = pd.read_csv(fullpath)
            rand_rstats = pd.read_csv(rand_fullpath)

            if i == 0:
                title=["Tumor", "NG2+", "Branches", "TVC"]
            else:
                title=[None]*4
    
            # Calculate theoretical K values and add to rstats
            rstats['theoretical_K'] = ((4/3) * np.pi * rstats['Radius (r)']**3) + rstats["K(r)"].min()

            # draw_normalized_graph(rstats, rand_rstats, ax=axes[i,j], title=title[j])
            patches = _plot_normalized_graph(rstats, rand_rstats, palette=palette, ax=axes[i,j])
            axes[i, j].legend().remove()
            axes[i, j].set(xlabel="")
            axes[i, j].set(ylabel="")
            axes[i, j].tick_params(axis="both", labelsize="14")

            # _config_legend(patches, ax=axes[i, j])
    if save:
        plt.savefig(os.path.join(output_folder, f"{filename}_individual.svg"))

    # Create a new figure for the legend
    # fig_legend = plt.figure(figsize=(3, 2))
    # ax_legend = fig_legend.add_subplot(111)
    # Make the subplot area transparent and remove the axes
    # ax_legend.axis('off')
    # fig_legend.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Draw the legend on the new figure
    # ax_legend.legend(*axes[0,0].get_legend_handles_labels(), loc='center')

    # Save the figure containing only the legend
    # fig_legend.savefig('/Users/danielkermany/Desktop/legend_only.svg')
    # plt.savefig("/Users/danielkermany/Desktop/S6.svg", bbox_inches="tight")

def _draw_combined_graph(df, title=None, legend=None, var_name=None, force_ylim=None,
                         palette=None, save=False, output_folder="./ripley_results/",
                         output_filename="ripley_combined_plot.svg"):
    # Set the tick label format to plain
    plt.ticklabel_format(style="plain")

    # Update the font size for the plot
    plt.rcParams.update({"font.size": 20})

    # Create a subplot with 3 rows and 1 column
    f, ax = plt.subplots(figsize=(7,7))
    
    # Plot the K(r) values for the data and the random data
    # l_ax = sns.lineplot(data=df, x="Radius (r)", y="K_norm", hue="Sample",
    #                     style="Sample", ax=ax, legend=legend, zorder=3)
    l_ax = sns.lineplot(data=df, x="Radius (r)", y="K_norm", hue=var_name,
                        style=var_name, ax=ax, legend=legend, zorder=3,
                        palette=palette)

    for line in plt.gca().get_lines():
        label = line.get_label()
        if label == "Average":
            line.set_alpha(1.0)
        else:
            line.set_alpha(0.4)

    if len(l_ax. lines) > 0:
        average_line = l_ax.lines[0]
        average_line.set_linewidth(2)
        average_line.set_alpha(1.0)
        average_line.set_zorder(5)
        average_line.set_linestyle("-")
    
    # # Get the current x-axis limits after plotting the data
    # ax = plt.gca()

    xlims = ax.get_xlim() if ax is not None else plt.xlim()
    
    # Get the current y-axis limits after plotting the data
    ylims = ax.get_ylim() if ax is not None else plt.ylim()

    # Manually set the y-axis limits to include the -1 to 1 range
    max_val = max(df["K_norm"])
    min_val = min(df["K_norm"])
    
    new_ylims = (min(-1-(0.05*max_val), min_val), max(1+(0.05*max_val), max_val))  # This includes both your data and the -1 to 1 interval
    
    # Fill between y=1 and y=-1, extending to the edges of the graph
    if ax is not None:
        ax.fill_between(xlims, -1, 1, color='#ffe5ce', alpha=0.9)

        # Adding thicker lines for the top and bottom borders, extending to the edges
        ax.hlines(1, *xlims, colors='#ffbb80', linewidth=1.5)
        ax.hlines(-1, *xlims, colors='#ffbb80', linewidth=1.5)
        
        # Set the x and y-axis limits to ensure the fill and borders extend to the edges
        ax.set_xlim(xlims)
        ax.set_ylim(new_ylims)
        print("ylims", new_ylims)

        if force_ylim:
            ax.set_ylim(force_ylim)
        
        ax.set(xlabel="")
        ax.set(ylabel="")

        ax.title.set_text(title)
    else:
        plt.fill_between(xlims, -1, 1, color='#ffe5ce', alpha=0.9)

        # Adding thicker lines for the top and bottom borders, extending to the edges
        plt.hlines(1, *xlims, colors='#ffbb80', linewidth=1.5)
        plt.hlines(-1, *xlims, colors='#ffbb80', linewidth=1.5)
        
        # Set the x and y-axis limits to ensure the fill and borders extend to the edges
        plt.xlim(xlims)
        plt.ylim(new_ylims)
        
        plt.xlabel("")
        plt.ylabel("")

        plt.title.set_text(title)
    
    print("average line", average_line.get_label())
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        hue_handles = [average_line] + handles[1:]
        hue_labels = ["Average"] + [f"{var_name} {i}" for i in range(1, len(handles))]
        ax.legend(hue_handles, hue_labels, prop={'size': 12}, loc="upper left", bbox_to_anchor=(1.0, 1.0))  # Set the font size to 10
        # plt.legend(prop={'size': 14})  # Set the font size to 10

    if save:
        create_directory(output_folder)
        plt.savefig(os.path.join(output_folder, output_filename), bbox_inches="tight")
        print(output_folder, output_filename)

def plot_combined_univariate(rstats_path):
    def get_rstats_files(path):
        return glob(f"{path}/*.csv")

    def group_items(l, n=4):
        """Groups every 'n' items in the list into sublists."""
        return [l[i:i + n] for i in range(0, len(l), n)]

    def rearrange_sublists(lst, order):
        """Rearranges elements within each sublist of a list based on a specified order."""
        return [[sublist[i] for i in order] for sublist in lst]
        
    rstats_files = get_rstats_files(rstats_path)
    print("Loaded:", rstats_files)

    # filter our monte carlo results
    rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
    u_rstats_files = sorted([f for f in rstats_files if "univariate" in f])
    m_rstats_files = sorted([f for f in rstats_files if "multivariate" in f])

    types = ["tumor", "ng2", "branch", "tvc"]
    for t in types:
        t_files = [i for i in u_rstats_files if t in i]

        df = pd.DataFrame()
        df["Radius (r)"] = pd.Series(np.arange(2,100))
        for i, filename in enumerate(t_files):
            # Construct the path for the CSV file
            fullpath = os.path.join(rstats_path, f"{filename}.csv")
            prefix, _, date, id, mode, label, _ = filename.split("_")
            
            # Construct the path for the random CSV file
            rand_fullpath = os.path.join(rstats_path, f"FV10__{date}_{id}_random_{mode}_{label}_rstats.csv")
    
            # Load the CSV file and random CSV file into DataFrames
            rstats = pd.read_csv(fullpath)
            rand_rstats = pd.read_csv(rand_fullpath)

            K_norm = normalize(rstats, rand_rstats)
            df[f"Sample {i+1}"] = pd.Series(K_norm)

        df['Average'] = df.drop('Radius (r)', axis=1).mean(axis=1)
        df_long = pd.melt(df, id_vars=["Radius (r)"], value_vars=["Average"]+[f"Sample {i+1}" for i in range(8)],
                          var_name="Sample", value_name="K_norm")
        _draw_combined_graph(df_long, title=t)

def plot_combined_multivariate(rstats_path):
    def get_rstats_files(path):
        return glob(f"{path}/*.csv")

    def group_items(l, n=4):
        """Groups every 'n' items in the list into sublists."""
        return [l[i:i + n] for i in range(0, len(l), n)]

    def rearrange_sublists(lst, order):
        """Rearranges elements within each sublist of a list based on a specified order."""
        return [[sublist[i] for i in order] for sublist in lst]
        
    rstats_files = get_rstats_files(rstats_path)
    print("Loaded:", rstats_files)

    # filter our monte carlo results
    rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
    m_rstats_files = sorted([f for f in rstats_files if "multivariate" in f])

    anchor = ["tumor", "ng2"]
    types = ["ng2", "branch", "tvc"]
    titles = ["Tumor-NG2 Relationship", "Tumor-Branch Relationship", "Tumor-Tortuous Vessel Relationship", "NG2-Branch Relationship",
              "NG2-Tortuous Vessel Relationship"]
    i_cnt = 0
    for a in anchor:
        for t in types:
            if a != t: 
                a_files = [i for i in m_rstats_files if a in i and t in i]

                df = pd.DataFrame()
                df["Radius (r)"] = pd.Series(np.arange(2,100))
                for i, filename in enumerate(a_files):
                    # Construct the path for the CSV file
                    fullpath = os.path.join(rstats_path, f"{filename}.csv")
                    prefix, _, date, id, mode, anchor, target, _ = filename.split("_")
            
                    # Construct the path for the random CSV file
                    rand_fullpath = os.path.join(rstats_path, f"FV10__{date}_{id}_random_{mode}_{anchor}_{target}_rstats.csv")
            
                    # Load the CSV file and random CSV file into DataFrames
                    rstats = pd.read_csv(fullpath)
                    rand_rstats = pd.read_csv(rand_fullpath)
        
                    K_norm = normalize(rstats, rand_rstats)
                    df[f"Sample {i+1}"] = pd.Series(K_norm)

                df['Average'] = df.drop('Radius (r)', axis=1).mean(axis=1)
        
                df_long = pd.melt(df, id_vars=["Radius (r)"], value_vars=["Average"]+[f"Sample {i+1}" for i in range(8)],
                                  var_name="Sample", value_name="K_norm")
                _draw_combined_graph(df_long, title=titles[i_cnt])
                i_cnt += 1

def plot_combined_platelets(rstats_path):
    def get_rstats_files(path):
        return glob(f"{path}/*.csv")

    def get_id_from_name(filename):
        return "_".join(filename.split("_")[:3])

    ttest_data = {}

    types = ["control", "plerixafor"]
    for t in types:
        t_rstats_path = os.path.join(rstats_path, t)
        rstats_files = sorted(get_rstats_files(t_rstats_path))

        # filter our monte carlo results
        main_rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
        rand_rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" in f]
        assert len(main_rstats_files) == len(rand_rstats_files)
        paired_files = list(zip(main_rstats_files, rand_rstats_files))

        ids = sorted(list(set([get_id_from_name(f) for f in main_rstats_files])))
        id_map = {f: i for i, f in enumerate(ids)}
        sample_map = {}

        # Stores filename: segment rstat values for every mouse
        # This allows averaging across multiple samples from each mouse
        individual_mouse_segments = {f: pd.DataFrame() for f in ids}

        # After average rstats values are calculated for each mice, the average
        # is migrated to the all_mice_averages_df dataframe
        all_mice_averages_df = pd.DataFrame()
        sample_counts = defaultdict(int)

        df = pd.DataFrame()
        for i, pair in enumerate(paired_files):
            filename, rand_filename = pair
            s_id = get_id_from_name(filename)

            # Construct the path for the CSV file
            fullpath = os.path.join(t_rstats_path, f"{filename}.csv")
            rand_fullpath = os.path.join(t_rstats_path, f"{rand_filename}.csv")

            # Load the CSV file and random CSV file into DataFrames
            rstats = pd.read_csv(fullpath)
            rand_rstats = pd.read_csv(rand_fullpath)

            # Run once at the beginning
            if i == 0:
                for dframe in [df, all_mice_averages_df] + list(individual_mouse_segments.values()):
                    dframe["Radius (r)"] = pd.Series(np.arange(2, rstats["Radius (r)"].max()+1))
                # all_mice_averages_df["Radius (r)"] =  pd.Series(np.arange(2, rstats["Radius (r)"].max()+1))
                # for mouse_id, mouse_df in individual_mouse_segments.items():
                #     mouse_df["Radius (r)"] = pd.Series(np.arange(2, rstats["Radius (r)"].max()+1))

            # Perform MDSpacer normalization to normalize 95% interval to
            # between -1 and 1 with other values adjusted proportionately
            K_norm = normalize(rstats, rand_rstats)
            
            sample_counts[s_id] += 1
            individual_mouse_segments[s_id][f"Sample {sample_counts[s_id]}"] = pd.Series(K_norm)
            
            # df[f"Sample {i+1}"] = pd.Series(K_norm)
            # sample_map[f"Sample {i+1}"] = int(id_map[get_id_from_name(filename)])


        for i, (mouse_id, mouse_df) in enumerate(individual_mouse_segments.items()):
            all_mice_averages_df[f"Mouse {i+1}"] = mouse_df.iloc[:, 1:].mean(axis=1)
            mouse_df["Average"] = mouse_df.iloc[:, 1:].mean(axis=1)
            
        all_mice_averages = all_mice_averages_df.drop('Radius (r)', axis=1)
        all_mice_averages_df["Average"] = all_mice_averages.mean(axis=1)
        df_long = pd.melt(all_mice_averages_df, id_vars=["Radius (r)"],
                          value_vars=["Average"]+[f"Mouse {i+1}" for i in range(len(ids))],
                          var_name="Mouse", value_name="K_norm")

        palette = sns.color_palette("hls", len(ids)+1)
        _draw_combined_graph(df_long, title=t, var_name="Mouse", force_ylim=(-24, 140),
                             palette=palette, legend=False, save=False,
                             output_filename=f"ripley_combined_plot_{t}.svg")

        for mouse_id, mouse_df in individual_mouse_segments.items():
            palette = sns.color_palette("hls", sample_counts[mouse_id]+1)
            mouse_df_long = pd.melt(mouse_df, id_vars=["Radius (r)"],
                                    value_vars=["Average"]+[f"Sample {i+1}" for i in range(sample_counts[mouse_id])],
                                    var_name="Sample", value_name="K_norm")
            _draw_combined_graph(mouse_df_long, title=t, var_name="Sample", force_ylim=(-24, 180),
                                 palette=palette, legend=False, save=True,
                                 output_filename=f"ripley_individual_plot_{mouse_id}_{t}.svg")


        mouse_columns = [col for col in all_mice_averages_df.columns if col.startswith("Mouse")]
        columns_to_select = ["Radius (r)"] + mouse_columns

        ttest_data[t] = all_mice_averages_df[columns_to_select]

    ttest_results = run_ttest(ttest_data["control"], ttest_data["plerixafor"])
    significant_radii = [(r, p) for r, p in ttest_results if p <= 0.05]
    v_significant_radii = [(r, p) for r, p in ttest_results if p <= 0.01]
    print(significant_radii)
    print(v_significant_radii)

def run_ttest(group1_df, group2_df):
    p_values = []
    for (r1, g1), (r2, g2) in zip(group1_df.groupby("Radius (r)"), group2_df.groupby("Radius (r)")):
        mouse_columns = [col for col in g1.columns if col.startswith("Mouse")]
        g1_values = g1[mouse_columns].values.tolist()[0]
        g2_values = g2[mouse_columns].values.tolist()[0]

        t_stat, p_value = stats.ttest_ind(g1_values, g2_values)
        p_values.append((r1, p_value))
    return p_values

def plot_p_values(rstats_path, save=False, output_folder="./ripley_results/"):
    def get_rstats_files(path):
        return glob(f"{path}/*.csv")

    def group_items(l, n=4):
        """Groups every 'n' items in the list into sublists."""
        return [l[i:i + n] for i in range(0, len(l), n)]

    def rearrange_sublists(lst, order):
        """Rearranges elements within each sublist of a list based on a specified order."""
        return [[sublist[i] for i in order] for sublist in lst]
        
    rstats_files = get_rstats_files(rstats_path)
    print("Loaded:", rstats_files)

    # filter our monte carlo results
    rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
    u_rstats_files = sorted([f for f in rstats_files if "univariate" in f])
    m_rstats_files = sorted([f for f in rstats_files if "multivariate" in f])

    types = ["tumor", "ng2", "branch", "tvc"]
    for t in types:
        t_files = [i for i in u_rstats_files if t in i]

        df = pd.DataFrame()
        df["Radius (r)"] = pd.Series(np.arange(2,100))
        for i, filename in enumerate(t_files):
            # Construct the path for the CSV file
            fullpath = os.path.join(rstats_path, f"{filename}.csv")
            prefix, _, date, id, mode, label, _ = filename.split("_")
            
            # Construct the path for the random CSV file
            rand_fullpath = os.path.join(rstats_path, f"FV10__{date}_{id}_random_{mode}_{label}_rstats.csv")
    
            # Load the CSV file and random CSV file into DataFrames
            rstats = pd.read_csv(fullpath)
            rand_rstats = pd.read_csv(rand_fullpath)
            print(rstats)
            break


def create_directory(path):
    """
    Creates directory at path if not exists
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    if isinstance(arr, cp.ndarray):
        nonzero = cp.nonzero
    elif isinstance(arr, np.ndarray):
        nonzero = np.nonzero
    else:
        raise ValueError("arr needs to be np or cp ndarray type")

    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in nonzero(arr))
    return arr[slices]

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
    point1 (array-like): An array-like object representing the first point.
    point2 (array-like): An array-like object representing the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    # Use numpy's linalg.norm function to calculate the Euclidean distance
    return np.linalg.norm(np.array(point1)-np.array(point2))

def replace_np_values(arr: np.ndarray, map: dict) -> np.ndarray:
    fn = np.vectorize(lambda x: map.get(x, 0) * 255)
    return fn(arr)

# from glob import glob
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# def _calculate_p_values_two_tailed(observed_ripleyK, simulated_ripleyKs):
#     """
#     Calculate two-tailed p-values for Ripley's K function at each radius.
# 
#     Parameters:
#     observed_ripleyK (array-like): An array of Ripley's K values calculated from the observed data for each radius.
#     simulated_ripleyKs (array-like of array-like): A 2D array where each row represents the Ripley's K values 
#                                                     from a single Monte Carlo simulation across the same radii as the observed data.
# 
#     Returns:
#     np.array: An array of two-tailed p-values for each radius.
#     """
#     # Initialize an array to hold p-values for each radius
#     p_values = np.zeros_like(observed_ripleyK, dtype=float)
# 
#     # Calculate p-value for each radius
#     for i in range(len(observed_ripleyK)):
#         # Count how many simulated Ripley's K values are more extreme than the observed value
#         # For a two-tailed test, consider both tails
#         # print((i, simulated_ripleyKs[:, i], observed_ripleyK[i]))
#         lower_extreme_count = np.sum(simulated_ripleyKs[:, i] < observed_ripleyK[i])
#         upper_extreme_count = np.sum(simulated_ripleyKs[:, i] >= observed_ripleyK[i])
#         
#         # For two-tailed, the extremeness is considered from both ends; thus, take the minimum count
#         # and multiply by 2 to account for both tails
#         extreme_count = min(lower_extreme_count, upper_extreme_count) * 2
#         
#         # Ensure the p-value does not exceed 1
#         p_values[i] = np.minimum(extreme_count / simulated_ripleyKs.shape[0], 1.0)
# 
#     return p_values
# 
# def plot_p_values(rstats_path, save=False, output_folder="./ripley_results/"):
#     palette = sns.color_palette("rocket_r")
#     def get_rstats_files(path):
#         return glob(f"{path}/*.csv")
# 
#     def _plot(p_values_array, radii):
#         # p_values_df = pd.DataFrame({
#         #     "P-Value": np.concatenate(p_values_array),  # Flatten the list of arrays into one long array
#         #     "Set": np.repeat(np.arange(1, len(p_values_array)+1), len(p_values_array[0])),
#         #     "Radius": np.tile(radii, len(p_values_array)),
#         # })
#         p_values_long_list = []
#         for i, p_values in enumerate(p_values_array, start=1):
#             for radius, p_value in zip(radii, p_values):
#                 p_values_long_list.append({
#                     "Radius": radius,
#                     "P-Value": p_value,
#                     "Set": f"Sample {i}"  # Label each set for identification
#                 })
#         p_values_df = pd.DataFrame(p_values_long_list)
# 
#         plt.figure(figsize=(8, 6))  # Set the figure size for better visibility
# 
#         # Plot using seaborn
#         sns.lineplot(data=p_values_df, x='Radius', y='P-Value', hue="Set", style="Set", linestyle='-', zorder=3, alpha=0.3, legend="brief")
# 
#         # Draw a dotted horizontal line for the 0.05 threshold
#         color = palette[4]
#         plt.hlines(
#               y=0.05,
#               xmin=np.min(radii),
#               xmax=np.max(radii),
#               color=get_equivalent_color(color, 0.4),
#               label='p<0.05 Threshold',
#               linewidth=1
#         )
#         plt.fill_between(radii, 0, 0.05, color=get_equivalent_color(color, 0.1))
# 
#         # Flip the y-axis
#         plt.gca().invert_yaxis()
# 
#         # Add labels and title for clarity
#         plt.xlabel('Radius')
#         plt.ylabel('P-Value')
#         plt.legend()
#         plt.show()
#         
#     
# 
#     rstats_files = get_rstats_files(rstats_path)
#     print("Loaded:", rstats_files)
# 
#     # filter our monte carlo results
#     rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
#     u_rstats_files = sorted([f for f in rstats_files if "univariate" in f])
#     m_rstats_files = sorted([f for f in rstats_files if "multivariate" in f])
# 
#     types = ["tumor", "ng2", "branch", "tvc"]
#     for t in types[:1]:
#         t_files = [i for i in u_rstats_files if t in i]
# 
#         p_values_array = []
#         for i, filename in enumerate(t_files):
#             # Construct the path for the CSV file
#             fullpath = os.path.join(rstats_path, f"{filename}.csv")
#             prefix, _, date, id, mode, label, _ = filename.split("_")
#             
#             # Construct the path for the random CSV file
#             rand_fullpath = os.path.join(rstats_path, f"FV10__{date}_{id}_random_{mode}_{label}_rstats.csv")
#     
#             # Load the CSV file and random CSV file into DataFrames
#             rstats = pd.read_csv(fullpath)
#             rand_rstats = pd.read_csv(rand_fullpath)
# 
#             # plot_ripley(rstats, rand_rstats)
# 
#             observed_ripleyK = rstats["K(r)"].to_numpy()
# 
#             # Determine the range of radii
#             min_radius, max_radius = rstats["Radius (r)"].min(), rstats["Radius (r)"].max()
#             expected_length = max_radius - min_radius + 1  # Adjust based on your radius range
# 
#             # Group by 'Line' and extract 'K(r)' values
#             groups = rand_rstats.groupby('Line')['K(r)']
# 
#             simulated_arrays = []
#             for name, group in groups:
#                 # Ensure the group has the expected length; you might need to handle cases where it doesn't
#                 assert len(group) == expected_length
#                 simulated_arrays.append(group.values)
#             simulated_ripleyKs = np.array(simulated_arrays)
# 
#             p_values = _calculate_p_values_two_tailed(observed_ripleyK, simulated_ripleyKs)
#             p_values_array.append(p_values)
#         _plot(p_values_array, np.arange(min_radius, max_radius+1))
#         break
#             