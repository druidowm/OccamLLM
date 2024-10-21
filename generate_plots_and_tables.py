import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from get_statistics import get_performance_and_error, get_average_performance_and_error

# Assuming this function exists
def get_data(model, dataset, appendage, type = "accuracy", print_results = False):
    data = f"results/{model}/{dataset}{appendage}.json"
    accuracy, accuracy_std, error, error_std = get_performance_and_error(data, model, print_results=print_results)

    if type == "accuracy":
        return accuracy, accuracy_std
    return error, error_std

def get_average_data(model, datasets, appendage, type = "accuracy"):
    datafiles = [f"results/{model}/{dataset}{appendage}.json" for dataset in datasets]
    accuracy, accuracy_std, error, error_std = get_average_performance_and_error(datafiles)

    if type == "accuracy":
        return accuracy, accuracy_std
    return error, error_std

def plot_results(models, model_names, model_dataset_appendages, datasets, dataset_names, y_label, fig_sizes, saveto=None, type="accuracy", print_results=False, n_rows=1):
    x = np.arange(len(datasets) + 1)
    width = fig_sizes["bar_width"]
    # Prepare the plot
    fig, ax = plt.subplots(figsize=(fig_sizes["width"], fig_sizes["height"]), constrained_layout=True)

    # Adding grids
    ax.yaxis.grid(True)

    # Adding a vertical line to separate the average
    ax.axvline(len(datasets) - 0.5, color='gray', linestyle='--')

    # Adding a shaded background for the average
    ax.axvspan(len(datasets) - 0.5, len(datasets) + 0.5, color='gray', alpha=0.2)

    # Adding alternating background colors for datasets
    for i in range(len(datasets)):
        color = 'lightgrey' if i % 2 == 1 else 'white'
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.2)

    for i, (model, model_name, appendage) in enumerate(zip(models, model_names, model_dataset_appendages)):
        values = []
        errors = []
        for dataset in datasets:
            performance, error = get_data(model, dataset, appendage, type=type, print_results=print_results)
            if performance is None:
                values.append(np.nan)
                errors.append(np.nan)
            else:
                values.append(100 * performance)
                errors.append(100 * error)

        performance, error = get_average_data(model, datasets, appendage, type=type)
        if performance is None:
            values.append(np.nan)
            errors.append(np.nan)
        else:
            values.append(100 * performance)
            errors.append(100 * error)

        if not all(np.isnan(values)):
            ax.bar(x + i * width - (len(models) - 1) * width / 2, values, width, yerr=errors, label=model_name)

    # Adding labels
    ax.set_ylabel(y_label, fontsize=fig_sizes["ylabel_fontsize"])
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fig_sizes["yticklabel_fontsize"])
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names + ["Average"], rotation=45, fontsize=fig_sizes["xticklabel_fontsize"])

    # Adjust legend
    ax.legend(loc='upper center', bbox_to_anchor=fig_sizes["bbox_to_anchor"], ncol=math.ceil(len(models)/n_rows), fontsize=fig_sizes["legend_fontsize"])

    ax.set_xlim(-0.5, len(datasets) + 0.5)

    ax.set_axisbelow(True)

    if saveto is not None:
        plt.savefig(saveto)

    # Display the plot
    plt.show()

def generate_latex_table(models, model_names, model_dataset_appendages, datasets, dataset_names, table_caption, table_ref, type="accuracy", print_results=False, bold_max = True):
    # Initialize the table string
    table = """
\\begin{table}[t]
\\centering
\\caption{""" + table_caption + """}
\\begin{tabular}{l""" + "c" * len(model_names) + """}
\\toprule
& """ + " & ".join([name[0] for name in model_names]) + " \\\\ \n" + \
    "& " + " & ".join([name[1] for name in model_names]) + " \\\\ \n" + \
    "\n".join(["\\cmidrule(r){" + str(i+2) + "-" + str(i+2) + "}" for i in range(len(model_names))]) + "\n"

    # Append rows to the table
    for dataset_name, dataset in zip(dataset_names, datasets):
        row = [dataset_name]

        performances = []
        errors = []
        for model, model_name, appendage in zip(models, model_names, model_dataset_appendages):
            performance, error = get_data(model, dataset, appendage, type=type, print_results=print_results)

            performances.append(performance)
            errors.append(error)

        try:
            if bold_max:
                max_performance = max([performance for performance in performances if performance is not None])
            else:
                max_performance = min([performance for performance in performances if performance is not None])
        except:
            max_performance = None

        for performance, error in zip(performances, errors):
            if performance is None:
                row.append("")
            else:
                if performance == max_performance:
                    row.append("\\textbf{" + f"{performance*100:.1f}\\scriptsize" + "{\\textcolor{gray}{" + f"$\\pm${error*100:.1f}" + "}}}")
                else:
                    row.append(f"{performance*100:.1f}\\scriptsize" + "{\\textcolor{gray}{" + f"$\\pm${error*100:.1f}" + "}}")
        table += " & ".join(row) + " \\\\ \n"

    # Add the footer of the table
    table += "\\midrule\n\\textsc{Average} "

    performances = []
    errors = []
    for model, appendage in zip(models, model_dataset_appendages):
        performance, error = get_average_data(model, datasets, appendage, type=type)

        performances.append(performance)
        errors.append(error)

    if bold_max:
        max_performance = max([performance for performance in performances if performance is not None])
    else:
        max_performance = min([performance for performance in performances if performance is not None])

    for performance, error in zip(performances, errors):
        if performance is None:
            table += "& "
        else:
            if performance == max_performance:
                table += "& \\textbf{" + f"{performance*100:.1f}\\scriptsize" + "{\\textcolor{gray}{" + f"$\\pm${error*100:.1f}" + "}}}"
            else:
                table += f"& {performance*100:.1f}\\scriptsize" + "{\\textcolor{gray}{" + f"$\\pm${error*100:.1f}" + "}} "
    table += """\\\\\n
\\bottomrule
\\end{tabular}
\\label{tab:"""
    table += f"{table_ref}" + "}\n\\end{table}"

    print(table)
    return table

def generate_markdown_table(models, model_names, model_dataset_appendages, datasets, dataset_names, table_caption, table_ref, type="accuracy", print_results=False, bold_max=True):
    # Initialize the table string
    table = f"{table_caption}\n\n"
    
    # Create the header row
    header = "| Dataset | " + " | ".join([f"{name[0]}<br>{name[1]}" for name in model_names]) + " |"
    separator = "|:--|" + "|".join([":--:" for _ in model_names]) + "|"
    
    table += header + "\n" + separator + "\n"
    
    # Append rows to the table
    for dataset_name, dataset in zip(dataset_names, datasets):
        row = [dataset_name]
        performances = []
        errors = []
        for model, model_name, appendage in zip(models, model_names, model_dataset_appendages):
            performance, error = get_data(model, dataset, appendage, type=type, print_results=print_results)
            performances.append(performance)
            errors.append(error)
        try:
            if bold_max:
                max_performance = max([p for p in performances if p is not None])
            else:
                max_performance = min([p for p in performances if p is not None])

        except:
            max_performance = None
        
        for performance, error in zip(performances, errors):
            if performance is None:
                row.append("")
            else:
                value = f"{performance*100:.1f} ± {error*100:.1f}"
                if performance == max_performance:
                    row.append(f"**{value}**")
                else:
                    row.append(value)
        
        table += "| " + " | ".join(row) + " |\n"
    
    # Add the footer of the table
    footer = ["**Average**"]
    performances = []
    errors = []
    for model, appendage in zip(models, model_dataset_appendages):
        performance, error = get_average_data(model, datasets, appendage, type=type)
        performances.append(performance)
        errors.append(error)
    
    if bold_max:
        max_performance = max([p for p in performances if p is not None])
    else:
        max_performance = min([p for p in performances if p is not None])
    
    for performance, error in zip(performances, errors):
        if performance is None:
            footer.append("")
        else:
            value = f"{performance*100:.1f} ± {error*100:.1f}"
            if performance == max_performance:
                footer.append(f"**{value}**")
            else:
                footer.append(value)
    
    table += "| " + " | ".join(footer) + " |\n"
    
    table += f"\n*Table {table_ref}*"
    print(table)
    return table

def main():
    parser = argparse.ArgumentParser(description='Plot performance of models on various datasets')
    parser.add_argument('--models', nargs='+', required=True, help='List of models to evaluate')
    parser.add_argument('--datasets', nargs='+', required=True, help='List of datasets to evaluate on')

    args = parser.parse_args()
    models = args.models
    datasets = args.datasets

    plot_results(models, datasets)

if __name__ == "__main__":
    main()