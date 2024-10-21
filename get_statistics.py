import argparse
import json
import os

import numpy as np

import matplotlib.pyplot as plt

from occam_llm.evaluation import check_output


def fix_statistics(data, print_results=False):
    changed_something = False

    for datapoint in data:
        is_correct, rel_error = check_output(datapoint["correct_answer"], datapoint["output"], print_results = print_results)

        if int(is_correct) != datapoint["is_correct"]:
            datapoint["is_correct"] = int(is_correct)
            changed_something = True

        if not np.isclose(rel_error, datapoint["rel_error"]):
            datapoint["rel_error"] = rel_error
            changed_something = True

    return changed_something

def get_performance_and_error(file, model_name, print_results=False):
    try:
        with open(file) as f:
            data = json.load(f)

            changed_something = fix_statistics(data, print_results=print_results)


            is_correct = [datapoint["is_correct"] for datapoint in data]

            relative_error = np.array([datapoint["rel_error"] for datapoint in data])
            relative_error = relative_error[np.isfinite(relative_error)]

            if print_results:
                print(f"On {file.split('/')[-1][:-5]}, {model_name} has accuracy {np.mean(is_correct)*100:0.1f}\\pm {100*np.std(is_correct, ddof=1)/np.sqrt(len(is_correct)):0.1f}\\% and relative error {np.mean(relative_error)*100:0.1f}\\pm {100*np.std(relative_error, ddof=1)/np.sqrt(len(relative_error)):0.1f}\\%")
        
        if changed_something:
            with open(file, 'w') as f:
                json.dump(data, f)
                print(f"Fixed {file}\n\n")

        return np.mean(is_correct), np.std(is_correct, ddof=1)/np.sqrt(len(is_correct)), np.mean(relative_error), np.std(relative_error, ddof=1)/np.sqrt(len(relative_error))
    
    except:
        return None, None, None, None


def get_average_performance_and_error(files):
    accuracies = []
    accuracy_std_errors = []
    relative_errors = []
    relative_error_std_errors = []
    tokens_used = []

    for file in files:
        try:
            with open(file) as f:
                data = json.load(f)

                changed_something = fix_statistics(data)

                is_correct = [datapoint["is_correct"] for datapoint in data]

                relative_error = [datapoint["rel_error"] for datapoint in data]

                if len(is_correct) == 0:
                    return None, None, None, None

                accuracies.append(np.mean(is_correct))
                accuracy_std_errors.append(np.std(is_correct, ddof=1)/np.sqrt(len(is_correct)))

                relative_errors.append(np.mean(relative_error))
                relative_error_std_errors.append(np.std(relative_error, ddof=1)/np.sqrt(len(relative_error)))

                if "metadata" in data[0]:
                    tokens_used += [datapoint["metadata"]["completion_tokens"] for datapoint in data]

            if changed_something:
                with open(file, 'w') as f:
                    json.dump(data, f)
                    print(f"Fixed {file}\n\n")
        except:
            pass

    return np.mean(accuracies), np.sqrt(np.sum(np.array(accuracy_std_errors)**2))/np.sqrt(len(accuracies)), np.mean(relative_errors), np.sqrt(np.sum(np.array(relative_error_std_errors)**2))/np.sqrt(len(accuracies))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--model_type', type = str, default="gpt")
    parser.add_argument('-m', '--model', type = str, default="3.5-turbo")

    # Parse arguments

    args = parser.parse_args()

    directory_path = f"results/{args.model_type}/{args.model}"

    file_paths = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    file_paths.sort()

    print(f"Loading statistics for {file_paths}\n\n")


    for file in file_paths:
        get_performance_and_error(file, f"{args.model_type}-{args.model}", print_results=True)

