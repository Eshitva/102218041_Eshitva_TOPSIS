import sys
import pandas as pd
import numpy as np


def validate_inputs(weights, impacts, num_criteria):
    weights = weights.split(',')
    impacts = impacts.split(',')

    if len(weights) != num_criteria or len(impacts) != num_criteria:
        raise ValueError(
            "Number of weights and impacts must be equal to the number of criteria (columns from 2nd to last).")

    weights = [float(w) for w in weights]

    for impact in impacts:
        if impact not in ['+', '-']:
            raise ValueError("Impacts must be either '+' or '-'.")

    return weights, impacts


def topsis(input_file, weights, impacts, result_file):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{input_file}' not found.")

    if df.shape[1] < 3:
        raise ValueError("Input file must contain at least three columns.")

    data = df.iloc[:, 1:].values
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("From 2nd to last columns must contain numeric values only.")

    num_criteria = data.shape[1]
    weights, impacts = validate_inputs(weights, impacts, num_criteria)

    
    norm_data = data / np.sqrt((data ** 2).sum(axis=0))

    weighted_data = norm_data * weights

    
    ideal_best = np.max(weighted_data, axis=0)
    ideal_worst = np.min(weighted_data, axis=0)

    for i, impact in enumerate(impacts):
        if impact == '-':
            ideal_best[i], ideal_worst[i] = ideal_worst[i], ideal_best[i]

    separation_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    separation_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

    
    topsis_score = separation_worst / (separation_best + separation_worst)


    df['Topsis Score'] = topsis_score
    df['Rank'] = df['Topsis Score'].rank(method='max', ascending=False).astype(int)

   
    df.to_csv(result_file, index=False)
    print(f"Results saved to '{result_file}'.")


def main():
    if len(sys.argv) != 5:
        print("Usage: topsis <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    result_file = sys.argv[4]

    try:
        topsis(input_file, weights, impacts, result_file)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()