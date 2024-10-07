import sys
import pandas as pd
import numpy as np

def validateInputParameters(weightList, impactList, criteriaCount):
    weightList = weightList.split(',')
    impactList = impactList.split(',')

    if len(weightList) != criteriaCount or len(impactList) != criteriaCount:
        raise ValueError("The number of weights and impacts must match the number of criteria (columns from 2nd to last).")

    weightList = [float(w) for w in weightList]

    for impact in impactList:
        if impact not in ['+', '-']:
            raise ValueError("Impacts should be either '+' or '-'.")

    return weightList, impactList


def performTopsis(inputFileName, weightList, impactList, outputFileName):
    try:
        inputData = pd.read_csv(inputFileName)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{inputFileName}' not found.")

    if inputData.shape[1] < 3:
        raise ValueError("The input file must have at least three columns.")

    criteriaData = inputData.iloc[:, 1:].values
    if not np.issubdtype(criteriaData.dtype, np.number):
        raise ValueError("Columns from 2nd to last should contain numeric values only.")

    criteriaCount = criteriaData.shape[1]
    weightList, impactList = validateInputParameters(weightList, impactList, criteriaCount)

    
    normalizedData = criteriaData / np.sqrt((criteriaData ** 2).sum(axis=0))

    
    weightedData = normalizedData * weightList

    
    idealBest = np.max(weightedData, axis=0)
    idealWorst = np.min(weightedData, axis=0)

    for idx, impact in enumerate(impactList):
        if impact == '-':
            idealBest[idx], idealWorst[idx] = idealWorst[idx], idealBest[idx]

    
    distanceToBest = np.sqrt(((weightedData - idealBest) ** 2).sum(axis=1))
    distanceToWorst = np.sqrt(((weightedData - idealWorst) ** 2).sum(axis=1))

    
    topsisScores = distanceToWorst / (distanceToBest + distanceToWorst)

    
    inputData['Topsis Score'] = topsisScores
    inputData['Rank'] = inputData['Topsis Score'].rank(method='max', ascending=False).astype(int)

    
    inputData.to_csv(outputFileName, index=False)
    print(f"Results successfully saved to '{outputFileName}'.")


def main():
    if len(sys.argv) != 5:
        print("Usage: topsis <Weights> <Impacts> <InputDataFile> <OutputFileName>")
        sys.exit(1)

    inputFileName = sys.argv[3]
    weightList = sys.argv[1]
    impactList = sys.argv[2]
    outputFileName = sys.argv[4]

    try:
        performTopsis(inputFileName, weightList, impactList, outputFileName)
    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main()
