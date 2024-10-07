# TOPSIS Implementation

## Overview

TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) is a multi-criteria decision-making method. This Python package implements the TOPSIS method, allowing you to rank alternatives based on multiple criteria.

This implementation is specifically designed to work with a CSV input file and provides results with TOPSIS scores and ranks.

## Features

- Handles multi-criteria decision-making problems
- Accepts CSV input files
- Outputs a CSV file with TOPSIS scores and ranks
- Validates input data and handles common errors
- Command-line interface for ease of use

## Installation

You can install this package directly from PyPI using pip:

```bash
pip install 102218041-topsis
```

## Usage

### Command-Line Interface

After installing the package, you can use it from the command line. The basic usage is:

```bash
topsis <InputDataFile> <Weights> <Impacts> <ResultFileName>
```

### Parameters

- **InputDataFile**: The path to the input CSV file containing the data.
- **Weights**: A string of comma-separated numerical weights (e.g., `"1,1,1,2"`).
- **Impacts**: A string of comma-separated `+` or `-` signs indicating whether each criterion is beneficial (`+`) or non-beneficial (`-`) (e.g., `"+,+,-,+"`).
- **ResultFileName**: The path to the output CSV file where results will be saved.

### Example

Suppose you have an input file named `102218041-data.csv` with the following content:

```csv
Alternative,Criterion1,Criterion2,Criterion3,Criterion4
M1,250,16,12,5
M2,200,6,8,3
M3,300,16,8,4
M4,275,10,10,4
```

You can run the following command:

```bash
topsis 102218041-data.csv "1,1,1,2" "+,+,-,+" 102218041-result.csv
```

This will generate a file `102218041-result.csv` containing the input data along with two additional columns for the TOPSIS score and the rank:

```csv
Alternative,Criterion1,Criterion2,Criterion3,Criterion4,TOPSIS Score,Rank
M1,250,16,12,5,0.556,2
M2,200,6,8,3,0.222,4
M3,300,16,8,4,0.778,1
M4,275,10,10,4,0.444,3
```

### Error Handling

The program includes several checks to handle common errors:

1. **File Not Found**: If the input file is not found, an appropriate error message will be displayed.
2. **Invalid Number of Parameters**: The program checks if the correct number of parameters is provided.
3. **Invalid Data**: The input file must contain at least three columns, with the first column being the name of alternatives, and the rest must contain numeric values.
4. **Weights and Impacts Validation**: The number of weights and impacts must match the number of criteria in the input file. Impacts must be either `+` or `-`.

## Examples

Here are a few examples to demonstrate how the package can be used:

### Example 1: Basic Usage

```bash
topsis data.csv "1,1,1,1" "+,+,+,+" result.csv
```

### Example 2: Different Weights and Impacts

```bash
topsis data.csv "0.5,1,1.5,2" "+,-,+,-" result.csv
```

### Example 3: Handling Errors

If the number of weights or impacts does not match the number of criteria:

```bash
topsis data.csv "1,1,1" "+,+" result.csv
```

This will result in an error message indicating the mismatch.

## Testing

To ensure everything works correctly, you can run some tests. If you haven't already, you can clone the repository and run the tests locally.

```bash
# Clone the repository
git clone https://github.com/Eshitva/102218041_Eshitva_TOPSIS.git

# Navigate to the project directory
cd 102218041_Eshitva_TOPSIS

# Install the required dependencies
pip install -r requirements.txt

# Run the tests (assuming you've created test scripts)
python -m unittest discover tests
```

## Development

If you'd like to contribute to this project, follow these steps:

1. Fork the repository on GitHub.
2. Clone your fork to your local machine.
3. Create a new branch for your feature or bugfix.
4. Make your changes and commit them with descriptive messages.
5. Push your changes to your fork.
6. Submit a pull request to the original repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions, suggestions, or issues, feel free to contact me at `goyaleshitva@gmail.com`.

## Acknowledgments

- Thanks to P.S. Rana for their support.
- Inspired by the original TOPSIS method developed by C.L. Hwang and K. Yoon.

