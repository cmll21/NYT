# NYT Puzzle Solvers

This repository contains solvers for four New York Times puzzles: **Wordle**, **Connections**, **Spelling Bee**, and **Strands**.

## Wordle Solver

- **Strategies:** frequency, entropy, or hybrid.
- **Modes:**
  - **Simulation:** Runs the solver on all target words and displays stats.
  - **Manual:** Interactive mode for entering guesses and feedback.
- **Visualisation:** Optional plots to show performance.
- **Files:**
  - `wordle/allowed-guesses.txt` – Allowed guess words.
  - `wordle/answers-alphabetical.txt` – Valid answer words.
  - `wordle/wordle.py` – Main Wordle solver logic.
  - `wordle/cli.py` - Main command line interface script.

## Connections Solver

- **Method:** Uses SpaCy and KMeans clustering.
- **Modes:**
  - **Simulation:** Automatically solves a puzzle and shows a summary.
  - **Manual:** Interactive mode for entering guesses.
- **Visualisation:** Optional plots to display feedback and progress.
- **Files:**
  - `connections/answers.json` – JSON file with puzzle data.
  - `connections/connections.py` – Main Connections solver script.
  - `connections/cli.py` - Main command line interface script.

## Requirements

- Python 3.7 or later.
- Packages: `argparse`, `numpy`, `matplotlib`, `spacy`, `scikit-learn`, `nltk`.

For the Connections solver, download the SpaCy model:
```bash
python -m spacy download en_core_web_lg
```

## Usage

### Wordle Solver

#### Simulation Mode:
```bash
python3 wordle/cli.py --mode simulate --strategy hybrid --initial-guess slate --visualize
```

#### Manual Mode:
```bash
python3 wordle/cli.py --mode manual --strategy frequency
```

#### Arguments:
- `--mode`: simulate or manual (default is simulate).
- `--strategy`: frequency, entropy, or hybrid (default is hybrid).
- `--words-file`: Path to allowed guesses file.
- `--answers-file`: Path to answers file.
- `--initial-guess`: (Optional) 5 letter starting guess.
- `--visualize`: Enable visualization after solving.

### Connections Solver

#### Simulation Mode:
```bash
python3 connections/cli.py --mode simulate --file connections/answers.json --id 1 --visualize
```

#### Manual Mode:
```bash
python3 connections/cli.py --mode manual --file connections/answers.json --id 1
```

#### Arguments:
- `--mode`: simulate or manual (default is simulate).
- `--file`: Path to the JSON file with puzzle data.
- `--id`: (Optional) Specific game ID to solve.
- `--visualize`: Enable visualization after simulation.

## License
This project is licensed under the MIT License.
