# NYT Puzzle Solvers

This repository contains solvers for two New York Times puzzles: **Wordle** and **Connections**.

## Wordle Solver

- **Strategies:** frequency, entropy, or hybrid.
- **Modes:**
  - **Simulation:** Runs the solver on all target words and displays stats.
  - **Manual:** Interactive mode for entering guesses and feedback.
- **Visualisation:** Optional plots to show performance.
- **Files:**
  - `wordle/allowed-guesses.txt` – Allowed guess words.
  - `wordle/answers-alphabetical.txt` – Valid answer words.
  - `wordle.py` – Main Wordle solver logic.
  - `cli.py` - Main command line interface script.

## Connections Solver

- **Method:** Uses SpaCy and KMeans clustering.
- **Modes:**
  - **Simulation:** Automatically solves a puzzle and shows a summary.
  - **Manual:** Interactive mode for entering guesses.
- **Visualisation:** Optional plots to display feedback and progress.
- **Files:**
  - `connections/answers.json` – JSON file with puzzle data.
  - `connections_solver.py` – Main Connections solver script.
  - `cli.py` - Main command line interface script.

## Requirements

- Python 3.7 or later.
- Packages: `argparse`, `numpy`, `pandas`, `matplotlib`, `spacy`, `scikit-learn`.

For the Connections solver, download the SpaCy model:
```bash
python -m spacy download en_core_web_lg
```

## Usage

### Wordle Solver

#### Simulation Mode:
```bash
python wordle_solver.py --mode simulate --strategy hybrid --initial-guess slate --visualise
```

#### Manual Mode:
```bash
python wordle_solver.py --mode manual --strategy frequency
```

#### Arguments:
- `--mode`: simulate or manual (default is simulate).
- `--strategy`: frequency, entropy, or hybrid (default is hybrid).
- `--words-file`: Path to allowed guesses file.
- `--answers-file`: Path to answers file.
- `--initial-guess`: (Optional) 5 letter starting guess.
- `--visualise`: Enable visualisation after solving.

### Connections Solver

#### Simulation Mode:
```bash
python connections_solver.py --mode simulate --file connections/answers.json --id 1 --visualise
```

#### Manual Mode:
```bash
python connections_solver.py --mode manual --file connections/answers.json --id 1
```

#### Arguments:
- `--mode`: simulate or manual (default is simulate).
- `--file`: Path to the JSON file with puzzle data.
- `--id`: (Optional) Specific game ID to solve.
- `--visualise`: Enable visuailsation after simulation.

## License
This project is licensed under the MIT License.
