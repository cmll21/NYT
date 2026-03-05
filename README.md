# NYT Puzzle Solvers

This repository contains solvers for four New York Times puzzles: **Wordle**, **Connections**, **Spelling Bee**, and **Strands**.

## Wordle Solver

- **Strategies:** frequency, entropy, or hybrid.
- **Modes:**
  - **Simulation:** Runs the solver on all target words and displays stats.
  - **Manual:** Interactive mode for entering guesses and feedback.
- **Visualization:** Optional plots to show performance.
- **Files:**
  - `wordle/solver.py` – Core solving logic.
  - `wordle/data.py` – Word list loading.
  - `wordle/views.py` – Terminal/plot rendering.
  - `wordle/app.py` – App orchestration (simulate/manual).
  - `wordle/cli.py` – Command line entry point.
  - `wordle/allowed-guesses.txt` – Allowed guess words.
  - `wordle/answers-alphabetical.txt` – Valid answer words.

## Connections Solver

- **Method:** Uses SpaCy and KMeans clustering.
- **Modes:**
  - **Simulation:** Automatically solves a puzzle and shows a summary.
  - **Manual:** Interactive mode for entering guesses.
- **Visualization:** Optional plots to display feedback and progress.
- **Files:**
  - `connections/solver.py` – Core solving logic.
  - `connections/data.py` – Puzzle loading/parsing.
  - `connections/views.py` – Terminal/plot rendering.
  - `connections/app.py` – App orchestration (simulate/manual).
  - `connections/cli.py` – Command line entry point.
  - `connections/answers.json` – JSON file with puzzle data.

## Strands Solver

- **Files:**
  - `strands/solver.py` – Core solving logic.
  - `strands/views.py` – Terminal board visualization.
  - `strands/app.py` – App orchestration (simulate/manual).
  - `strands/cli.py` – Command line entry point.

## Bee Solver

- **Files:**
  - `bee/solver.py` – Core solving logic.
  - `bee/app.py` – App orchestration (simulate/manual).
  - `bee/cli.py` – Command line entry point.

## Requirements

- Python 3.10 or later.
- Packages: `argparse`, `numpy`, `matplotlib`, `spacy`, `scikit-learn`, `nltk`.

For the Connections solver, download the SpaCy model:
```bash
python -m spacy download en_core_web_lg
```

## Usage

### Wordle Solver

#### Simulation Mode:
```bash
python3 -m wordle.cli --mode simulate --strategy hybrid --initial-guess slate --visualize
```

#### Manual Mode:
```bash
python3 -m wordle.cli --mode manual --strategy frequency
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
python3 -m connections.cli --mode simulate --file connections/answers.json --id 1 --visualize
```

#### Manual Mode:
```bash
python3 -m connections.cli --mode manual --file connections/answers.json --id 1
```

#### Arguments:
- `--mode`: simulate or manual (default is simulate).
- `--file`: Path to the JSON file with puzzle data.
- `--id`: (Optional) Specific game ID to solve.
- `--visualize`: Enable visualization after simulation.

### Strands Solver

#### Simulation Mode:
```bash
python3 -m strands.cli --mode simulate --theme "The cutting edge" --max-solutions 20 --visualize
```

#### Manual Mode:
```bash
python3 -m strands.cli --mode manual --theme "The cutting edge"
```

#### Arguments:
- `--mode`: simulate or manual (default is simulate).
- `--max-solutions`: Maximum number of solutions to enumerate.
- `--no-spangram-required`: Allow solutions without a spangram.
- `--visualize`: Show interactive terminal visualization in simulate mode.
- `--theme`: Optional theme phrase used for semantic ranking.

### Bee Solver

#### Manual Mode:
```bash
python3 -m bee.cli --mode manual
```

## License
This project is licensed under the MIT License.
