# Dynamic Discounting Model Evaluation

## Overview
This application helps evaluate and compare different forecasting models for dynamic discounting. It provides a command-line interface to run model evaluations and compare performance across different models.

## Features
- Run evaluations for any forecasting model
- Compare performance metrics between models
- Visualize evaluation results
- Analyze model performance by discount levels

## Directory Structure
```
.
├── app.py                        # Main application entry point
├── data_files/                   # Training and test datasets
├── evaluation_scripts/           # Model evaluation scripts
├── model_evaluation_results/     # Evaluation results and visualizations
├── models/                       # Trained model files
└── requirements.txt              # Dependencies
```

## Prerequisites
- Python 3.7+
- Required packages listed in requirements.txt

## Installation
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the application:
```
python DynamicDiscounting.py
```

### Available Operations:
1. **Run model evaluations** - Execute evaluation scripts for individual models
2. **Compare models** - Compare performance metrics between two models
3. **Exit** - Exit the application

## Adding New Models
1. Place model files in a subfolder under `models/`
2. Create an evaluation script in `evaluation_scripts/`
3. Follow the naming convention:
   - Script: `your_model_name.py`
   - Results: `model_evaluation_results/your_model_name/`

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Specify your license here]