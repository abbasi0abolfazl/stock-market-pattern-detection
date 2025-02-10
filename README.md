# Stock Market Pattern Detection using YOLOv8

This project is designed to detect stock market patterns (e.g., candlestick patterns) in financial time-series data using YOLOv8, a state-of-the-art object detection model. It processes historical stock or forex data, generates candlestick charts, and detects patterns using a pre-trained YOLOv8 model. Detected patterns are saved for further analysis.

## Features

- **Candlestick Chart Generation**: Converts time-series data into candlestick charts for visualization.
- **Time Gap Filtering**: Skips chart generation if there is a time gap greater than 10 minutes in the data.
- **Pattern Detection**: Uses YOLOv8 to detect stock market patterns in the generated charts.
- **Automatic Saving**: Saves charts and detected patterns into organized directories.
- **Customizable Parameters**: Allows users to define window sizes for chart generation and adjust detection confidence thresholds.

## Requirements

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The project expects a CSV file containing historical stock or forex data with the following columns:
- `time`: Timestamp of the data.
- `open`: Opening price.
- `high`: Highest price.
- `low`: Lowest price.
- `close`: Closing price.

Example dataset format:

| time                | open  | high  | low   | close |
|---------------------|-------|-------|-------|-------|
| 2023-10-01 12:00:00 | 1800  | 1810  | 1795  | 1805  |
| 2023-10-01 12:05:00 | 1805  | 1815  | 1800  | 1810  |

## How It Works

1. **Data Preprocessing**:
   - The script reads the CSV file and processes the data into a pandas DataFrame.
   - It converts the `time` column to datetime format and sets it as the index.

2. **Time Gap Filtering**:
   - The script checks for time gaps greater than 10 minutes in the data.
   - If a time gap is found, the corresponding chart is skipped.

3. **Candlestick Chart Generation**:
   - The script generates candlestick charts for specified window sizes (e.g., 2 hours, 4.1 hours, 6 hours).
   - Charts are saved in the `images` directory.

4. **Pattern Detection**:
   - The script uses a pre-trained YOLOv8 model (`foduucom/stockmarket-pattern-detection-yolov8`) to detect patterns in the generated charts.
   - Detected patterns are annotated and saved in the `detected_patterns` directory.

5. **Output**:
   - Charts with detected patterns are displayed using OpenCV and saved.
   - Logs are printed to indicate whether patterns were detected in each chart.

## Supported Patterns

- **Head and shoulders bottom**
- **Head and shoulders top**
- **M_Head**
- **StockLine**
- **Triangle**
- **W_Bottom**

## Usage

### Running Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/abbasi0abolfazl/stock-market-pattern-detection.git
   cd stock-market-pattern-detection
   ```

2. Place your dataset (CSV file) in the project directory.

3. Run the script or **pattern_detection.ipynb**:

   ```bash
   python main.py
   ```

4. Check the output:
   - Generated candlestick charts are saved in the `images` directory.
   - Detected patterns are saved in the `detected_patterns` directory.

### Running on Google Colab

You can also run this project directly on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LsecDYp6FCH9INvOnVf6FnW3R1wJpKMr?usp=sharing)

1. Open the Colab notebook using the link above.
2. Upload your dataset (CSV file) to the Colab environment.
3. Run all cells in the notebook.
4. Check the output:
   - Generated candlestick charts are saved in the `images` directory.
   - Detected patterns are saved in the `detected_patterns` directory.

## Customization

- **Window Sizes**: Modify the `window_sizes` list in the `main` function to change the time intervals for chart generation.
- **Time Gap Threshold**: Adjust the `max_time_gap` variable in the `generate_candlestick_charts` function to change the maximum allowed time gap.
- **Model Parameters**: Adjust the YOLOv8 model parameters (e.g., confidence threshold, IoU threshold) in the `main` function.
- **Input Data**: Update the `file_path` variable in the `load_and_preprocess_data` function to point to your dataset.

## Example Output

### Generated Candlestick Chart
![Candlestick Chart](./images/candlestick_chart_500_2024-12-26%2002_35_to_2024-12-26%2004_30.png)

### Detected Pattern
![Detected Pattern](./detected_patterns/detected_candlestick_chart_500_2024-12-27%2003_45_to_2024-12-27%2009_40.png)


## Directory Structure

```
stock-market-pattern-detection/
├── main.py                                     # Main script
├── pattern_detection.ipynb                     # ipython script
├── README.md                                   # Project documentation
├── XAUUSD_M5.csv                               # Example dataset
├── images/                                     # Generated candlestick charts
└── detected_patterns/                          # Saved detected patterns
```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.



