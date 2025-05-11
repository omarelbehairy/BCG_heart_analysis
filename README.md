#  Ballistocar Diography and Respiratory Analysis Pipeline

This project processes BCG (Ballistocardiogram) and RR (R-R Interval) data to analyze heart rate and respiratory patterns. The pipeline includes data extraction, synchronization, and analysis, producing visualizations and metrics for each patient.



## How to Run

1. **Install Dependencies**:
   Install the required Python libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Code**:
   Execute the main script to process all patient data:
   ```bash
   python finalCodeToTasleem.py
   ```

3. **Output**:
   - Results are saved in the `results/` directory.
   - Each patient will have:
     - Text files with metrics (MAE, RMSE, Correlation).
     - Visualizations (correlation plots, Bland-Altman plots).

## Notes

- Ensure the `data/` directory is properly structured before running the script.
- The script automatically matches BCG and RR files based on the date in their filenames.
- If no matching files are found for a patient, they will be skipped.

## Troubleshooting

- **Missing Libraries**: Ensure all required libraries are installed.
- **File Format Issues**: Verify that the input files follow the expected format and naming convention.
- **No Results**: Check the `data/` directory for correctly named and structured files.

## Contact

For any issues or questions, please contact the project maintainer.
