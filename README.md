### How to Run
1. **Prepare Input**:
   - Download the zip file from [this link](http://drive.google.com/file/d/1guxbAVRs1ylf16IMyicyoOFUjJa5u9yp/view) and place it in a directory (e.g., `C:\Users\omara\Downloads\01`).
   - zip should contain:
     - BCG files in a `BCG/` subdirectory (e.g., `01_20231105_BCG.csv`).
     - ECG (RR) files in a `Reference/RR/` subdirectory (e.g., `01_20231105_RR.csv`).
   - Filenames must include a date (e.g., `20231105`) for pairing.

2. **Set Directories**:
   - Update `zip_dir` in the script to your zip file directory.
   - Update `results_base_dir` to where you want results saved (e.g., `C:\Users\omara\Downloads\results`).

3. **Run the Script**:
   - Execute the script in a Python environment:
     ```bash
     python finalCodeToTasleem.py
     ```
   - The script will:
     - Extract zip files.
     - Process BCG files (add time vector, resample to 50 Hz).
     - Synchronize BCG and ECG data by time windows.
     - Analyze data and save results.
