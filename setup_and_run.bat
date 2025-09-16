@echo off
echo Installing Python packages...
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

echo Running Iris EDA analysis...
python iris_eda.py

echo Analysis complete! Check the plots folder for generated images.
pause
