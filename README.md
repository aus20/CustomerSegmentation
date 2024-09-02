Create project mkdir
Create env python -m venv env

Activate env source env/bin/activate

pip install pandas numpy matplotlib seaborn scikit-learn nltk tweepy flask

create a structre 
mkdir data/raw data/processed
mkdir notebooks
mkdir scripts
mkdir app
mkdir app/static app/templates
mkdir reports
mkdir tests
mkdir logs


env/: Holds virtual environment.
app/: Contains web application components.
data/: Stores raw and processed datasets.
logs/: Stores logs for debugging and monitoring.
notebooks/: Contains Jupyter notebooks for data exploration.
reports/: Stores generated reports.
scripts/: Contains Python scripts for data processing, analysis, etc.
tests/: Holds test cases.
.gitignore: Specifies files and directories to ignore in version control.
README.md: Provides project documentation.
requirements.txt: Lists dependencies for the project.

pip freeze > requirements.txt

gitignore is filled

SatisfactionScore  PredictedSatisfaction  Difference
0                0.8                  0.764       0.036
1                0.6                  0.593       0.007
2                0.3                  0.370      -0.070
3                0.5                  0.395       0.105
4                0.7                  0.691       0.009

New data new approach
