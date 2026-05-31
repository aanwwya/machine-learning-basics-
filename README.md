ML - insurance cost analysis 

This is my first ml practice project where i explore data analysis, feature selection, and preprocessing techniques using an insurance dataset.
This project explores how user attributes like age, bmi, smoking status, and region influence insurance charges through exploratory data analysis (EDA) and chi-square based feature selection



Files in this project: 

- ml.yt.ipynb → main notebook containing data analysis and experimentation  
- ml.py → clean python version of the notebook  
- insurance.csv → raw dataset used for analysis  
- final_insurance_project_df.csv → cleaned dataset after preprocessing  

Tools & Libraries:

- python 
- pandas 
- numpy  
- matplotlib  
- seaborn  
- scipy  
- scikit-learn  


project workflow:

- loaded and explored the dataset  
- performed data cleaning and preprocessing  
- conducted exploratory data analysis (EDA)  
- visualized relationships between key features  
- applied chi-square test for feature selection  
- converted notebook into a clean python script  

key insights:

- smoking status has a strong effect on insurance charges  
- age and bmi are important contributing factors  
- some categorical variables show statistically significant relationships with the target  

how to run:
install dependencies:

```bash id="z8v1n3"
pip install pandas numpy matplotlib seaborn scipy scikit-learn

