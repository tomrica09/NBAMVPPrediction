## Predicting the NBA MVP using an Optimized XGBoost Classifier

### Project Summary

This project focuses on predicting the NBA Most Valuable Player (MVP) using a machine learning model trained on historical player performance data from the 1990-2024 NBA seasons. Given the highly imbalanced nature of MVP classification, where only one player per season is awarded the MVP among hundreds, this project leverages advanced techniques to address the imbalance and improve prediction accuracy. The model pipeline includes data preprocessing, oversampling with SVM-SMOTE, undersampling with EasyEnsemble, hyperparameter optimization using Bayesian Optimization, and feature selection using a Genetic Algorithm. An XGBoost Classifier serves as the core algorithm due to its robustness and interpretability. The project also evaluates model performance using AUC and G-Mean for handling imbalanced dataset effectiveness. In addition, it also evaluates model performance using Accuracy, Precision, Recall, F1-Score, and AUC-ROC metrics for its effectiveness in predicting the NBA MVP. The final trained model is used to predict the NBA MVP for the 2024-2025 season to check whether its predictive effectivity is accurate.
***
### Data Sources

The dataset used in this project was obtained through web scraping from **basketball-reference.com**, a reliable source of comprehensive NBA statistics. The scraping process was automated and documented in the *we_scraping.ipynb* notebook file, which collects player statistics and MVP voting results from the 1990-1991 season up to the 2024-2025 season. The data includes various player performance metrics such as points, assists, rebounds, and advanced stats like efficiency ratings. This raw data was then cleaned, processed, and transformed into a structured format suitable for machine learning classification target variable, distinguishing MVPs from non-MVPs across multiple seasons. 
***
### File Description

#### Folders
- *mvp*: Contains HTML file format data of MVP Voting Table from 1990-1991 season up to 2024-2025 season.
- *player*: Contains HTML file format data of Player Stats Table from 1990-1991 season up to 2024-2025 season.
- *team*: Contains HTML file format data of Team Standings Table from 1990-1991 season up to 2024-2025 season.

#### Jupyter Notebooks (ipynb)
- *web_scraping*: Automates the collection of NBA player statistics and MVP voting data from **basketball-reference.com**.
- *data_cleaning*: Preprocesses the scraped NBA data by handling null values and other unnecessary values present in data, and merging player and MVP data.
- *optimized_xgboost_classifier*: Trains and fine-tunes an XGBoost Classifier using Bayesian Optimization and Genetic Algorithm for feature selection to effectively predict the NBA MVP.
- *[2024-2025]_scrape_data*: Collects and compiles NBA player statistics for the 2024-2025 season through web scraping for use in MVP prediction.
- *[2024-2025]_clean_data*: Preprocessed the scraped NBA 2024-2025 season data by cleaning it merging player and team statistics for model prediction.
- *[2024-2025]_MVP_prediction*: Uses the optimized XGBoost Classifier model to predict the NBA MVP for the 2024-2025 season based on the cleaned and preprocessed player data.

#### CSVs
- *mvps*: Contains historical NBA MVP voting results, including player names, teams, and MVP vote shares, used as ground truth labels for training and evaluating the prediction model.
- *players*: Contains the names of NBA players and their statistics, used to retain player identity and details during data processing and MVP prediction. 
- *teams*: Contains official NBA team names and team information and statistics. 
- *nicknames*: Contains the official NBA team abbreviations to map its team name to the merged **player_mvp_stats**.
- *player_mvp_stats*: Contains the compiled season-by-season statistical performance of NBA players along with MVP voting results used as the primary dataset for model training and evaluation.
- *[2024-2025]_players*: Contains the cleaned and structured statistical data of NBA players for the 2024-2025 season, used as input for MVP prediction.
- *[2024-2025]_teams*: Contains team information and statistics for the 2024-2025 NBA season to support player-team associations in the dataset.
- *[2024-2025]_player_team_stats*: Contains comprehensive performance statistics for NBA players with team information and statistics during the 2024-2025 season, used for MVP prediction and model evaluation. 
***
### Model Evaluation Results

#### Evaluating the model based on handling class imbalance datasets 
- **AUC**: 0.9909   
- **G-Mean**: 0.9221

The evaluation of the optimized XGBoost Classifier model in handling class imbalance datasets yielded strong results, with an **AUC (Area Under the Curve) of 0.9909** and a **G-Mean (Geometric Mean) of 0.9221**. These metrics indicate the model's excellent discriminative ability in distinguishing between MVPs and non-MVPs, even with a highly imbalanced dataset. A high AUC reflects the model's capability to rank positive examples (MVPs) ahead of negative ones (non-MVPs) across thresholds, while the strong G-Mean suggests balanced sensitivity (recall) and specificity, avoiding bias toward the majority class. This demonstrates the effectiveness of the applied data balancing techniques (SVM-SMOTE for oversampling and EasyEnsemble for undersampling) combined with the optimized classifier in addressing the class imbalance problem inherent in NBA MVP prediction.

#### Evaluating the model based on predicting the NBA MVP 
- **Accuracy**: 0.9825
- **F1-Score**: 0.7594
- **Precision**: 0.6788
- **Recall**: 0.8618
- **AUC-ROC**: 0.9242

The evaluation of the optimized XGBoost Classifier model for predicting the NBA MVP produced highly promising results, achieving an **Accuracy of 0.9825, an F1-Score of 0.7594, a Precision of 0.6788, a Recall of 0.8618, and an AUC-ROC of 0.9242**. These metrics collectively highlight the model's strong predictive capability. The high accuracy indicates that the model correctly classifies a large majority of players. The elevated recall suggests it is particularly effective in identifying actual MVPs, while the F1-score balances this recall with precision, ensuring fewer false positives. The precision, though slightly lower than recall, remains strong, reflecting reasonable reliability in MVP predictions. The AUC-ROC score further reinforces the model's ability to distinguish between MVPs and non-MVPs across classification thresholds. These results underscore the model's effectiveness in supporting MVP predictions in the context of sports analytics, particularly in the NBA. 

#### Predicting the NBA MVP 2024-2025 season 

![2025 NBA MVP Predictions](https://github.com/user-attachments/assets/9a627449-00d4-42c1-94c2-9c7c0f86c406)

The optimized XGBoost Classifier model demonstrated outstanding predictive accuracy in forecasting the NBA MVP for the 2024-2025 season. The model successfully ranked **Shai Gilgeous-Alexander (Oklahoma City Thunder)** as the season's Most Valuable Player, followed by **Nikola Jokic (Denver Nuggets)** in second place and **Giannis Antetokounmpo (Milwaukee Bucks)** in third, exactly matching the official results of the 2024-2025 NBA MVP voting. This outcome highlights the model's ability to not only identify MVP-caliber players but also accurately reflect real-world voting trends.

![Feature Importance](https://github.com/user-attachments/assets/da7e5d29-655d-43e4-8a95-57396e711785)

