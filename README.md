# Movie Recommender System

A movie recommendation engine built with **Apache Spark ML** that learns user–item preferences from the **MovieLens 1M** dataset and generates personalized top‑N recommendations using **ALS (Alternating Least Squares)** collaborative filtering.

## Objective
Build a scalable recommender system that predicts user ratings and surfaces movies a user is likely to enjoy, using distributed model training with Spark.

## Dataset
- **Source:** MovieLens 1M (GroupLens)
- **Scale:** ~1,000,000 ratings from ~6,000 users across ~4,000 movies
- **Notes:** You can download the dataset from the GroupLens website and place it in a local `data/` folder.

## Tech Stack
- Apache Spark (MLlib)
- PySpark
- Python
- ALS, RegressionEvaluator

## Approach
This project focuses on **model‑based collaborative filtering** using **ALS matrix factorization**. ALS learns latent vectors for users and items by minimizing prediction error, enabling scalable recommendations on large datasets.

### Recommendation Strategies (Overview)
- **Content‑based filtering:** recommends items similar to what a user already liked, based on item features.
- **Collaborative filtering:** recommends items based on patterns learned from many users (this project).

## Workflow
1. Load ratings data into Spark
2. Clean and prepare user‑item interactions
3. Train ALS model
4. Evaluate with RMSE
5. Generate top‑N recommendations per user

## Results
- **RMSE:** _add your final score here_
- **Top‑N recommendations:** generated for each user using the trained ALS model

## How to Run
> Update these steps to match your environment.

1. Install Java (8 or 11) and Apache Spark 3.x
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install pyspark
   ```
3. Download MovieLens 1M and place files in `data/`
4. Run your notebook or script:
   ```bash
   jupyter notebook
   ```

## Project Structure (Recommended)
```
.
├── data/                  # MovieLens dataset files
├── notebooks/             # Jupyter notebooks
├── src/                   # Spark/PySpark pipeline code
├── outputs/               # Evaluation reports / recommendation samples
└── README.md
```

## Future Improvements
- Add hyperparameter tuning for ALS (rank, regParam, iterations)
- Add implicit feedback support
- Deploy a lightweight demo app (Streamlit/Flask)

## Acknowledgements
- MovieLens dataset by GroupLens Research

## License
This project is licensed under the MIT License. See the LICENSE file for details.
