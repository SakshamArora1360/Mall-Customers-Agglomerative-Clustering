# Mall Customers Segmentation with Agglomerative Clustering

This project performs **customer segmentation** on mall customer data using **Agglomerative Hierarchical Clustering (Ward linkage)**.  
It includes dendrogram visualization, silhouette score evaluation, and final cluster visualization.

---

## Features
- Loads and preprocesses `Mall_Customers.csv` dataset.
- Standardizes features (`Annual Income`, `Spending Score`).
- Generates **dendrogram** to determine optimal cluster splits.
- Applies **Agglomerative Clustering (Ward linkage)** with Euclidean distance.
- Evaluates clustering with **Silhouette Score**.
- Visualizes resulting clusters.

---

## Technologies Used
- **Python** – Core programming language  
- **Pandas** – Data manipulation and preprocessing  
- **NumPy** – Numerical computations  
- **Matplotlib** – Data visualization  
- **Scikit-learn** – Machine learning (StandardScaler, Clustering, Metrics)  
- **SciPy** – Hierarchical clustering & dendrograms

---

## Project Structure
Mall-Customers-Agglomerative-Clustering/
- ── Mall_Customers.csv # Dataset 
- ── mall_customers_clustering.py # Main Python script
- ── README.md # Project documentation

---

## Output Example
- **Dendrogram**
- <img width="561" height="335" alt="image" src="https://github.com/user-attachments/assets/e7e64be2-c50f-4480-9ce9-cad7fac649f9" />
- **Agglomertive Clustering with Silhouette Score**
- <img width="726" height="549" alt="image" src="https://github.com/user-attachments/assets/2af33455-de4a-40b7-9e15-e84ce763327e" />

---

## Results
- Dendrogram illustrates hierarchical merging of customers.
- Silhouette Score helps evaluate cluster quality.
- Final clusters displayed in a 2D scatter plot.

---

## Use Case
This project demonstrates:
- Customer segmentation for marketing and business insights.
- Application of Hierarchical Clustering on real-world datasets.
- Practical evaluation of unsupervised learning methods.
