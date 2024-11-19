# **Customer Segmentation Using K-Means Clustering**

This project demonstrates customer segmentation using the **K-Means Clustering algorithm** on a sample dataset. The goal is to group customers into clusters based on attributes like **Age**, **Annual Income**, and **Spending Score**, helping businesses understand customer behavior and make data-driven decisions.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Dependencies](#dependencies)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## **Project Overview**

Customer segmentation is a vital marketing strategy that involves dividing customers into groups based on their characteristics or behavior. This project implements K-Means Clustering for segmentation. K-Means groups customers with similar traits into clusters, helping businesses tailor their marketing strategies.

---

## **Dataset**

### Sample Data:
| CustomerID | Gender | Age | Annual Income (k$) | Spending Score (1-100) |
|------------|--------|-----|--------------------|-------------------------|
| 1          | Male   | 19  | 15                 | 39                      |
| 2          | Male   | 21  | 15                 | 81                      |
| 3          | Female | 20  | 16                 | 6                       |
| 4          | Female | 23  | 16                 | 77                      |

- **Columns:**
  - **CustomerID**: Unique ID for each customer.
  - **Gender**: Gender of the customer (Male/Female).
  - **Age**: Age of the customer.
  - **Annual Income (k$)**: Yearly income of the customer in thousands of dollars.
  - **Spending Score (1-100)**: Customer's spending habit score.

- **Dataset Source**: A simulated dataset created for this project.

---

## **Features**
1. **Preprocessing**: Cleaning and transforming the dataset for clustering.
2. **Visualization**: Visualizing customer segments using 2D and 3D scatter plots.
3. **Clustering**: Using K-Means to identify customer segments.
4. **Insights**: Extracting actionable insights based on clustering results.

---

## **Dependencies**

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

Install all dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## **Project Workflow**

1. **Load Data**: Import the dataset into a Pandas DataFrame.
2. **Preprocessing**: Clean data, select relevant features, and normalize it.
3. **K-Means Clustering**:
   - Use the **Elbow Method** to determine the optimal number of clusters.
   - Apply K-Means to segment the customers.
4. **Visualization**:
   - 2D scatter plot to visualize clusters based on two features.
   - 3D scatter plot for better understanding of multi-dimensional clusters.
5. **Insights**:
   - Analyze each cluster to interpret customer behavior.

---

## **Results**

- **Cluster Visualizations**:
  - Customers are grouped into distinct clusters based on their age, annual income, and spending score.
  - Visualizations help identify patterns, such as high-income customers with low spending scores or young customers with high spending scores.

---

## **How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-kmeans.git
   cd customer-segmentation-kmeans
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook customer_segmentation.ipynb
   ```

4. Run the cells step-by-step to preprocess the data, train the model, and visualize the clusters.

---

## **Screenshots**

### 1. **Elbow Method to Determine Optimal Clusters**
![Elbow Method](https://via.placeholder.com/800x400?text=Elbow+Method+Chart)

### 2. **2D Cluster Visualization**
![2D Clusters](https://via.placeholder.com/800x400?text=2D+Cluster+Visualization)

### 3. **3D Cluster Visualization**
![3D Clusters](https://via.placeholder.com/800x400?text=3D+Cluster+Visualization)

---

## **Future Enhancements**

- Include **demographic data** such as location or occupation to improve clustering.
- Explore other clustering algorithms like **DBSCAN** or **Hierarchical Clustering**.
- Develop an interactive dashboard to display customer segments.

---

## **License**

This project is licensed under the [MIT License](LICENSE).
