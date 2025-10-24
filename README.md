#  Restaurant Segmentation

This project applies unsupervised machine learning techniques to segment restaurants based on their geographic and business characteristics.

##  Goal

Identify homogeneous groups of restaurants to support strategic and marketing decision-making.

##  Files

- `segmentation_res.py`: Full pipeline including preprocessing, clustering, visualization, and analysis.
- `restaurants.csv`: Source dataset used for clustering.
   [Download the dataset](./restaurants.csv)


##  Dataset Description

The dataset includes the following attributes for each restaurant:

- `Latitude`, `Longitude`: Geographical location
- `Price_Num`: Numerical encoding of price level
- `Alcohol_Service`: Type of alcohol served (No, Wine & Beer, Full Bar)
- `Smoking_Allowed`: Smoking policy (No, Smoking Section, Yes, Bar Only)
- `Franchise`: Whether the restaurant is a franchise

**Note**: Categorical columns are preprocessed into numerical values for modeling.

##  Technologies

- Python 3.x
- pandas, numpy
- scikit-learn (PCA, KMeans, t-SNE)
- seaborn, matplotlib
- scipy

##  Pipeline Overview

1. Data cleaning and categorical variable encoding
2. Correlation analysis
3. Feature scaling with `StandardScaler`
4. Dimensionality reduction with PCA
5. Clustering:
   - **Elbow method**
   - **Silhouette score**
   - **Hierarchical Clustering (Ward linkage)**
6. 2D visualizations using PCA and t-SNE
7. Manual labeling of clusters with business-oriented names

##  Results

- **4 main clusters** identified:
  - `High-end with alcohol`
  - `Low-cost without alcohol`
  - `Smoking-friendly venues`
  - `Mid-range franchised`

##  Included Visuals

- Correlation heatmap
- PCA projections (2D and 3D)
- Hierarchical dendrogram
- PCA variable correlation circle
- t-SNE projection with cluster labels

##  Future Improvements

- Cross-segmentation with customer preferences
- Interactive geo-segmentation
- Cluster-based recommendation systems
