import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# --- Load models ---
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)


with open("tsne_coordinates.pkl", "rb") as f:
    X_tsne = pickle.load(f)

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv("restaurants.csv", delimiter=';')
    df['Alcohol_Service'] = df['Alcohol_Service'].fillna('No').str.strip()
    df['Smoking_Allowed'] = df['Smoking_Allowed'].str.strip()
    df['Franchise'] = df['Franchise'].fillna('No')
    df['Alcohol_Num'] = df['Alcohol_Service'].map({'No': 0, 'Wine & Beer': 1, 'Full Bar': 2})
    df['Smoking_Num'] = df['Smoking_Allowed'].map({'No': 0, 'Smoking Section': 1, 'Bar Only': 2, 'Yes': 3})
    df['Franchise_Num'] = df['Franchise'].map({'No': 0, 'Yes': 1})
    return df

df = load_data()
features = ['Latitude', 'Longitude', 'Price_Num', 'Alcohol_Num', 'Smoking_Num', 'Franchise_Num']
X = df[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

# --- Predict and project ---

df['Cluster'] = kmeans.predict(X_scaled)

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Streamlit UI ---


col1, col2 = st.columns([1.3, 1])
with col1:
    st.title("🍽️ 🍽️ Restaurant Segmentation (t-SNE)")
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Cluster'], palette="Set2", ax=ax)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Projection of Clusters")
    st.pyplot(fig)

    st.write("### Cluster Summary")
    st.dataframe(df.groupby("Cluster")[['Price_Num', 'Alcohol_Num', 'Smoking_Num', 'Franchise_Num']].mean().round(2))

    if st.checkbox("Show raw data"):
        st.dataframe(df.head())

with col2:
    st.subheader("🎯 Predict a Restaurant's Cluster")

    with st.form("predict_form"):
        lat = st.number_input("Latitude", value=23.7602683)
        lon = st.number_input("Longitude", value=-99.1658646)
        price = st.slider("Price_Num", 1, 2, 3)
        alcohol = st.selectbox("Alcohol Service", options=["No", "Wine & Beer", "Full Bar"])
        smoking = st.selectbox("Smoking Allowed", options=["No", "Smoking Section", "Bar Only", "Yes"])
        franchise = st.selectbox("Franchise", options=["No", "Yes"])
        
        submit = st.form_submit_button("Predict")

    if submit:
        alcohol_map = {'No': 0, 'Wine & Beer': 1, 'Full Bar': 2}
        smoking_map = {'No': 0, 'Smoking Section': 1, 'Bar Only': 2, 'Yes': 3}
        franchise_map = {'No': 0, 'Yes': 1}

        input_data = [[
            lat,
            lon,
            price,
            alcohol_map[alcohol],
            smoking_map[smoking],
            franchise_map[franchise]
        ]]

        # Réutiliser le scaler entraîné sur X
        scaler = StandardScaler().fit(X)
        input_point = scaler.transform(input_data)

        cluster_id = kmeans.predict(input_point)[0]

        cluster_labels = {
            0: "Haut de gamme avec alcool",
            1: "Économiques sans alcool",
            2: "Fumeurs tolérés",
            3: "Milieu de gamme franchisé"
        }

        label = cluster_labels.get(cluster_id, "Inconnu")

        st.success(f"🚀 Predicted Cluster: {cluster_id} ({label})")

        cluster_data = df[df['Cluster'] == cluster_id]
        st.info(f"Cluster Characteristics:\n"
                f"- Average Price_Num: {cluster_data['Price_Num'].mean():.2f}\n"
                f"- Average Alcohol_Num: {cluster_data['Alcohol_Num'].mean():.2f}\n"
                f"- Average Smoking_Num: {cluster_data['Smoking_Num'].mean():.2f}\n"
                f"- Average Franchise_Num: {cluster_data['Franchise_Num'].mean():.2f}")
