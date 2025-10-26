import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- Page Config ---
st.set_page_config(
    page_title="Restaurant Segmentation",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        /* Main container styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        
        /* Title styling */
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        
        /* Card styling */
        .custom-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            margin-bottom: 2rem;
        }
        
        /* Form styling */
        .stForm {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid #667eea30;
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 0.75rem 2rem;
            border-radius: 10px;
            border: none;
            font-size: 1.1rem;
            transition: transform 0.2s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
        }
        
        .metric-label {
            font-size: 1rem;
            opacity: 0.9;
            margin-top: 0.5rem;
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Info box styling */
        .stAlert {
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #2c3e50;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load models ---
@st.cache_resource
def load_models():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("tsne_coordinates.pkl", "rb") as f:
        X_tsne = pickle.load(f)
    return kmeans, X_tsne

kmeans, X_tsne = load_models()

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
df['Cluster'] = kmeans.predict(X_scaled)

# --- Header ---
st.markdown('<h1 class="main-title">🍽️ Restaurant Segmentation Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover restaurant clusters using advanced machine learning with t-SNE visualization</p>', unsafe_allow_html=True)

# --- Metrics Row ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <p class="metric-value">{len(df)}</p>
            <p class="metric-label">Total Restaurants</p>
        </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <p class="metric-value">4</p>
            <p class="metric-label">Clusters Identified</p>
        </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <p class="metric-value">{len(features)}</p>
            <p class="metric-label">Features Analyzed</p>
        </div>
    """, unsafe_allow_html=True)

with col_m4:
    avg_price = df['Price_Num'].mean()
    st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <p class="metric-value">{avg_price:.1f}</p>
            <p class="metric-label">Average Price Level</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Main Layout ---
tab1, tab2, tab3 = st.tabs(["📊 Visualization", "🎯 Predict Cluster", "📈 Cluster Analysis"])

with tab1:
    st.markdown('<p class="section-header">t-SNE Projection of Restaurant Clusters</p>', unsafe_allow_html=True)
    
    col_viz1, col_viz2 = st.columns([2, 1])
    
    with col_viz1:
        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = sns.scatterplot(
            x=X_tsne[:, 0], 
            y=X_tsne[:, 1], 
            hue=df['Cluster'], 
            palette="Set2", 
            ax=ax,
            s=80,
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5
        )
        plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
        plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
        plt.title("Restaurant Segmentation using t-SNE", fontsize=13, fontweight='bold', pad=15)
        plt.legend(title='Cluster', title_fontsize=10, fontsize=9, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_viz2:
        st.markdown("### 🎨 Cluster Color Guide")
        cluster_info = {
            0: {"name": "Haut de gamme avec alcool", "color": "#8dd3c7", "icon": "🍷"},
            1: {"name": "Économiques sans alcool", "color": "#fb8072", "icon": "💰"},
            2: {"name": "Fumeurs tolérés", "color": "#80b1d3", "icon": "🚬"},
            3: {"name": "Milieu de gamme franchisé", "color": "#bebada", "icon": "🏪"}
        }
        
        for cluster_id, info in cluster_info.items():
            count = len(df[df['Cluster'] == cluster_id])
            st.markdown(f"""
                <div style="background: {info['color']}; padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; color: white;">
                    <strong>{info['icon']} Cluster {cluster_id}</strong><br>
                    <small>{info['name']}</small><br>
                    <strong>{count} restaurants</strong>
                </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<p class="section-header">Predict Restaurant Cluster</p>', unsafe_allow_html=True)
    
    col_pred1, col_pred2 = st.columns([1, 1])
    
    with col_pred1:
        with st.form("predict_form"):
            st.markdown("#### 📍 Location Details")
            col_lat, col_lon = st.columns(2)
            with col_lat:
                st.markdown("**Latitude**")
                lat = st.number_input("lat", value=23.7602683, format="%.7f", label_visibility="collapsed")
            with col_lon:
                st.markdown("**Longitude**")
                lon = st.number_input("lon", value=-99.1658646, format="%.7f", label_visibility="collapsed")
            
            st.markdown("#### 🍴 Restaurant Features")
            st.markdown("**Price Level** (1=Low, 2=Medium, 3=High)")
            price = st.slider("price", 1, 3, 2, label_visibility="collapsed")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**🍷 Alcohol Service**")
                alcohol = st.selectbox("alcohol", options=["No", "Wine & Beer", "Full Bar"], label_visibility="collapsed")
                st.markdown("**🚬 Smoking Policy**")
                smoking = st.selectbox("smoking", options=["No", "Smoking Section", "Bar Only", "Yes"], label_visibility="collapsed")
            with col_b:
                st.markdown("**🏪 Franchise**")
                franchise = st.selectbox("franchise", options=["No", "Yes"], label_visibility="collapsed")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("🔮 Predict Cluster")
    
    with col_pred2:
        if submit:
            alcohol_map = {'No': 0, 'Wine & Beer': 1, 'Full Bar': 2}
            smoking_map = {'No': 0, 'Smoking Section': 1, 'Bar Only': 2, 'Yes': 3}
            franchise_map = {'No': 0, 'Yes': 1}
            
            input_data = [[lat, lon, price, alcohol_map[alcohol], smoking_map[smoking], franchise_map[franchise]]]
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
            
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                    <h2 style="margin: 0;">Cluster {cluster_id}</h2>
                    <h3 style="margin-top: 0.5rem; font-weight: 400;">{label}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            cluster_data = df[df['Cluster'] == cluster_id]
            st.markdown("### 📊 Cluster Characteristics")
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Average Price", f"{cluster_data['Price_Num'].mean():.2f}")
                st.metric("Alcohol Service", f"{cluster_data['Alcohol_Num'].mean():.2f}")
            with metrics_col2:
                st.metric("Smoking Policy", f"{cluster_data['Smoking_Num'].mean():.2f}")
                st.metric("Franchise Rate", f"{cluster_data['Franchise_Num'].mean():.2%}")
        else:
            st.info("👈 Fill in the form and click 'Predict Cluster' to see results")

with tab3:
    st.markdown('<p class="section-header">Cluster Analysis & Statistics</p>', unsafe_allow_html=True)
    
    st.markdown("### 📋 Average Feature Values by Cluster")
    cluster_summary = df.groupby("Cluster")[['Price_Num', 'Alcohol_Num', 'Smoking_Num', 'Franchise_Num']].mean().round(2)
    
    # Style the dataframe
    st.dataframe(
        cluster_summary.style.background_gradient(cmap='Blues', axis=0).format(precision=2),
        use_container_width=True
    )
    
    st.markdown("### 📊 Cluster Distribution")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig, ax = plt.subplots(figsize=(8, 6))
        cluster_counts = df['Cluster'].value_counts().sort_index()
        colors = ['#8dd3c7', '#fb8072', '#80b1d3', '#bebada']
        bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Restaurants', fontsize=12, fontweight='bold')
        ax.set_title('Restaurant Count by Cluster', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    with col_chart2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = cluster_counts.values
        explode = (0.05, 0.05, 0.05, 0.05)
        ax.pie(sizes, explode=explode, labels=[f'Cluster {i}' for i in cluster_counts.index],
               colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Cluster Proportion', fontsize=14, fontweight='bold', pad=15)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
    
    if st.checkbox("🔍 Show detailed raw data"):
        st.markdown("### Raw Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)