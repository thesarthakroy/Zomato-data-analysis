import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------
# PAGE CONFIGURATION & THEME STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Zomato Analytics Dashboard",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for a minimalist, modern typography-driven UI
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1E293B;
        background-color: #FFFFFF;
    }
    
    /* Clean Minimalist Typography Header */
    .dashboard-header {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 1.25rem;
        margin-bottom: 2.25rem;
        margin-top: -1.5rem;
    }
    .main-title {
        color: #E23744;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.15rem;
        letter-spacing: -0.03em;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
    }
    
    /* Clean Minimalist KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.01), 0 1px 2px rgba(0,0,0,0.03);
        margin-bottom: 0.75rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.02), 0 2px 4px rgba(0,0,0,0.04);
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Highlight Cards for Insights */
    .executive-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #E23744;
        margin-bottom: 1.1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .executive-card-title {
        font-weight: 700;
        color: #E23744;
        font-size: 1rem;
        margin-bottom: 0.35rem;
        font-family: 'Outfit', sans-serif;
    }
    .executive-card-desc {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    
    /* ML Prediction Output Styling */
    .result-box {
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        text-align: center;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .result-box-success {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        color: #14532D;
    }
    .result-box-fail {
        background-color: #FEF2F2;
        border-left: 5px solid #DC2626;
        color: #7F1D1D;
    }
    .result-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom footer styles */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #94A3B8;
        font-size: 0.8rem;
        border-top: 1px solid #F1F5F9;
        margin-top: 4rem;
        letter-spacing: 0.02em;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DATA LOADING & CACHING
# -------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("Zomato-data-.csv")
    
    # Process ratings (e.g. "4.1/5" to float)
    def clean_rating(val):
        try:
            val_str = str(val).strip()
            if '/' in val_str:
                return float(val_str.split('/')[0])
            return float(val_str)
        except Exception:
            return np.nan
            
    df['rate_clean'] = df['rate'].apply(clean_rating)
    df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')
    
    return df

try:
    df_raw = load_and_preprocess_data()
except Exception as e:
    st.error(f"Error loading Zomato-data-.csv: {e}")
    st.stop()

# -------------------------------------------------------------
# MACHINE LEARNING CLASSIFIER TRAINING (LOGISTIC REGRESSION)
# -------------------------------------------------------------
@st.cache_resource
def train_popularity_classifier(df):
    # Process target variable: 1 if votes > median votes (Highly Popular), 0 otherwise
    median_votes = df['votes'].median()
    df_ml = df.copy()
    df_ml['is_popular'] = (df_ml['votes'] > median_votes).astype(int)
    
    # Drop rows with NaN in rate_clean or approx_cost
    df_ml = df_ml.dropna(subset=['rate_clean', 'approx_cost(for two people)'])
    
    # Features & Target
    features = ['online_order', 'book_table', 'rate_clean', 'approx_cost(for two people)', 'listed_in(type)']
    X = df_ml[features]
    y = df_ml['is_popular']
    
    # Train-test split for metrics (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Set up preprocessing pipeline
    categorical_features = ['online_order', 'book_table', 'listed_in(type)']
    numeric_features = ['rate_clean', 'approx_cost(for two people)']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create Pipeline with Logistic Regression for continuous and monotonic predictions
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Train model on splits for validation accuracy metric
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    
    # Retrain on full pipeline for final production model & coefficients
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    full_pipeline.fit(X, y)
    
    # Extract model coefficients to show feature impacts
    coefs = full_pipeline.named_steps['classifier'].coef_[0]
    cat_encoder = full_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    feature_names = numeric_features + list(cat_encoder.get_feature_names_out(categorical_features))
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs
    })
    
    # Format feature names for clean presentation
    name_map = {
        'rate_clean': 'Review Rating',
        'approx_cost(for two people)': 'Cost for Two',
        'online_order_Yes': 'Online Order: Yes',
        'online_order_No': 'Online Order: No',
        'book_table_Yes': 'Table Booking: Yes',
        'book_table_No': 'Table Booking: No',
        'listed_in(type)_Buffet': 'Type: Buffet',
        'listed_in(type)_Cafes': 'Type: Cafes',
        'listed_in(type)_Dining': 'Type: Dining',
        'listed_in(type)_other': 'Type: Other'
    }
    importance_df['Feature'] = importance_df['Feature'].map(lambda x: name_map.get(x, x))
    importance_df = importance_df.sort_values(by='Coefficient', ascending=True)
    
    return full_pipeline, accuracy, importance_df, median_votes

try:
    model_pipeline, model_accuracy, model_importance_df, median_votes_threshold = train_popularity_classifier(df_raw)
except Exception as e:
    st.error(f"Error training Machine Learning Classifier: {e}")

# -------------------------------------------------------------
# SIDEBAR / FILTERING PANEL
# -------------------------------------------------------------
st.sidebar.markdown(
    "<div style='text-align: center; padding-bottom: 10px;'>"
    "<h2 style='color: #E23744; margin-bottom: 0; font-family: \"Outfit\", sans-serif;'>Filters</h2>"
    "<p style='color: #64748B; font-size: 0.85rem;'>Refine Dashboard Data</p>"
    "</div>", 
    unsafe_allow_html=True
)

# Restaurant Type Filter
type_options = list(df_raw['listed_in(type)'].unique())
selected_types = st.sidebar.multiselect(
    "Restaurant Type", 
    options=type_options, 
    default=type_options,
    help="Select categories of restaurants to view."
)

# Online Order Option Filter
online_options = list(df_raw['online_order'].unique())
selected_online = st.sidebar.multiselect(
    "Offers Online Ordering",
    options=online_options,
    default=online_options
)

# Table Booking Filter
booking_options = list(df_raw['book_table'].unique())
selected_booking = st.sidebar.multiselect(
    "Offers Table Booking",
    options=booking_options,
    default=booking_options
)

# Approx Cost slider
min_cost = int(df_raw['approx_cost(for two people)'].min())
max_cost = int(df_raw['approx_cost(for two people)'].max())
selected_cost_range = st.sidebar.slider(
    "Approx Cost for Two (Rs.)",
    min_value=min_cost,
    max_value=max_cost,
    value=(min_cost, max_cost),
    step=50
)

# Rating slider
min_rate = float(df_raw['rate_clean'].min())
max_rate = float(df_raw['rate_clean'].max())
selected_rating_range = st.sidebar.slider(
    "Rating Range",
    min_value=min_rate,
    max_value=max_rate,
    value=(min_rate, max_rate),
    step=0.1
)

# Apply global sidebar filters
df_filtered = df_raw.copy()
df_filtered = df_filtered[
    (df_filtered['listed_in(type)'].isin(selected_types)) &
    (df_filtered['online_order'].isin(selected_online)) &
    (df_filtered['book_table'].isin(selected_booking)) &
    (df_filtered['approx_cost(for two people)'].between(selected_cost_range[0], selected_cost_range[1])) &
    (df_filtered['rate_clean'].between(selected_rating_range[0], selected_rating_range[1]))
]

# -------------------------------------------------------------
# MAIN HEADER SECTION
# -------------------------------------------------------------
st.markdown(
    "<div class='dashboard-header'>"
    "<h1 class='main-title'>Zomato Data Analytics</h1>"
    "<p class='subtitle'>Interactive Business Intelligence & Market Predictions Dashboard</p>"
    "</div>", 
    unsafe_allow_html=True
)

if df_filtered.empty:
    st.warning("⚠️ No restaurants match the selected filters. Please adjust the parameters in the left sidebar.")
    st.stop()

# -------------------------------------------------------------
# APP NAVIGATION TABS
# -------------------------------------------------------------
tab_overview, tab_visualizations, tab_explorer, tab_ml_predictor, tab_takeaways = st.tabs([
    "📊 Market Overview",
    "📈 Interactive Analytics",
    "🔎 Restaurant Finder",
    "🤖 Popularity Predictor (ML)",
    "💡 Strategic Takeaways"
])

# -------------------------------------------------------------
# TAB 1: MARKET OVERVIEW
# -------------------------------------------------------------
with tab_overview:
    # Custom Styled KPI Columns
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Restaurants</div>
            <div class="kpi-value">{df_filtered['name'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        avg_rate = df_filtered['rate_clean'].mean()
        val_rate = f"{avg_rate:.2f} / 5.0" if not np.isnan(avg_rate) else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Rating</div>
            <div class="kpi-value">{val_rate}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        total_votes = df_filtered['votes'].sum()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Votes</div>
            <div class="kpi-value">{total_votes:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        avg_cost = df_filtered['approx_cost(for two people)'].mean()
        val_cost = f"Rs. {avg_cost:.0f}" if not np.isnan(avg_cost) else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Cost for Two</div>
            <div class="kpi-value">{val_cost}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Executive Highlights split
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("Key Performance Highlights")
        
        # Max Votes Card
        max_votes_idx = df_filtered['votes'].idxmax()
        max_voted_rest = df_filtered.loc[max_votes_idx]
        
        # Max Rating Card
        max_rate_val = df_filtered['rate_clean'].max()
        max_rated_rest = df_filtered[df_filtered['rate_clean'] == max_rate_val].iloc[0]
        
        st.markdown(f"""
        <div class="executive-card">
            <div class="executive-card-title">🔥 Most Popular Restaurant (Highest Customer Votes)</div>
            <div class="executive-card-desc">
                <strong>{max_voted_rest['name']}</strong> has received the maximum customer engagement with 
                <strong>{max_voted_rest['votes']:,}</strong> votes. It is categorized under 
                <em>{max_voted_rest['listed_in(type)']}</em> and has a rating of <strong>{max_voted_rest['rate']}</strong>, 
                costing approx. <strong>Rs. {max_voted_rest['approx_cost(for two people)']}</strong> for two people.
            </div>
        </div>
        
        <div class="executive-card">
            <div class="executive-card-title">⭐ Top Rated Dining Experience</div>
            <div class="executive-card-desc">
                <strong>{max_rated_rest['name']}</strong> has the highest rating in this selection at 
                <strong>{max_rated_rest['rate']}</strong>. It categorizes as <em>{max_rated_rest['listed_in(type)']}</em>, 
                receives <strong>{max_rated_rest['votes']:,}</strong> votes, and costs approx. 
                <strong>Rs. {max_rated_rest['approx_cost(for two people)']}</strong> for two people.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        st.subheader("Operational Statistics")
        
        summary_stats = {
            "Operational Detail": [
                "Online Ordering Available",
                "Table Booking Offered",
                "Average Votes per Restaurant",
                "Affordable Outlets (<= Rs. 400)",
                "Premium Outlets (> Rs. 600)"
            ],
            "Value": [
                f"{len(df_filtered[df_filtered['online_order'] == 'Yes'])} ({len(df_filtered[df_filtered['online_order'] == 'Yes'])/len(df_filtered)*100:.1f}%)",
                f"{len(df_filtered[df_filtered['book_table'] == 'Yes'])} ({len(df_filtered[df_filtered['book_table'] == 'Yes'])/len(df_filtered)*100:.1f}%)",
                f"{df_filtered['votes'].mean():.1f} votes",
                f"{len(df_filtered[df_filtered['approx_cost(for two people)'] <= 400])} restaurants",
                f"{len(df_filtered[df_filtered['approx_cost(for two people)'] > 600])} restaurants"
            ]
        }
        st.table(pd.DataFrame(summary_stats))

# -------------------------------------------------------------
# TAB 2: INTERACTIVE ANALYTICS
# -------------------------------------------------------------
with tab_visualizations:
    st.subheader("Interactive Visualizations & Distributions")
    
    # 2x2 Chart Grid
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # Chart 1: Restaurant Count by Type
        type_counts = df_filtered['listed_in(type)'].value_counts().reset_index()
        type_counts.columns = ['Type of Restaurant', 'Count']
        
        fig1 = px.bar(
            type_counts,
            x='Type of Restaurant',
            y='Count',
            title='Distribution of Restaurant Types',
            labels={'Count': 'Number of Restaurants'},
            color='Type of Restaurant',
            color_discrete_sequence=['#E23744', '#F8FAFC', '#94A3B8', '#475569']
        )
        fig1.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig1, width='stretch')
        st.markdown("<p style='font-size: 0.85rem; color: #64748B; font-style: italic; margin-top: -10px;'>Insight: Illustrates which restaurant categories are most prominent in the market sample.</p>", unsafe_allow_html=True)
        
    with row1_col2:
        # Chart 2: Total Votes by Restaurant Type
        grouped_votes = df_filtered.groupby('listed_in(type)')['votes'].sum().reset_index()
        grouped_votes.columns = ['Type of Restaurant', 'Votes']
        grouped_votes = grouped_votes.sort_values(by='Votes', ascending=False)
        
        fig2 = px.line(
            grouped_votes,
            x='Type of Restaurant',
            y='Votes',
            title='Customer Engagement (Total Votes) by Restaurant Type',
            markers=True,
            line_shape='linear',
            labels={'Votes': 'Total Votes'}
        )
        fig2.update_traces(line=dict(color='#E23744', width=3), marker=dict(size=10))
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig2, width='stretch')
        st.markdown("<p style='font-size: 0.85rem; color: #64748B; font-style: italic; margin-top: -10px;'>Insight: Displays voter participation across segments. Higher votes imply higher public interaction.</p>", unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: #F1F5F9;'>", unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # Chart 3: Ratings Distribution
        fig3 = px.histogram(
            df_filtered,
            x='rate_clean',
            nbins=10,
            title='Ratings Distribution Profile',
            labels={'rate_clean': 'Restaurant Rating (out of 5)'},
            color_discrete_sequence=['#E23744']
        )
        fig3.update_layout(bargap=0.05, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig3, width='stretch')
        st.markdown("<p style='font-size: 0.85rem; color: #64748B; font-style: italic; margin-top: -10px;'>Insight: Shows distribution of customer review scores. Helps identify general quality benchmarks.</p>", unsafe_allow_html=True)
        
    with row2_col2:
        # Chart 4: Cost for Two Distribution
        cost_counts = df_filtered['approx_cost(for two people)'].value_counts().reset_index()
        cost_counts.columns = ['Approx Cost', 'Count']
        cost_counts = cost_counts.sort_values(by='Approx Cost')
        
        fig4 = px.bar(
            cost_counts,
            x='Approx Cost',
            y='Count',
            title='Price Distribution (Approx. Cost for Two People)',
            labels={'Count': 'Number of Restaurants', 'Approx Cost': 'Cost for Two (Rs.)'},
            color_discrete_sequence=['#334155']
        )
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig4, width='stretch')
        st.markdown("<p style='font-size: 0.85rem; color: #64748B; font-style: italic; margin-top: -10px;'>Insight: Breakdown of restaurant prices. Most options lie in the budget to mid-range categories.</p>", unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: #F1F5F9;'>", unsafe_allow_html=True)
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        # Chart 5: Online Order Option Impact on Ratings
        fig5 = px.box(
            df_filtered,
            x='online_order',
            y='rate_clean',
            color='online_order',
            title='Rating Spread: Online Ordering Impact',
            labels={'online_order': 'Offers Online Ordering', 'rate_clean': 'Rating (out of 5)'},
            color_discrete_map={'Yes': '#E23744', 'No': '#94A3B8'}
        )
        fig5.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig5, width='stretch')
        st.markdown("<p style='font-size: 0.85rem; color: #64748B; font-style: italic; margin-top: -10px;'>Insight: Reviews box plots for online delivery availability. Identifies differences in customer satisfaction.</p>", unsafe_allow_html=True)
        
    with row3_col2:
        # Chart 6: Heatmap - Restaurant Type vs. Online Order availability
        pivot_table = df_filtered.pivot_table(
            index='listed_in(type)', 
            columns='online_order', 
            aggfunc='size', 
            fill_value=0
        )
        
        x_labels = list(pivot_table.columns)
        y_labels = list(pivot_table.index)
        z_values = pivot_table.values
        
        fig6 = px.imshow(
            z_values,
            x=x_labels,
            y=y_labels,
            aspect="auto",
            color_continuous_scale='Reds',
            title='Operational Heatmap: Restaurant Type vs. Online Ordering',
            labels=dict(x="Online Ordering", y="Listed In Type", color="Restaurant Count")
        )
        
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                fig6.add_annotation(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=str(z_values[i][j]),
                    showarrow=False,
                    font=dict(color="black" if z_values[i][j] < (z_values.max() * 0.6) else "white", size=14, weight="bold")
                )
                
        fig6.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig6, width='stretch')
        st.markdown("<p style='font-size: 0.85rem; color: #64748B; font-style: italic; margin-top: -10px;'>Insight: Highlights operational concentrations. Shows how segments prioritize delivery functions.</p>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: RESTAURANT FINDER
# -------------------------------------------------------------
with tab_explorer:
    st.subheader("Explore Restaurants")
    
    # Minimalist Search Container with Confirm Button
    st.markdown("<div style='margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    with st.form("confirmed_search_form"):
        col_search, col_btn = st.columns([5, 1])
        with col_search:
            search_query = st.text_input("🔍 Search Restaurant by Name", "", placeholder="Enter keyword (e.g. Jalsa, Cafe)")
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            confirm_search = st.form_submit_button("Confirm Search")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Filter dataset for table
    df_explorer = df_filtered.copy()
    if confirm_search and search_query:
        df_explorer = df_filtered[df_filtered['name'].str.contains(search_query, case=False, na=False)]
    
    # Highlight lists side-by-side
    col_explorer_left, col_explorer_right = st.columns(2)
    
    with col_explorer_left:
        st.markdown("##### 🏆 Top 5 Most Popular (Highest Votes)")
        top_voted_df = df_explorer.sort_values(by='votes', ascending=False).head(5)[['name', 'listed_in(type)', 'rate', 'votes']]
        st.dataframe(top_voted_df, width='stretch', hide_index=True)
        
    with col_explorer_right:
        st.markdown("##### ⭐ Top 5 Highest Rated")
        top_rated_df = df_explorer.sort_values(by=['rate_clean', 'votes'], ascending=[False, False]).head(5)[['name', 'listed_in(type)', 'rate', 'approx_cost(for two people)']]
        st.dataframe(top_rated_df, width='stretch', hide_index=True)

    st.markdown("<br>##### Complete Filtered Records List", unsafe_allow_html=True)
    
    display_cols = ['name', 'online_order', 'book_table', 'rate', 'votes', 'approx_cost(for two people)', 'listed_in(type)']
    clean_display_df = df_explorer[display_cols].rename(columns={
        'name': 'Restaurant Name',
        'online_order': 'Online Ordering',
        'book_table': 'Table Booking',
        'rate': 'Rating',
        'votes': 'Votes',
        'approx_cost(for two people)': 'Cost for Two (Rs.)',
        'listed_in(type)': 'Type'
    })
    
    st.dataframe(clean_display_df, width='stretch', hide_index=True)
    
    # Download Button
    csv = df_explorer.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Result Table (CSV)",
        data=csv,
        file_name="zomato_explorer_data.csv",
        mime="text/csv",
        help="Export current filtered view to open in Excel or BI tools."
    )

# -------------------------------------------------------------
# TAB 4: MACHINE LEARNING CLASSIFIER (POPULARITY PREDICTOR)
# -------------------------------------------------------------
with tab_ml_predictor:
    st.subheader("🤖 Machine Learning Popularity Predictor")
    st.markdown(
        "Using a mathematical **Logistic Regression Model** trained on the Zomato database, "
        "this tool predicts whether a restaurant will achieve **above-median customer engagement** "
        f"(defined as receiving more than **{median_votes_threshold:.0f} votes**)."
    )
    
    # Model KPIs
    ml_kpi1, ml_kpi2 = st.columns(2)
    with ml_kpi1:
        st.metric(
            label="Model Prediction Accuracy", 
            value=f"{model_accuracy * 100:.1f}%",
            help="Validated on a hold-out 20% validation split."
        )
    with ml_kpi2:
        st.metric(
            label="Success Threshold", 
            value=f"> {median_votes_threshold:.0f} Votes",
            help="Median number of customer votes in the Zomato dataset."
        )
        
    st.markdown("<br><hr style='border-color: #F1F5F9;'>", unsafe_allow_html=True)
    
    # Form layout
    ml_col_left, ml_col_right = st.columns([3, 2])
    
    with ml_col_left:
        # Plotly horizontal bar chart of coefficients (relative impact)
        fig_imp = px.bar(
            model_importance_df,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Factors Driving Restaurant Popularity',
            labels={'Coefficient': 'Directional Impact on Popularity', 'Feature': 'Restaurant Attribute'},
            color='Coefficient',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            color_continuous_midpoint=0
        )
        fig_imp.update_layout(coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_imp, width='stretch')
        st.markdown("<p style='font-size: 0.85rem; color: #64748B; font-style: italic; margin-top: -10px;'>Insight: Displays the mathematical coefficient impact of each attribute. Green bars boost popularity probability; red bars decrease it.</p>", unsafe_allow_html=True)
        
    with ml_col_right:
        st.markdown("##### 🔮 Success Probability Calculator")
        st.markdown("Specify restaurant inputs below to run a real-time logical prediction:")
        
        # User input form
        with st.form("ml_prediction_form"):
            input_type = st.selectbox("Restaurant Type", options=type_options, index=type_options.index('Dining') if 'Dining' in type_options else 0)
            input_cost = st.slider("Approx Cost for Two (Rs.)", min_value=min_cost, max_value=max_cost, value=400, step=50)
            input_rating = st.slider("Expected Customer Rating", min_value=min_rate, max_value=max_rate, value=3.8, step=0.1)
            input_online = st.radio("Offers Online Ordering?", options=['Yes', 'No'], horizontal=True)
            input_booking = st.radio("Offers Table Booking?", options=['Yes', 'No'], horizontal=True)
            
            submit_btn = st.form_submit_button("Predict Success probability")
            
        if submit_btn:
            # Create user input DataFrame
            user_input_df = pd.DataFrame([{
                'online_order': input_online,
                'book_table': input_booking,
                'rate_clean': input_rating,
                'approx_cost(for two people)': input_cost,
                'listed_in(type)': input_type
            }])
            
            # Predict class and probability
            pred_class = model_pipeline.predict(user_input_df)[0]
            pred_probs = model_pipeline.predict_proba(user_input_df)[0]
            success_prob = pred_probs[1]
            
            # Display formatted result box
            if pred_class == 1:
                st.markdown(f"""
                <div class="result-box result-box-success">
                    <div class="result-header">Likely to be Popular! 🎉</div>
                    <div>This layout is mathematically aligned with successful restaurant profiles.</div>
                    <div style="font-size: 1.15rem; font-weight: bold; margin-top: 10px;">Predicted Success Probability: {success_prob * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-box-fail">
                    <div class="result-header">Likely to have Average Popularity 📉</div>
                    <div>This model predicts the layout will fall below the {median_votes_threshold:.0f} votes mark.</div>
                    <div style="font-size: 1.15rem; font-weight: bold; margin-top: 10px;">Predicted Success Probability: {success_prob * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 5: KEY TAKEAWAYS
# -------------------------------------------------------------
with tab_takeaways:
    st.subheader("Key Findings & Business Recommendations")
    st.write("Use these curated findings for presenting the analysis outcomes to management or stakeholders:")
    
    st.markdown("""
    <div class="executive-card">
        <div class="executive-card-title">1. Segment Dominance: Dining as a Core Pillar</div>
        <div class="executive-card-desc">
            The dataset consists of <strong>148 restaurants</strong>, with <strong>Dining</strong> accounting for the vast majority (110 establishments). Buffet, Cafes, and other categories form niche segments. High-volume dining represents the primary business landscape.
        </div>
    </div>
    
    <div class="executive-card">
        <div class="executive-card-title">2. Customer Engagement & Popularity Alignment</div>
        <div class="executive-card-desc">
            Voter participation shows dining establishments capture the highest share of consumer feedback. Businesses categorized under <strong>Dining</strong> receive the highest total votes, which means high customer interactions.
        </div>
    </div>
    
    <div class="executive-card">
        <div class="executive-card-title">3. Online Ordering Impact on review spreads</div>
        <div class="executive-card-desc">
            Restaurants that offer online ordering display a much wider range of ratings (spanning from 2.6 to 4.6). The median rating remains fairly consistent, but online delivery availability facilitates increased review frequencies and helps popular places maintain a steady flow of high ratings.
        </div>
    </div>
    
    <div class="executive-card">
        <div class="executive-card-title">4. Market Pricing Trends</div>
        <div class="executive-card-desc">
            Pricing ranges from <strong>Rs. 100 to Rs. 950</strong> for two. Most restaurants cluster in the Rs. 300 to Rs. 600 range, representing a highly competitive mid-tier dining economy. High-end restaurants (> Rs. 600) correspond directly to higher average votes, showing a correlation between premium offerings and high popularity.
        </div>
    </div>
    
    <div class="executive-card">
        <div class="executive-card-title">5. Operational Synergy (Heatmap analysis)</div>
        <div class="executive-card-desc">
            Heatmap clustering shows dining restaurants without online ordering represent a significant share of traditional business models. Out of 110 dining setups, <strong>80 do not participate in online ordering</strong>, representing a large untapped opportunity to introduce digitized delivery options to increase customer convenience.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown(
    "<div class='footer'>"
    "Zomato Data Analytics Application | Built for Office Presentation & Strategic Showcasing"
    "</div>",
    unsafe_allow_html=True
)
