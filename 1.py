import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import statsmodels.api as sm
from scipy.stats import f_oneway, chi2_contingency, ttest_ind
import io
import warnings

warnings.filterwarnings('ignore')

# Set wide page layout and custom theme
st.set_page_config(
    page_title="Advanced Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better looking app
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    div.stButton > button:first-child {
        background-color: #0099ff;
        color: white;
    }
    div.stButton > button:hover {
        background-color: #00ff00;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🚀 Advanced Data Analysis Suite")
st.markdown("""
    ### A Comprehensive Data Analysis and Visualization Platform
    Upload your data and explore powerful insights through interactive visualizations and advanced analytics.
""")

# Sidebar configuration
st.sidebar.title("Configuration")

# File upload section
st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV or Excel file",
    type=['csv', 'xlsx'],
    help="Upload your data file in CSV or Excel format"
)


# Function to load and process data
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Automatically detect and convert date columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except ValueError:
                    pass

        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# Initialize session state for storing data and settings
if 'data' not in st.session_state:
    st.session_state['data'] = None

# Load data if file is uploaded
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.session_state['data'] = df

    if df is not None:
        # Data Overview Section
        st.header("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        with col4:
            st.metric("Duplicate Rows", df.duplicated().sum())

        # Data Preview with custom styling
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Column Info
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.notnull().sum(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)

        # Automatic Data Type Detection and Conversion Options
        st.sidebar.header("2. Data Preprocessing")

        # Column selection and type conversion
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns

        # Data Cleaning Options
        if st.sidebar.checkbox("Show Data Cleaning Options"):
            st.subheader("🧹 Data Cleaning Options")

            col1, col2, col3 = st.columns(3)
            with col1:
                handle_missing = st.radio(
                    "Handle Missing Values",
                    ["Drop", "Fill with Mean/Mode", "Fill with Zero", "Fill with Custom Value"],
                    key="handle_missing_radio"
                )

            with col2:
                handle_duplicates = st.checkbox("Remove Duplicate Rows", key="handle_duplicates_checkbox")

            with col3:
                handle_outliers = st.checkbox("Remove Outliers", key="handle_outliers_checkbox")

            if handle_missing == "Fill with Mean/Mode":
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
            elif handle_missing == "Drop":
                df.dropna(inplace=True)
            elif handle_missing == "Fill with Zero":
                df.fillna(0, inplace=True)
            elif handle_missing == "Fill with Custom Value":
                custom_value = st.text_input("Enter custom value for missing data:", key="custom_value_input")
                if custom_value:
                    df.fillna(custom_value, inplace=True)

            if handle_duplicates:
                df.drop_duplicates(inplace=True)

            if handle_outliers:
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # Visualization Options
        st.sidebar.header("3. Visualization Settings")

        # Create tabs for different types of analysis
        tabs = st.tabs([
            "📈 Basic Analysis",
            "🔍 Advanced Analysis",
            "📊 Custom Plots",
            "📉 Statistical Analysis",
            "🤖 Machine Learning"
        ])

        # Basic Analysis Tab
        with tabs[0]:
            st.header("📈 Basic Analysis")

            # Automatic correlation matrix for numeric columns
            if len(numeric_cols) > 0:
                st.subheader("Correlation Matrix")
                fig = px.imshow(
                    df[numeric_cols].corr(),
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True, key="correlation_matrix")

            # Distribution plots for numeric columns
            st.subheader("Distribution Analysis")
            selected_col = st.selectbox(
                "Select column for distribution analysis",
                df.columns,
                key="dist_analysis_col_select_basic"
            )

            col1, col2 = st.columns(2)
            with col1:
                if df[selected_col].dtype in ['int64', 'float64']:
                    fig = px.histogram(
                        df,
                        x=selected_col,
                        title=f"Histogram of {selected_col}"
                    )
                else:
                    value_counts = df[selected_col].value_counts().reset_index()
                    value_counts.columns = ['Category', 'Count']
                    fig = px.bar(
                        value_counts,
                        x='Category',
                        y='Count',
                        title=f"Bar Plot of {selected_col}"
                    )
                st.plotly_chart(fig, use_container_width=True, key=f"dist_plot_1_{selected_col}")

            with col2:
                if df[selected_col].dtype in ['int64', 'float64']:
                    fig = px.box(
                        df,
                        y=selected_col,
                        title=f"Box Plot of {selected_col}"
                    )
                else:
                    value_counts = df[selected_col].value_counts().reset_index()
                    value_counts.columns = ['Category', 'Count']
                    fig = px.pie(
                        value_counts,
                        values='Count',
                        names='Category',
                        title=f"Pie Chart of {selected_col}"
                    )
                st.plotly_chart(fig, use_container_width=True, key=f"dist_plot_2_{selected_col}")

            # Categorical Data Analysis
            if len(categorical_cols) > 0:
                st.subheader("Categorical Data Analysis")
                cat_col = st.selectbox("Select categorical column", categorical_cols, key="cat_analysis_col_select")
                value_counts = df[cat_col].value_counts().reset_index()
                value_counts.columns = ['Category', 'Count']
                fig = px.bar(value_counts, x='Category', y='Count', title=f"Bar Plot of {cat_col}")
                st.plotly_chart(fig, use_container_width=True, key=f"cat_plot_{cat_col}")

        # Advanced Analysis Tab
        with tabs[1]:
            st.header("🔍 Advanced Analysis")

            # Time Series Analysis (if date columns exist)
            if len(date_cols) > 0:
                st.subheader("Time Series Analysis")
                date_col = st.selectbox("Select Date Column", date_cols, key="time_series_date_col_select_advanced")
                value_col = st.selectbox("Select Value Column", numeric_cols,
                                         key="time_series_value_col_select_advanced")

                # Resample options
                resample_dict = {
                    'Daily': 'D',
                    'Weekly': 'W',
                    'Monthly': 'M',
                    'Quarterly': 'Q',
                    'Yearly': 'Y'
                }
                resample_period = st.selectbox(
                    "Select Time Period",
                    list(resample_dict.keys()),
                    key="time_series_period_select"
                )

                # Create time series plot
                df_time = df.set_index(date_col)
                df_resampled = df_time[value_col].resample(
                    resample_dict[resample_period]
                ).mean()

                fig = px.line(
                    df_resampled,
                    title=f"{value_col} over Time ({resample_period})"
                )
                st.plotly_chart(fig, use_container_width=True, key=f"time_series_{value_col}_{resample_period}")

            # Multivariate Analysis
            st.subheader("Multivariate Analysis")
            selected_cols = st.multiselect("Select columns for analysis", df.columns, default=numeric_cols[:3],
                                           key="multivariate_cols_select")
            if len(selected_cols) > 1:
                fig = px.scatter_matrix(df[selected_cols])
                st.plotly_chart(fig, use_container_width=True, key=f"scatter_matrix_{'_'.join(selected_cols)}")

            # Heatmap
            st.subheader("Correlation Heatmap")
            heatmap_cols = st.multiselect("Select columns for heatmap", numeric_cols, default=numeric_cols,
                                          key="heatmap_cols_select")
            if len(heatmap_cols) > 1:
                fig = px.imshow(df[heatmap_cols].corr(), color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{'_'.join(heatmap_cols)}")

            # Pair Plot
            st.subheader("Pair Plot")
            pair_cols = st.multiselect("Select columns for pair plot", numeric_cols, default=numeric_cols[:3],
                                       key="pair_plot_cols_select")
            if len(pair_cols) > 1:
                fig = sns.pairplot(df[pair_cols])
                st.pyplot(fig)

        # Custom Plots Tab
        with tabs[2]:
            st.header("📊 Custom Plots")

            plot_types = ["Scatter", "Line", "Bar", "Box", "Violin", "3D Scatter", "Bubble", "Contour", "Heatmap",
                          "Polar", "Area", "Funnel", "Treemap"]
            plot_type = st.selectbox("Select Plot Type", plot_types, key="custom_plot_type_select")


            # Function to create plots
            def create_plot(plot_type, plot_id):
                if plot_type == "Scatter":
                    x_col = st.selectbox("Select X axis", df.columns, key=f"scatter_x_{plot_type}_{plot_id}")
                    y_col = st.selectbox("Select Y axis", df.columns, key=f"scatter_y_{plot_type}_{plot_id}")
                    color_col = st.selectbox("Select Color Variable", df.columns,
                                             key=f"scatter_color_{plot_type}_{plot_id}")
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Scatter Plot: {x_col} vs {y_col}")

                elif plot_type == "Line":
                    x_col = st.selectbox("Select X axis", df.columns, key=f"line_x_{plot_type}_{plot_id}")
                    y_cols = st.multiselect("Select Y axis (multiple)", numeric_cols,
                                            key=f"line_y_{plot_type}_{plot_id}")
                    fig = px.line(df, x=x_col, y=y_cols, title="Line Plot")

                elif plot_type == "Bar":
                    x_col = st.selectbox("Select X axis", df.columns, key=f"bar_x_{plot_type}_{plot_id}")
                    y_col = st.selectbox("Select Y axis", numeric_cols, key=f"bar_y_{plot_type}_{plot_id}")
                    color_col = st.selectbox("Select Color Variable", df.columns,
                                             key=f"bar_color_{plot_type}_{plot_id}")
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col, title="Bar Plot")

                elif plot_type == "Box":
                    x_col = st.selectbox("Select X axis", df.columns, key=f"box_x_{plot_type}_{plot_id}")
                    y_col = st.selectbox("Select Y axis", numeric_cols, key=f"box_y_{plot_type}_{plot_id}")
                    fig = px.box(df, x=x_col, y=y_col, title="Box Plot")

                elif plot_type == "Violin":
                    x_col = st.selectbox("Select X axis", df.columns, key=f"violin_x_{plot_type}_{plot_id}")
                    y_col = st.selectbox("Select Y axis", numeric_cols, key=f"violin_y_{plot_type}_{plot_id}")
                    fig = px.violin(df, x=x_col, y=y_col, title="Violin Plot")

                elif plot_type == "3D Scatter":
                    x_col = st.selectbox("Select X axis", numeric_cols, key=f"3d_x_{plot_type}_{plot_id}")
                    y_col = st.selectbox("Select Y axis", numeric_cols, key=f"3d_y_{plot_type}_{plot_id}")
                    z_col = st.selectbox("Select Z axis", numeric_cols, key=f"3d_z_{plot_type}_{plot_id}")
                    color_col = st.selectbox("Select Color Variable", df.columns, key=f"3d_color_{plot_type}_{plot_id}")
                    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col, title="3D Scatter Plot")

                elif plot_type == "Bubble":
                    x_col = st.selectbox("Select X axis", numeric_cols, key=f"bubble_x_{plot_type}_{plot_id}")
                    y_col = st.selectbox("Select Y axis", numeric_cols, key=f"bubble_y_{plot_type}_{plot_id}")
                    size_col = st.selectbox("Select Size Variable", numeric_cols,
                                            key=f"bubble_size_{plot_type}_{plot_id}")
                    color_col = st.selectbox("Select Color Variable", df.columns,
                                             key=f"bubble_color_{plot_type}_{plot_id}")
                    fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col, title="Bubble Plot")

                elif plot_type == "Contour":
                    x_col = st.selectbox("Select X axis", numeric_cols, key=f"contour_x_{plot_type}_{plot_id}")
                    y_col = st.selectbox("Select Y axis", numeric_cols, key=f"contour_y_{plot_type}_{plot_id}")
                    z_col = st.selectbox("Select Z axis", numeric_cols, key=f"contour_z_{plot_type}_{plot_id}")
                    fig = px.density_contour(df, x=x_col, y=y_col, z=z_col, title="Contour Plot")

                elif plot_type == "Heatmap":
                    x_col = st.selectbox("Select X axis", df.columns, key=f"heatmap_x_{plot_type}_{plot_id}")
                    y_col = st.selectbox("Select Y axis", df.columns, key=f"heatmap_y_{plot_type}_{plot_id}")
                    z_col = st.selectbox("Select Z axis (values)", numeric_cols, key=f"heatmap_z_{plot_type}_{plot_id}")
                    fig = px.density_heatmap(df, x=x_col, y=y_col, z=z_col, title="Heatmap")

                elif plot_type == "Polar":
                    r_col = st.selectbox("Select Radius", numeric_cols, key=f"polar_r_{plot_type}_{plot_id}")
                    theta_col = st.selectbox("Select Theta", df.columns, key=f"polar_theta_{plot_type}_{plot_id}")
                    color_col = st.selectbox("Select Color Variable", df.columns,
                                             key=f"polar_color_{plot_type}_{plot_id}")
                    fig = px.scatter_polar(df, r=r_col, theta=theta_col, color=color_col, title="Polar Plot")

                elif plot_type == "Area":
                    x_col = st.selectbox("Select X axis", df.columns, key=f"area_x_{plot_type}_{plot_id}")
                    y_cols = st.multiselect("Select Y axis (multiple)", numeric_cols,
                                            key=f"area_y_{plot_type}_{plot_id}")
                    fig = px.area(df, x=x_col, y=y_cols, title="Area Plot")

                elif plot_type == "Funnel":
                    x_col = st.selectbox("Select Values", numeric_cols, key=f"funnel_x_{plot_type}_{plot_id}")
                    y_col = st.selectbox("Select Stages", df.columns, key=f"funnel_y_{plot_type}_{plot_id}")
                    fig = px.funnel(df, x=x_col, y=y_col, title="Funnel Plot")

                elif plot_type == "Treemap":
                    path_cols = st.multiselect("Select Hierarchy (multiple)", df.columns,
                                               key=f"treemap_path_{plot_type}_{plot_id}")
                    values_col = st.selectbox("Select Values", numeric_cols,
                                              key=f"treemap_values_{plot_type}_{plot_id}")
                    fig = px.treemap(df, path=path_cols, values=values_col, title="Treemap")

                return fig


            # Create and display the selected plot
            fig = create_plot(plot_type, "main")
            st.plotly_chart(fig, use_container_width=True, key=f"custom_plot_{plot_type}_main")

            # Multiple plot creation
            st.subheader("Create Multiple Plots")
            num_plots = st.number_input("Number of additional plots", min_value=1, max_value=10, value=1,
                                        key="num_plots_input")

            for i in range(num_plots):
                st.subheader(f"Additional Plot {i + 1}")
                plot_type = st.selectbox(f"Select Plot Type for Plot {i + 1}", plot_types, key=f"add_plot_type_{i}")
                fig = create_plot(plot_type, f"additional_{i}")
                st.plotly_chart(fig, use_container_width=True, key=f"additional_plot_{i}_{plot_type}")

        # Statistical Analysis Tab
        with tabs[3]:
            st.header("📉 Statistical Analysis")

            # Summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)

            # Statistical Tests
            st.subheader("Statistical Tests")
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    var1 = st.selectbox("Select First Variable", numeric_cols, key="stat_var1_select_statistical")
                with col2:
                    var2 = st.selectbox("Select Second Variable", numeric_cols, key="stat_var2_select_statistical")

                # Correlation test
                correlation = df[var1].corr(df[var2])
                st.write(f"Correlation coefficient: {correlation:.4f}")

                # Linear regression
                X = sm.add_constant(df[var1])
                model = sm.OLS(df[var2], X).fit()
                st.write("Regression Summary:")
                st.text(model.summary().as_text())

            # ANOVA
            st.subheader("ANOVA Test")
            cat_col = st.selectbox("Select Categorical Variable", categorical_cols, key="anova_cat_select")
            num_col = st.selectbox("Select Numeric Variable", numeric_cols, key="anova_num_select")
            groups = [group for name, group in df.groupby(cat_col)[num_col]]
            f_value, p_value = f_oneway(*groups)
            st.write(f"F-value: {f_value:.4f}")
            st.write(f"p-value: {p_value:.4f}")

            # Chi-square test
            st.subheader("Chi-square Test")
            cat_col1 = st.selectbox("Select First Categorical Variable", categorical_cols, key="chi2_cat1_select")
            cat_col2 = st.selectbox("Select Second Categorical Variable", categorical_cols, key="chi2_cat2_select")
            contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            st.write(f"Chi-square statistic: {chi2:.4f}")
            st.write(f"p-value: {p_value:.4f}")

            # T-test
            st.subheader("Independent T-Test")
            t_cat_col = st.selectbox("Select Categorical Variable", categorical_cols, key="ttest_cat_select")
            t_num_col = st.selectbox("Select Numeric Variable", numeric_cols, key="ttest_num_select")
            if len(df[t_cat_col].unique()) >= 2:
                group1 = df[df[t_cat_col] == df[t_cat_col].unique()[0]][t_num_col]
                group2 = df[df[t_cat_col] == df[t_cat_col].unique()[1]][t_num_col]
                t_stat, p_value = ttest_ind(group1, group2)
                st.write(f"T-statistic: {t_stat:.4f}")
                st.write(f"p-value: {p_value:.4f}")
            else:
                st.write("Please select a categorical variable with at least two unique values for the T-test.")

        # Machine Learning Tab
        with tabs[4]:
            st.header("🤖 Machine Learning")

            # Clustering Analysis
            st.subheader("Clustering Analysis")
            cluster_cols = st.multiselect(
                "Select columns for clustering",
                numeric_cols,
                key="cluster_cols_select"
            )

            if len(cluster_cols) > 0:
                n_clusters = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=10,
                    value=3,
                    key="n_clusters_slider"
                )

                # Perform clustering
                X = df[cluster_cols]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)

                # Visualize clusters
                if len(cluster_cols) >= 2:
                    fig = px.scatter(
                        df,
                        x=cluster_cols[0],
                        y=cluster_cols[1],
                        color=clusters,
                        title="Cluster Analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"cluster_plot_{'_'.join(cluster_cols)}")

            # Principal Component Analysis (PCA)
            st.subheader("Principal Component Analysis (PCA)")
            pca_cols = st.multiselect("Select columns for PCA", numeric_cols, key="pca_cols_select")
            if len(pca_cols) > 1:
                n_components = st.slider("Number of components", min_value=2, max_value=min(len(pca_cols), 10), value=2,
                                         key="n_components_slider")
                X = df[pca_cols]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(X_scaled)
                pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i + 1}' for i in range(n_components)])
                fig = px.scatter(pca_df, x='PC1', y='PC2', title='PCA Plot')
                st.plotly_chart(fig, use_container_width=True, key=f"pca_plot_{'_'.join(pca_cols)}")

            # Predictive Modeling
            st.subheader("Predictive Modeling")
            target_col = st.selectbox("Select Target Variable", df.columns, key="target_col_select_ml")
            feature_cols = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_col],
                                          key="feature_cols_select")

            if len(feature_cols) > 0:
                # Prepare data
                X = df[feature_cols]
                y = df[target_col]

                # Handle categorical variables
                categorical_features = X.select_dtypes(include=['object']).columns
                if len(categorical_features) > 0:
                    le = LabelEncoder()
                    for col in categorical_features:
                        X[col] = le.fit_transform(X[col])

                if y.dtype == 'object':
                    y = le.fit_transform(y)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Model selection
                model_type = st.radio("Select Model Type", ["Classification", "Regression"], key="model_type_radio")

                if model_type == "Classification":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Model Accuracy: {accuracy:.4f}")
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"Mean Squared Error: {mse:.4f}")
                    st.write(f"R-squared Score: {r2:.4f}")

                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                st.subheader("Feature Importance")
                fig = px.bar(feature_importance, x='feature', y='importance', title="Feature Importance")
                st.plotly_chart(fig, use_container_width=True, key="feature_importance_plot")

        # Export Options
        st.sidebar.header("4. Export Options")
        if st.sidebar.button("Export Processed Data"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                label="Download Processed Data",
                data=output,
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    # Display sample data or instructions when no file is uploaded
    st.info("""
        👆 Please upload your data file to begin analysis.

        Supported formats:
        - CSV (.csv)
        - Excel (.xlsx)

        The tool will automatically detect:
        - Numeric columns for statistical analysis
        - Categorical columns for grouping
        - Date columns for time series analysis
    """)

# Add footer
st.markdown("""
    ---
    Created with ❤️ using Streamlit

    Features:
    - Automatic data type detection
    - Interactive visualizations
    - Advanced statistical analysis
    - Machine learning capabilities
    - Custom plot generation
    - Time series analysis
    - Data cleaning options
    - Export functionality
""")

