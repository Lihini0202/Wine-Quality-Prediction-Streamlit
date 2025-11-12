import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 

# Load dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
df.columns = df.columns.str.replace('\"', '').str.strip() 
df['quality_label'] = pd.cut(df['quality'], bins=[0, 5, 7, 10], labels=['Low', 'Medium', 'High'])
df['quality_label'] = df['quality_label'].map({'Low': 0, 'Medium': 1, 'High': 2})
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# Train-test split (needed here for evaluation metrics in app)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model (the best model, which is Random Forest in this case)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Re-train Logistic Regression here for comparison in the app, or load if saved

lr_model_for_app = LogisticRegression(max_iter=1000, random_state=42)
lr_model_for_app.fit(X_train, y_train)
y_pred_lr_app = lr_model_for_app.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr_app)

# Get predictions for the main model (Random Forest) for evaluation in the app
y_pred_main_model = model.predict(X_test)

# Layout
st.set_page_config(page_title="Wine Quality Predictor", layout="wide")
st.title("🍇 My LIVE Deployed Wine Predictor! 🚀)

# Tabs
tabs = st.tabs([
    "🏠 Home", "🔍 Explore Data", "📊 Visualizations", "🎯 Predict Quality",
    "📁 Upload CSV", "📂 Model Info", "🧪 Model Comparison", "📜 Feature Guide"
])

# Tab 1: Home
with tabs[0]:
    st.markdown("""
    <h1 style='text-align: center; font-size: 60px;'>🍷 Welcome to the Wine Quality Predictor App</h1>
    <h4 style='text-align: center; color: gray;'>Machine Learning for Wine Lovers 🍇</h4>
    """, unsafe_allow_html=True)

    st.image("https://c.tenor.com/FW6_Tfz5NpcAAAAd/wine.gif", use_container_width=True)

    st.markdown("""
    <div style="background-color:#white; padding:20px; border-radius:10px; border:1px solid #e0c097">
        <h4>📌 What You Can Do:</h4>
        <ul style="line-height: 1.8;">
            <li>🔍 Explore white wine data</li>
            <li>📊 Visualize relationships and distributions</li>
            <li>🎯 Predict wine quality using a trained model</li>
            <li>📁 Upload your own CSV for predictions</li>
            <li>📖 Learn about wine features</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("- **Dataset:** [UCI White Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)")
    st.markdown("- **Main Model:** Random Forest Classifier") 


# Tab 2: Explore Data
with tabs[1]:
    st.header("🔍 Explore Dataset")
    if st.checkbox("Show Raw Data"):
        st.write(df.head())
    if st.checkbox("Show Summary Statistics"):
        st.write(df.describe())
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    st.pyplot(fig)

# Tab 3: Visualizations
with tabs[2]:
    st.header("📊 Visualizations")
    st.subheader("Distribution Plot")
    dist_col = st.selectbox("Choose a feature", df.columns[:-2])
    fig1, ax1 = plt.subplots()
    sns.histplot(df[dist_col], kde=True, ax=ax1, color="teal")
    st.pyplot(fig1)

    st.subheader("Boxplot vs Wine Quality")
    box_col = st.selectbox("Select feature for boxplot", df.columns[:-2], index=10)
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="quality_label", y=box_col, data=df, ax=ax2)
    st.pyplot(fig2)

# Tab 4: Predict Quality
with tabs[3]:
    st.header("🎯 Predict Wine Quality")
    col1, col2 = st.columns(2)
    with col1:
        fa = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
        va = st.slider("Volatile Acidity", 0.1, 1.5, 0.3)
        ca = st.slider("Citric Acid", 0.0, 1.0, 0.3)
        rs = st.slider("Residual Sugar", 0.5, 15.0, 5.0)
        cl = st.slider("Chlorides", 0.01, 0.2, 0.045)
    with col2:
        fsd = st.slider("Free Sulfur Dioxide", 2.0, 80.0, 30.0)
        tsd = st.slider("Total Sulfur Dioxide", 9.0, 300.0, 115.0)
        dens = st.slider("Density", 0.987, 1.005, 0.994)
        ph = st.slider("pH", 2.5, 4.5, 3.2)
        sul = st.slider("Sulphates", 0.2, 1.5, 0.5)
        alc = st.slider("Alcohol", 8.0, 14.0, 10.0)
    if st.button("Predict"):
        input_df = pd.DataFrame([{"fixed acidity": fa, "volatile acidity": va, "citric acid": ca, "residual sugar": rs,
                                  "chlorides": cl, "free sulfur dioxide": fsd, "total sulfur dioxide": tsd,
                                  "density": dens, "pH": ph, "sulphates": sul, "alcohol": alc}])
        pred = model.predict(input_df)[0]
        label = {0: "Low", 1: "Medium", 2: "High"}[pred]
        st.success(f"Predicted Wine Quality: **{label}**")
    
    st.metric("Main Model (Random Forest) Accuracy on Test Set", f"{accuracy_score(y_test, y_pred_main_model):.2%}")

    st.subheader("Model Performance Details")
    st.text("Classification Report (Random Forest on Test Set):")
    st.code(classification_report(y_test, y_pred_main_model, target_names=['Low', 'Medium', 'High']))

    st.text("Confusion Matrix (Random Forest on Test Set):")
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    cmp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_main_model), display_labels=['Low', 'Medium', 'High'])
    cmp.plot(ax=ax_cm, cmap='Blues')
    plt.title('Confusion Matrix for Random Forest Classifier')
    st.pyplot(fig_cm)


# Tab 5: Upload CSV
with tabs[4]:
    st.header("📁 Upload Your Own Wine Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.write("Preview:", user_df.head())
        try:
            # Ensure columns match training data
            missing_cols = set(X.columns) - set(user_df.columns)
            if missing_cols:
                st.error(f"Error: Missing columns in uploaded CSV: {', '.join(missing_cols)}. Please ensure your CSV has all required features.")
            else:
                # Reorder columns to match the training data's column order
                user_df_processed = user_df[X.columns]
                preds = model.predict(user_df_processed)
                user_df["Predicted Quality"] = [ {0: "Low", 1: "Medium", 2: "High"}[p] for p in preds ]
                st.success("Predictions complete!")
                st.write(user_df)
        except Exception as e:
            st.error(f"Error making predictions. Check column names and formats. Details: {e}")

# Tab 6: Model Info
with tabs[5]:
    st.header("📂 Model Details & Download")
    st.markdown("**Main Model:** Random Forest Classifier (n_estimators=100, random_state=42)")
    st.markdown("**Classes:** Low, Medium, High")
    st.write("The `model.pkl` file contains the trained Random Forest Classifier.")
    with open("model.pkl", "rb") as f:
        st.download_button("Download model.pkl", f.read(), file_name="model.pkl")

# Tab 7: Model Comparison
with tabs[6]:
    st.header("🧪 Model Comparison")
    # Dynamically show accuracies of trained models
    rf_accuracy_app = accuracy_score(y_test, model.predict(X_test)) # Accuracy of the main RF model
    
    models_comparison = {
        "Random Forest": rf_accuracy_app,
        "Logistic Regression": lr_accuracy # Use the actual calculated LR accuracy
    }
    
    st.markdown(f"**Random Forest Accuracy:** `{rf_accuracy_app:.2%}`")
    st.markdown(f"**Logistic Regression Accuracy:** `{lr_accuracy:.2%}`")

    fig3, ax3 = plt.subplots()
    sns.barplot(x=list(models_comparison.values()), y=list(models_comparison.keys()), ax=ax3, palette='viridis')
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Accuracy")
    ax3.set_title("Model Accuracy Comparison on Test Set")
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Why Random Forest was chosen:")
    st.write("Random Forest models generally offer high accuracy, handle non-linear relationships well, and are less prone to overfitting compared to single decision trees.")
    st.write("Based on our evaluation, the Random Forest model performed better than Logistic Regression in classifying wine quality.")


# Tab 8: Feature Guide
with tabs[7]:
    st.header("📜 Feature Reference Guide")
    st.markdown("""
    - **Fixed Acidity**: Represents the non-volatile acids in wine. Tartaric acid is a major fixed acid. Higher levels can indicate a more tart taste.
    - **Volatile Acidity**: Primarily acetic acid, which can give wine a vinegar-like taste if too high. It's an indicator of spoilage.
    - **Citric Acid**: A small amount of citric acid adds 'freshness' and flavor to wines.
    - **Residual Sugar**: The amount of sugar remaining after fermentation. Sweetness in wine is directly related to this.
    - **Chlorides**: The amount of salt in the wine. Higher levels can indicate a salty taste.
    - **Free Sulfur Dioxide**: The unreactive form of SO2, which helps prevent microbial growth and oxidation in wine.
    - **Total Sulfur Dioxide**: The total amount of SO2 (free and bound forms) in the wine. Acts as a preservative.
    - **Density**: Relates to the sugar content, as sugar is denser than alcohol. Often indicates the alcohol and extract content.
    - **pH**: Measures the acidity or basicity of the wine, on a scale of 0 (very acidic) to 14 (very basic). Most wines are between 3-4 pH.
    - **Sulphates**: A wine additive which contributes to the sulfur dioxide levels and acts as an antimicrobial and antioxidant agent, enhancing shelf life.
    - **Alcohol**: The percentage of alcohol content by volume. Higher alcohol content generally leads to a fuller-bodied wine.
    """)
