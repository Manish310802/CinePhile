import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained models
with open("movie_recommendation_model1.pkl", "rb") as file:
    model1_data = pickle.load(file)

with open("movie_recommendation_model2.pkl", "rb") as file:
    model2_data = pickle.load(file)

# Extract data from the models
nn_model1 = model1_data['nn_model']
feature_matrix1 = model1_data['feature_matrix']
titles1 = model1_data['titles']
indices1 = model1_data['indices']

nn_model2 = model2_data['nn_model']
feature_matrix2 = model2_data['feature_matrix']
titles2 = model2_data['titles']
indices2 = model2_data['indices']

# Set up the sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Recommendation Model", ["Model 1: Bollywood Movies", "Model 2: Hollywood Movies"])

# Helper function to get recommendations
def get_recommendations(model, feature_matrix, titles, indices, movie_title, num_recommendations=5):
    movie_index = indices[movie_title]
    distances, indices = model.kneighbors(feature_matrix[movie_index], n_neighbors=num_recommendations+1)
    recommendations = []

    for i in range(1, len(indices[0])):
        recommendations.append(titles[indices[0][i]])

    return recommendations

# Function for the first model
def model_1_interface():
    st.title("Movie Recommendation System")
    st.header("Model 1: Bollywood Movies")

    # Dropdown for movie selection
    movie_title = st.selectbox("Select a movie", options=titles1)
    num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    # Button to get recommendations
    if st.button("Get Recommendations", key="model1_button"):
        recommendations = get_recommendations(nn_model1, feature_matrix1, titles1, indices1, movie_title, num_recommendations)
        st.write("### Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

# Function for the second model
def model_2_interface():
    st.title("Movie Recommendation System")
    st.header("Model 2: Hollywood Movies")

    # Dropdown for movie selection
    movie_title = st.selectbox("Select a movie", options=titles2)
    num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    # Button to get recommendations
    if st.button("Get Recommendations", key="model2_button"):
        recommendations = get_recommendations(nn_model2, feature_matrix2, titles2, indices2, movie_title, num_recommendations)
        st.write("### Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

# Display the corresponding model interface based on selection
if page == "Model 1: Bollywood Movies":
    model_1_interface()
else:
    model_2_interface()
