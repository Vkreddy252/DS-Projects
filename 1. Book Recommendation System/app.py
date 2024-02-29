# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:59:26 2024

@author: Vinu
"""

import pickle
import pandas as pd
import streamlit as st
final=pd.read_csv("final.csv")
# Load the pre-trained model
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

def predict_BRS(user_id, n, all_books):
    
    user_books = all_books[all_books['User_ID'] == user_id]['Book_Title'].unique()

    # Remove books already rated by the user
    to_predict = [book for book in all_books['Book_Title'].unique() if book not in user_books]

    # Make predictions for the books to predict
    test_data = [(user_id, book, 0) for book in to_predict]
    predictions = model.test(test_data)

    # Sort the predictions and get the top N recommendations
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    # Get book titles for the top N recommendations
    recommended_books = [item[1] for item in top_n]

    return recommended_books

def main():
    st.title("Book Recommendations")
    html_temp = """
    <div style="background-color: #000080 ; padding: 5px">
    <h2 style="color: white; text-align: center;">Book Recommendation ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    User_ID = st.text_input("User_ID")
    Number = st.text_input("Number of Books")

    # Convert inputs to integers
    User_ID = int(User_ID) if User_ID.isdigit() else None
    Number = int(Number) if Number.isdigit() else None

    result = ""
    if st.button("Recommend") and User_ID is not None and Number is not None:
        # Assuming 'final' is your DataFrame
        all_books = final[['User_ID', 'Book_Title']]
        recommendations = predict_BRS(user_id=User_ID, n=Number, all_books=all_books)

        # Display the recommended books with indices
        result += "Recommended Books:\n"
        for i, book in enumerate(recommendations, 1):
            result += f"{i}. {book}\n"

    st.success(result)

if __name__ == '__main__':
    main()