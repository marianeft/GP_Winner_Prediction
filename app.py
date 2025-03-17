import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

def predict_grand_prix_winner(previous_season_data):
    """
    Predicts the grand prix winner for the 2025 season based on previous season data.

    Args:
        previous_season_data (pd.DataFrame): DataFrame containing previous season results.

    Returns:
        pd.DataFrame: DataFrame containing predicted results for the 2025 season.
    """

    # Preprocessing
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    previous_season_data['Driver_encoded'] = le_driver.fit_transform(previous_season_data['Driver'])
    previous_season_data['Team_encoded'] = le_team.fit_transform(previous_season_data['Team'])

    # Feature Engineering
    features = ['Driver_encoded', 'Team_encoded', 'Points', 'Starting Grid']
    target = 'Position'

    # Model Training
    model = LinearRegression()
    model.fit(previous_season_data[features], previous_season_data[target])

    # Create a DataFrame for 2025 predictions
    last_race = previous_season_data.groupby('Driver').last().reset_index()

    # Predict positions for 2025
    predictions = model.predict(last_race[features])
    last_race['Predicted_Position'] = predictions
    last_race['Predicted_Position'] = last_race['Predicted_Position'].round().astype(int)

    # Decode labels back to original names
    last_race['Driver'] = le_driver.inverse_transform(last_race['Driver_encoded'])
    last_race['Team'] = le_team.inverse_transform(last_race['Team_encoded'])

    # Sort by predicted position
    predicted_results = last_race[['Driver', 'Team', 'Predicted_Position']].sort_values(by='Predicted_Position')

    return predicted_results

def main():
    st.title("2025 Grand Prix Winner Prediction")

    st.write("Upload the previous season's results (CSV format).")

    # Modified file_uploader to accept CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Input for GP name and date
    gp_name = st.text_input("Enter the Grand Prix Name (e.g., Australian Grand Prix):")
    gp_date = st.date_input("Enter the Date of the Grand Prix:")

    if uploaded_file is not None:
        try:
            previous_season_data = pd.read_csv(uploaded_file)
            st.write("Uploaded data:")
            st.dataframe(previous_season_data)

            # Ensure necessary columns exist
            required_columns = ['Driver', 'Team', 'Points', 'Position', 'Starting Grid']
            if not all(col in previous_season_data.columns for col in required_columns):
                st.error(f"Error: CSV must contain the following columns: {', '.join(required_columns)}")
                return

            if st.button("Predict 2025 Winners"):
                predicted_results = predict_grand_prix_winner(previous_season_data)
                st.write("Predicted 2025 Grand Prix Results:")
                st.dataframe(predicted_results)
                if not predicted_results.empty:
                    winner = predicted_results.iloc[0]['Driver']
                    team = predicted_results.iloc[0]['Team']
                    # Display the prediction with GP name and date
                    st.success(f"2025 {gp_name} Winner ({gp_date.strftime('%Y-%m-%d')}): {winner} from team {team}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a CSV file with the previous season's results.")

if __name__ == "__main__":
    main()