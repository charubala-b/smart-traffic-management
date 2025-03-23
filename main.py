import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import base64

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['dayofweek'] = data['timestamp'].dt.dayofweek
    data['is_weekend'] = data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    data.drop(columns=['timestamp'], inplace=True)

    le = LabelEncoder()
    data['weather_condition'] = le.fit_transform(data['weather_condition'])

    scaler = MinMaxScaler()
    scaled_columns = ['traffic_volume', 'avg_vehicle_speed', 'vehicle_count_cars', 'vehicle_count_trucks',
                      'vehicle_count_bikes', 'temperature', 'humidity']
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

    return data

SEQ_LENGTH = 24

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data.iloc[i:i + seq_length][['traffic_volume', 'avg_vehicle_speed', 'vehicle_count_cars',
                                              'vehicle_count_trucks', 'vehicle_count_bikes', 'weather_condition',
                                              'temperature', 'humidity', 'hour', 'dayofweek', 'is_weekend']].values)
        y.append(data.iloc[i + seq_length]['traffic_volume'])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def suggest_route(congestion_level):
    if congestion_level > 0.8:
        return "🚨 Take an alternative route:(Heavy traffic)"
    elif congestion_level > 0.5:
        return "⚠️ Moderate traffic, consider another route."
    else:
        return "✅ Traffic is clear. Continue on your current route."

def run_dashboard_with_filter(model, x_test):
    y_pred = model.predict(x_test)

    st.title("🚗 Traffic Congestion Prediction Dashboard")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("### Predicting traffic congestion and suggesting alternative routes for better decision-making")
    
    st.sidebar.header("About the Project")
    st.sidebar.write(
        "This model predicts traffic congestion based on historical data and suggests alternative routes to avoid traffic jams."
    )

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    img_data = encode_image('C:/Users/charubala/Desktop/placement/Projects/SmartTrafficPrediction/SmartTrafficPrediction/back1.jpg')
    st.markdown(f"""
        <style>
            .stApp {{
                background: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), 
                            url('data:image/jpg;base64,{img_data}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            .stTitle {{
                color: white;
            }}
            .stSidebar {{
                background: linear-gradient(to right, rgba(216, 184, 158, 0.8), rgba(201, 166, 141, 0.8));

            }}
            .stDataFrame{{
                width: 1000px;
            }}
            .stLineChart{{
                width: 1000px;
                height: 400px;
                margin: 0 auto;
            }}
        </style>    
    """, unsafe_allow_html=True)

    st.sidebar.subheader("Filter by Hour Range")
    start_hour = st.sidebar.slider("Start Hour", min_value=1, max_value=24, value=1)
    end_hour = st.sidebar.slider("End Hour", min_value=start_hour, max_value=24, value=24)

    filtered_indices = range(start_hour - 1, end_hour)
    filtered_predictions = y_pred.flatten()[filtered_indices]

    st.subheader(f"Predicted Traffic Congestion (Hours {start_hour} to {end_hour}):")
    st.line_chart(filtered_predictions)

    route_suggestions = [suggest_route(pred) for pred in filtered_predictions]

    congestion_data = {
        'Hour': list(filtered_indices),
        'Predicted Congestion Level': filtered_predictions,
        'Suggested Route': route_suggestions
    }
    congestion_df = pd.DataFrame(congestion_data)

    st.dataframe(congestion_df)

    st.download_button(
        label="Download Filtered Congestion Data",
        data=congestion_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_traffic_congestion.csv',
        mime='text/csv'
    )

def load_dataset():
    return pd.read_csv('smart_traffic_management_dataset.csv')

def chatbot_page(df):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    img_data = encode_image('C:/Users/charubala/Desktop/placement/Projects/SmartTrafficPrediction/SmartTrafficPrediction/back1.jpg')
    st.markdown(f"""
        <style>
            .stApp {{
                background: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), 
                            url('data:image/jpg;base64,{img_data}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            .stTitle {{
                color: white;
            }}
            .stSidebar {{
                background: linear-gradient(to right, rgba(216, 184, 158, 0.8), rgba(201, 166, 141, 0.8));

            }}
            .stMarkdown{{
                background:linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5));
            }}
        </style>    
    """, unsafe_allow_html=True)
    st.title("Smart Traffic Management Chatbot")

    if st.sidebar.checkbox("Show Dataset"):
        st.write(df)

    st.header("Ask Questions About the Dataset")
    question = st.text_input("Enter your question:")

    if question:
        if "average vehicle speed" in question.lower():
            st.write("Answer: Average Vehicle Speed:", df["avg_vehicle_speed"].mean())
        elif "average volume" in question.lower():
            st.write("Answer: Average Volume:", df["traffic_volume"].mean())
        elif "summary" in question.lower():
            st.write("Summary of the Dataset:")
            st.write("- Total Records:", len(df))
            st.write("- Average Vehicle Speed:", df["avg_vehicle_speed"].mean())
            st.write("- Average Volume:", df["traffic_volume"].mean())
        elif "trend" in question.lower():
            trend_col = st.selectbox("Select a column for trend analysis:", ["avg_vehicle_speed", "traffic_volume", "temperature", "humidity"])
            if trend_col:
                st.line_chart(df[trend_col])
        elif "bar chart" in question.lower() or "barchart" in question.lower():
            bar_col = st.selectbox("Select a column for bar chart:", ["vehicle_count_trucks", "vehicle_count_bikes", "signal_status"])
            if bar_col:
                fig, ax = plt.subplots()
                df[bar_col].value_counts().plot(kind="bar", ax=ax, color="skyblue")
                ax.set_title(f"Bar Chart of {bar_col}")
                ax.set_xlabel(bar_col)
                ax.set_ylabel("Frequency")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=4)  # Rotate x-axis labels vertically
                st.pyplot(fig)
        else:
            st.write("Sorry, I couldn't understand the question.")

# Main Function to Select Pages
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Traffic Congestion Prediction", "Traffic Management Chatbot"])

    if page == "Traffic Congestion Prediction":
        data = preprocess_data('smart_traffic_management_dataset.csv')
        x, y = create_sequences(data, SEQ_LENGTH)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

        model = build_lstm_model(input_shape=(SEQ_LENGTH, x_train.shape[2]))
        model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

        run_dashboard_with_filter(model, x_test)

    elif page == "Traffic Management Chatbot":
        df = load_dataset()
        chatbot_page(df)

if __name__ == "__main__":
    main()
