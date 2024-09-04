import pandas as pd
import streamlit as st
import psycopg2
import pickle


with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


connection = psycopg2.connect(
    host="localhost",
    port="5432",
    database="youtube_data",
    user="postgres",
    password="****"
)

writer = connection.cursor()


query1 = "SELECT * FROM final_youtube_data"
writer.execute(query1)
data = writer.fetchall()
df = pd.DataFrame(data)

query2 = """
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'final_youtube_data' 
    ORDER BY ordinal_position
"""
writer.execute(query2)
columns = writer.fetchall()
df.columns = [col[0] for col in columns]


connection.close()

def get_query_cluster(query):
    query_vector = vectorizer.transform([query])
    query_pca = pca.transform(query_vector.toarray())
    cluster = kmeans.predict(query_pca).item()
    return cluster

def recommend_videos_by_query(query):
    cluster = get_query_cluster(query)
    results = df[df['Cluster'] == cluster]
    return results

def recommend_videos_by_channel(channel):
    results = df[df['Channel_Title'] == channel]
    return results

def recommendation(query=None, channel=None):
    if query and channel:
        results = recommend_videos_by_channel(channel)
        cluster = get_query_cluster(query)
        results = results[results['Cluster'] == cluster]
    elif query:
        results = recommend_videos_by_query(query)
    elif channel:
        results = recommend_videos_by_channel(channel)
    else:
        results = pd.DataFrame()  
    return results

st.sidebar.image("You-Tube-1.png", width=200)

Channels = sorted(df['Channel_Title'].drop_duplicates().tolist())
Channel = st.sidebar.radio("Channels", Channels, index=None)

st.title("YouTube")
Query = st.text_input("Enter the query:")

if Query or Channel:
    recommended_videos = recommendation(query=Query, channel=Channel)
    if not recommended_videos.empty:
        for index, video in recommended_videos.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(video["Url"], use_column_width=True)
            with col2:
                a, b = st.columns([1, 2])
                with a:
                    st.write(f"Views: {video['View_Count']}")
                with b:
                    st.write(f"Likes: {video['Like_Count']}")
    else:
        st.write("No videos found for this query or channel.")
