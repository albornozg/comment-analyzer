import streamlit as st
import json
import pandas as pd
import re
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import subprocess
import os
import tempfile

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Streamlit app
st.title("YouTube Comments Sentiment Analysis")
st.write("Enter a YouTube video URL to analyze the sentiment of its comments.")

# Input URL
url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=VIDEOID")

# Function to run yt-dlp and fetch comments
def fetch_comments(url, temp_dir):
    try:
        # Generate a temporary output file path
        output_file = os.path.join(temp_dir, "video_info.json")
        # Run yt-dlp command
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-comments",
            "--no-warnings",
            "-o",
            output_file,
            url
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Find the generated JSON file
        for f in os.listdir(temp_dir):
            if f.endswith(".info.json"):
                return os.path.join(temp_dir, f)
        return None
    except subprocess.CalledProcessError as e:
        st.error(f"Error fetching comments: {e.stderr}")
        return None

# Function to analyze comments
def analyze_comments(json_file):
    try:
        # Load JSON
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract comments
        comments = [
            {"text": c.get("text", ""), "likes": c.get("like_count", 0)}
            for c in data.get("comments", [])
        ]
        if not comments:
            return None, None, None
        
        df = pd.DataFrame(comments)
        
        # Sentiment analysis
        sid = SentimentIntensityAnalyzer()
        df["sentiment"] = df["text"].apply(lambda t: sid.polarity_scores(t)["compound"])
        df["bucket"] = pd.cut(df["sentiment"], [-1, -0.05, 0.05, 1], labels=["neg", "neu", "pos"])
        
        # Top words
        stop = set("the and for you this that with have not are but was from your they just like what when why how she he him her has had were did out get got into over more very than then".split())
        tokens = (w.lower() for t in df["text"] for w in re.findall(r"[a-zA-Z]{3,}", t))
        top = Counter(w for w in tokens if w not in stop).most_common(25)
        
        return df, df["bucket"].value_counts(normalize=True).to_dict(), top
    except Exception as e:
        st.error(f"Error analyzing comments: {str(e)}")
        return None, None, None

# Process when URL is provided
if url:
    with st.spinner("Fetching and analyzing comments..."):
        # Use temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = fetch_comments(url, temp_dir)
            if json_file:
                # Provide JSON download link
                with open(json_file, "rb") as f:
                    st.download_button(
                        label="Download JSON File",
                        data=f,
                        file_name="video_info.json",
                        mime="application/json"
                    )
                
                # Analyze comments
                df, sentiment_dist, top_words = analyze_comments(json_file)
                if df is not None:
                    st.subheader("Analysis Results")
                    st.write(f"**Total Comments**: {len(df)}")
                    st.write("**Sentiment Distribution**:")
                    st.json(sentiment_dist)
                    st.write("**Top 25 Words**:")
                    st.json(top_words)
                else:
                    st.warning("No comments found or analysis failed.")
            else:
                st.error("Failed to fetch comments. Please check the URL or try again.")
