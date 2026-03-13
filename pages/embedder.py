# Library imports
import streamlit as st
import pandas as pd
import tiktoken
import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH (so "utils", "classes" can be imported)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.utils import normalize_text


from utils.page_components import add_common_page_elements

from classes.embeddings import Embeddings


def get_format(path):
    file_format = "." + path.split(".")[-1]
    if file_format == ".xlsx":
        read_func = pd.read_excel
    elif file_format == ".csv":
        read_func = pd.read_csv
    else:
        raise ValueError(f"File format {file_format} not supported.")
        print("unected file: " + path)
    return file_format, read_func


def embed(file_path, embeddings):
    file_format, read_func = get_format(file_path)

    df = read_func(file_path)
    embedding_path = file_path.replace("describe", "embeddings").replace(
        file_format, ".parquet"
    )

    st.write(f"Embedding file: {file_path}")
    # Check if the content of user exceeds max token length
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df["user_tokens"] = df["user"].apply(lambda x: len(tokenizer.encode(x)))
    df = df[df.user_tokens < 8192]
    df = df.drop("user_tokens", axis=1)

    # Check for common errors in the text
    df["user"] = df["user"].apply(lambda x: normalize_text(x))

    df["user_embedded"] = df["user"].apply(
        lambda x: str(embeddings.return_embedding(x))
    )

    directory = os.path.dirname(embedding_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    st.write("Embedded file:")
    st.write(df)
    df.to_parquet(embedding_path, index=False)


sidebar_container = add_common_page_elements()

st.divider()

embeddings = Embeddings()

# Get list of files in data/describe folder
describe_folder = "data/describe"
available_files = []

for root, dirs, files in os.walk(describe_folder):
    for file in files:
        if not file.endswith(".DS_Store"):
            # Get relative path from describe folder
            rel_path = os.path.relpath(os.path.join(root, file), describe_folder)
            available_files.append(rel_path)

if not available_files:
    st.warning("No files found in data/describe folder")
else:
    # File selection dropdown
    selected_file = st.selectbox(
        "Select a file to embed",
        options=available_files,
        index=0
    )
    
    # Show selected file path
    full_path = os.path.join(describe_folder, selected_file)
    st.info(f"Selected file: {full_path}")
    
    # Display the file contents
    try:
        file_format, read_func = get_format(full_path)
        df = read_func(full_path)
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
    
    # Check if file is already embedded
    try:
        embedding_path = full_path.replace("describe", "embeddings").replace(
            file_format, ".parquet"
        )
        is_embedded = os.path.exists(embedding_path)
    except:
        is_embedded = False
    
    # Embed / Re-embed button
    button_label = "Re-embed File" if is_embedded else "Embed File"
    if st.button(button_label, type="primary"):
        st.write(f"Starting to embed {selected_file}...")
        try:
            embed(full_path, embeddings)
            st.success(f"Successfully embedded {selected_file}!")
        except Exception as e:
            st.error(f"Error embedding file: {str(e)}")
