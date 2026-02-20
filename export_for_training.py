import sqlite3
import numpy as np
import os

def export_vectors(db_path="vectors.db"):
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found. Ensure the Qt app has generated the database.")
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Retrieve all records
        cursor.execute("SELECT id, text_chunk, vector_blob FROM embeddings")
        rows = cursor.fetchall()

        texts = []
        embeddings = []

        for row in rows:
            row_id, text, blob = row
            
            # Convert BLOB back to numpy array (float32 matches text-embedding-004)
            # Each float is 4 bytes.
            vector = np.frombuffer(blob, dtype=np.float32)
            
            texts.append(text)
            embeddings.append(vector)

        # Convert list of vectors to a 2D matrix
        embedding_matrix = np.array(embeddings)

        print(f"Successfully loaded {len(texts)} chunks.")
        print(f"Embedding matrix shape: {embedding_matrix.shape}")
        
        # Example for 1B model training:
        # X = embedding_matrix
        # y = some_labels_or_target
        
        return texts, embedding_matrix

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Check current directory for vectors.db (Qt app output)
    # The Qt app might put it in the 'build' or root folder depending on run location
    possible_paths = ["vectors.db", "build/vectors.db", "build/Release/vectors.db"]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from {path}...")
            texts, matrix = export_vectors(path)
            if matrix.size > 0:
                print("\nSample Data:")
                print(f"Text: {texts[0][:50]}...")
                print(f"Vector (first 5 values): {matrix[0][:5]}")
            break
    else:
        print("Could not find vectors.db in current or build folders.")
