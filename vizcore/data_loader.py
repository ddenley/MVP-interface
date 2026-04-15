import pandas as pd
import json
import math


def load_bubble_map_data(file_path="../data/bubble_layout.jsonl"):
    """
    Load main mapping data from a JSONL file.
    :param file_path: jsonl file path containing bubble data.
    :return: pd.DataFrame with bubble data.
    """
    bubbles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    bubbles.append(json.loads(line))
        return pd.DataFrame(bubbles)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading bubble map data: {e}")


def load_topic_info_data(file_path="../data/topic_info_spad-dataset-jsonl_sentence_clusters_"
                                   "junk_sentences_removed_semantic_chunked_w10_p95_embedded_chunks.parquet"):
    """
    Load topic information from a Parquet file.
    :param file_path: parquet file path containing topic information.
    :return: pd.DataFrame with topic information.
    """
    try:
        topic_info_df = pd.read_parquet(file_path, engine="pyarrow")
        return topic_info_df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading topic info data: {e}")


def pair_topic_label_data(bubble_df, topic_info_df):
    """
    Match topic labels to bubble data based on topic IDs.
    :param bubble_df: pd.DataFrame containing bubble data with 'id' column.
    :param topic_info_df:  pd.DataFrame containing topic information with 'topic_id' and 'topic_label' columns.
    :return: updated bubble_df with 'topic_label' and 'display_label' columns.
    """
    try:
        topic_label_map = dict(zip(topic_info_df['topic_id'], topic_info_df['topic_label']))
        bubble_df['topic_label'] = bubble_df['id'].map(topic_label_map)
        bubble_df['topic_label'] = bubble_df['topic_label'].fillna(bubble_df['id'].apply(lambda x: f"T{x}"))
        bubble_df['display_label'] = bubble_df['topic_label'].apply(lambda label: str(label) if label else "Unlabeled")
        return bubble_df
    except Exception as e:
        raise Exception(f"Error pairing topic labels: {e}")

def pair_topic_summary_data(bubble_df, topic_info_df):
    """
    Match topic summaries to bubble data based on topic IDs.
    :param bubble_df: pd.DataFrame containing bubble data with 'id' column.
    :param topic_info_df: pd.DataFrame containing topic information with 'topic_id' and 'summary' columns.
    :return: updated bubble_df with 'summary' column.
    """
    try:
        topic_summary_map = dict(zip(topic_info_df['topic_id'], topic_info_df['summary']))
        bubble_df['summary'] = bubble_df['id'].map(topic_summary_map)
        bubble_df['summary'] = bubble_df['summary'].fillna(bubble_df['id'].apply(lambda x: f"No summary available for topic {x}"))
        return bubble_df
    except Exception as e:
        raise Exception(f"Error pairing topic summaries: {e}")