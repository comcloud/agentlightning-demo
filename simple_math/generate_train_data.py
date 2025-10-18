#!/usr/bin/env python3
"""
Generate training data for the simple math agent.
"""

import os
import json
from typing import Any, Dict, List
import random


def generate_math_dataset(size: int = 100) -> List[Dict[str, Any]]:
    """
    Generate a simple math dataset for training.
    
    Args:
        size: Number of samples to generate
        
    Returns:
        List of math problem samples
    """
    samples = []
    operations = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]

    for i in range(size):
        # Generate random numbers
        a = random.randint(1, 100)
        b = random.randint(1, 100)

        # Select random operation
        op_symbol, op_func = random.choice(operations)

        # Create question and answer
        question = f"What is {a} {op_symbol} {b}?"
        answer = float(op_func(a, b))

        samples.append({
            "question": question,
            "answer": answer,
            "operation": op_symbol
        })

    return samples


def save_dataset_as_parquet(dataset: List[Dict[str, Any]], file_path: str):
    """
    Save dataset as parquet file.
    
    Args:
        dataset: List of samples
        file_path: Path to save the parquet file
    """
    try:
        import pandas as pd
        df = pd.DataFrame(dataset)
        df.to_parquet(file_path)
        print(f"Dataset saved to {file_path}")
    except ImportError:
        print("pandas not available, saving as JSON instead")
        # Fallback to JSON
        json_path = file_path.replace('.parquet', '.json')
        with open(json_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {json_path}")


def main():
    # Create data directory if it doesn't exist
    # data_dir = os.path.expanduser("~/data/rlhf/gsm8k")
    data_dir = os.path.expanduser("./data")
    os.makedirs(data_dir, exist_ok=True)

    # Generate training dataset
    print("Generating training dataset...")
    train_dataset = generate_math_dataset(100)

    # Save training dataset
    train_file = os.path.join(data_dir, "train.parquet")
    save_dataset_as_parquet(train_dataset, train_file)

    # Generate validation dataset
    print("Generating validation dataset...")
    val_dataset = generate_math_dataset(20)

    # Save validation dataset
    val_file = os.path.join(data_dir, "val.parquet")
    save_dataset_as_parquet(val_dataset, val_file)

    print(f"Generated {len(train_dataset)} training samples and {len(val_dataset)} validation samples")


if __name__ == "__main__":
    main()
