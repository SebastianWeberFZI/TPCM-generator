#!/usr/bin/env python3
"""Main entry point for TPCM model generation."""

import argparse
import os

# Import from our modules
from .dag_model_generator import DAGModelGenerator
from .utils import random_name
from .utils import convert_to_tpcm

def main():
    """Main entry point for the PCM model generation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate PCM models")
    parser.add_argument(
        "--convert", "-t", action="store_true", help="Convert the output to TPCM format"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=10,
        help="Number of nodes for the DAG",
    )
    parser.add_argument(
        "--edge-probability",
        type=float,
        default=0.3,
        help="Probability to generate an edge to a node with higher number in the DAG",
    )
    args = parser.parse_args()

    model_name = f"generated_tpcm_{random_name('')}"
    os.makedirs(model_name, exist_ok=True)
    output_file = f"{model_name}/{model_name}.xml"

    generator = DAGModelGenerator()

    # Generate all model elements and create the complete model
    model, model_resource = generator.generate_complete_model(
        output_file,  # Pass the model name with directory prefix
        nodes=args.nodes,
        edge_probability=args.edge_probability
    )

    # Convert to TPCM if requested
    if args.convert:       
        # Create TPCM path with just the base model name (no directory prefix)
        tpcm_path = f"{model_name}/{model_name}.tpcm"
        print(f"Converting {output_file} to TPCM format: {tpcm_path}...")
        if convert_to_tpcm(output_file, tpcm_path):
            print(f"Model converted to TPCM format: {tpcm_path}")