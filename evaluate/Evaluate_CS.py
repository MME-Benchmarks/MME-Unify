import json
import argparse
import torch
from PIL import Image
from pathlib import Path 
import os
from typing import Dict, List
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm

def load_image(image_path: str) -> Image.Image:
    """Load an image from path and convert to RGB."""
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def calculate_clip_similarity(model, processor, image1_path: str, image2_path: str) -> float:
    """Calculate CLIP-I similarity between two images."""
    try:
        # Load images
        image1 = load_image(image1_path)
        image2 = load_image(image2_path)
        
        if image1 is None or image2 is None:
            return 0.0

        # Process images
        inputs1 = processor(images=image1, return_tensors="pt")
        inputs2 = processor(images=image2, return_tensors="pt")

        # Move to device
        inputs1 = {k: v.to(model.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}

        # Get image features
        with torch.no_grad():
            features1 = model.get_image_features(**inputs1)
            features2 = model.get_image_features(**inputs2)

        # Normalize features
        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (features1 @ features2.T).item()
        
        return similarity
    except Exception as e:
        print(f"Error calculating similarity between {image1_path} and {image2_path}: {e}")
        return 0.0

def evaluate_results(args):
    """Evaluate the results from the QA model."""
    # Load CLIP model and processor
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    # Load results
    print(f"Loading results from {args.input_json}")
    with open(args.input_json, 'r') as f:
        results = json.load(f)

    total_samples = len(results)
    
    # Initialize counters
    metrics = {
        'text': {
            'total': total_samples,
            'attempted': 0, 
            'skipped': 0,    
            'correct': 0   
        },
        'image': {
            'total': total_samples,
            'attempted': 0,  
            'skipped': 0,    
            'correct': 0   
        },
        'combined': {
            'total': total_samples,
            'attempted': 0,  
            'skipped': 0,    
            'correct': 0   
        }
    }

    # Process each sample
    for sample in tqdm(results, desc="Evaluating samples"):
        # Initialize flags for current sample
        text_attempted = False
        text_correct = False
        image_attempted = False
        image_correct = False
        
        # Evaluate text answer
        if "output_choice" in sample.get("output", {}):
            text_attempted = True
            metrics['text']['attempted'] += 1
            if sample.get("answer") in sample["output"]["output_choice"]:
                text_correct = True
                metrics['text']['correct'] += 1
        else:
            metrics['text']['skipped'] += 1

        # Evaluate image similarity
        output_images = sample.get("output", {}).get("output_image", [])
        if output_images and os.path.exists(output_images):
            image_attempted = True
            metrics['image']['attempted'] += 1
            
            data = sample.get("data", {})
            similarities = {}
            
            # Calculate similarities with all reference images
            reference_images = {
                "image": data.get("image", ""),
                "fake_image1": data.get("fake_image1", ""),
                "fake_image2": data.get("fake_image2", ""),
                "fake_image3": data.get("fake_image3", "")
            }

            for ref_name, ref_path in reference_images.items():
                if ref_path:
                    full_ref_path = os.path.join(args.base_path, ref_path)
                    if os.path.exists(full_ref_path):
                        similarity = calculate_clip_similarity(model, processor, output_images, full_ref_path)
                        similarities[ref_name] = similarity
                    else:
                        print(f"Warning: Reference image not found: {full_ref_path}")

            # Check if the correct image has the highest similarity
            if similarities:
                max_sim_key = max(similarities, key=similarities.get)
                if max_sim_key == "image":
                    image_correct = True
                    metrics['image']['correct'] += 1
                print(f"Sample {sample.get('id', 'unknown')} similarities: {similarities}")
        else:
            metrics['image']['skipped'] += 1

        # Update combined metrics
        if text_attempted and image_attempted:
            metrics['combined']['attempted'] += 1
            if text_correct and image_correct:
                metrics['combined']['correct'] += 1
        else:
            metrics['combined']['skipped'] += 1

    # Calculate accuracies (using total_samples as denominator)
    results_dict = {
        "total_samples": total_samples,
        "text_metrics": {
            "total": metrics['text']['total'],
            "attempted": metrics['text']['attempted'],
            "skipped": metrics['text']['skipped'],
            "correct": metrics['text']['correct'],
            "accuracy": metrics['text']['correct'] / total_samples * 100,
            "attempt_rate": metrics['text']['attempted'] / total_samples * 100
        },
        "image_metrics": {
            "total": metrics['image']['total'],
            "attempted": metrics['image']['attempted'],
            "skipped": metrics['image']['skipped'],
            "correct": metrics['image']['correct'],
            "accuracy": metrics['image']['correct'] / total_samples * 100,
            "attempt_rate": metrics['image']['attempted'] / total_samples * 100
        },
        "combined_metrics": {
            "total": metrics['combined']['total'],
            "attempted": metrics['combined']['attempted'],
            "skipped": metrics['combined']['skipped'],
            "correct": metrics['combined']['correct'],
            "accuracy": metrics['combined']['correct'] / total_samples * 100,
            "attempt_rate": metrics['combined']['attempted'] / total_samples * 100
        }
    }

    # Print detailed results
    print("\nEvaluation Results:")
    print(f"Total samples: {total_samples}")
    
    print("\nText Metrics:")
    print(f"  Attempted: {metrics['text']['attempted']} ({metrics['text']['attempted']/total_samples*100:.2f}%)")
    print(f"  Skipped: {metrics['text']['skipped']}")
    print(f"  Correct: {metrics['text']['correct']}")
    print(f"  Accuracy: {results_dict['text_metrics']['accuracy']:.2f}%")

    print("\nImage Metrics:")
    print(f"  Attempted: {metrics['image']['attempted']} ({metrics['image']['attempted']/total_samples*100:.2f}%)")
    print(f"  Skipped: {metrics['image']['skipped']}")
    print(f"  Correct: {metrics['image']['correct']}")
    print(f"  Accuracy: {results_dict['image_metrics']['accuracy']:.2f}%")

    print("\nCombined Metrics:")
    print(f"  Attempted both: {metrics['combined']['attempted']} ({metrics['combined']['attempted']/total_samples*100:.2f}%)")
    print(f"  Skipped either: {metrics['combined']['skipped']}")
    print(f"  Both correct: {metrics['combined']['correct']}")
    print(f"  Combined accuracy: {results_dict['combined_metrics']['accuracy']:.2f}%")

    # Save results to file
    output_path = os.path.join(os.path.dirname(args.input_json), "evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model results")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to the input JSON file containing model results")
    parser.add_argument("--base_path", type=str, default="/data/xwl/xwl_data/decode_images",
                        help="Base path for reference images")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_results(args)
