import json
import torch
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import argparse

def load_image(image_path: str) -> Optional[Image.Image]:
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def calculate_clip_similarity(model, processor, image1_path: str, image2_path: str) -> float:
    try:
        image1 = load_image(image1_path)
        image2 = load_image(image2_path)
        
        if image1 is None or image2 is None:
            return 0.0

        inputs1 = processor(images=image1, return_tensors="pt")
        inputs2 = processor(images=image2, return_tensors="pt")

        inputs1 = {k: v.to(model.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}

        with torch.no_grad():
            features1 = model.get_image_features(**inputs1)
            features2 = model.get_image_features(**inputs2)

        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)

        similarity = (features1 @ features2.T).item()
        
        return similarity
    except Exception as e:
        print(f"Error calculating similarity between {image1_path} and {image2_path}: {e}")
        return 0.0

def evaluate_text_answer(item: Dict) -> Tuple[bool, bool]:

    if 'output' not in item or not isinstance(item['output'], dict):
        return False, False
        
    output = item['output']
    prediction = output.get('selected_answer')
    
    if not prediction or (isinstance(prediction, str) and prediction.strip() == ''):
        return False, False
        
    ground_truth = item.get('answer', '').strip()
    if not ground_truth:
        return False, False
        
    return True, prediction.strip() == ground_truth

def evaluate_image_prediction(item: Dict, model, processor, base_path: str) -> Tuple[bool, bool]:

    if 'output' not in item or not isinstance(item['output'], dict):
        return False, False
        
    output = item['output']
    output_image = output.get('difference_image')
    
    if not output_image or not os.path.exists(output_image):
        return False, False
        
    data = item.get('data', {})
    reference_images = {
        "img_diff_a": data.get("img_diff_a", ""),
        "img_diff_a_negative1": data.get("img_diff_a_negative1", ""),
        "img_diff_a_negative2": data.get("img_diff_a_negative2", ""),
        "img_diff_a_negative3": data.get("img_diff_a_negative3", "")
    }

    similarities = {}
    valid_similarities = True
    
    for ref_name, ref_path in reference_images.items():
        if ref_path:
            full_ref_path = os.path.join(base_path, ref_path)
            if os.path.exists(full_ref_path):
                similarity = calculate_clip_similarity(model, processor, output_image, full_ref_path)
                similarities[ref_name] = similarity
            else:
                valid_similarities = False
                break

    if valid_similarities and similarities:
        max_sim_key = max(similarities, key=similarities.get)
        print(f"Sample {item.get('id', 'unknown')} similarities: {similarities}")
        return True, max_sim_key == "img_diff_a"
        
    return False, False

def evaluate_results(input_json: str, base_path: str) -> Dict:
    
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    print(f"Loading results from {input_json}")
    with open(input_json, 'r') as f:
        results = json.load(f)

    total_samples = len(results)
    
    text_metrics = {
        'valid_samples': 0,
        'correct': 0,
        'skipped': 0
    }
    
    image_metrics = {
        'valid_samples': 0,
        'correct': 0,
        'skipped': 0
    }
    
    combined_metrics = {
        'valid_samples': 0,
        'correct': 0,
        'skipped': 0
    }
    
    for item in tqdm(results, desc="Evaluating samples"):
        # Evaluate text answer
        text_valid, text_correct = evaluate_text_answer(item)
        if text_valid:
            text_metrics['valid_samples'] += 1
            if text_correct:
                text_metrics['correct'] += 1
        else:
            text_metrics['skipped'] += 1
            
        # Evaluate image prediction
        image_valid, image_correct = evaluate_image_prediction(item, model, processor, base_path)
        if image_valid:
            image_metrics['valid_samples'] += 1
            if image_correct:
                image_metrics['correct'] += 1
        else:
            image_metrics['skipped'] += 1

        # Evaluate combined accuracy
        if text_valid and image_valid:
            combined_metrics['valid_samples'] += 1
            if text_correct and image_correct:
                combined_metrics['correct'] += 1
        else:
            combined_metrics['skipped'] += 1

    text_accuracy = (text_metrics['correct'] / total_samples * 100)
    image_accuracy = (image_metrics['correct'] / total_samples * 100)
    combined_accuracy = (combined_metrics['correct'] / total_samples * 100)

    evaluation_results = {
        'total_samples': total_samples,
        'text_metrics': {
            'valid_samples': text_metrics['valid_samples'],
            'correct': text_metrics['correct'],
            'skipped': text_metrics['skipped'],
            'accuracy': text_accuracy
        },
        'image_metrics': {
            'valid_samples': image_metrics['valid_samples'],
            'correct': image_metrics['correct'],
            'skipped': image_metrics['skipped'],
            'accuracy': image_accuracy
        },
        'combined_metrics': {
            'valid_samples': combined_metrics['valid_samples'],
            'correct': combined_metrics['correct'],
            'skipped': combined_metrics['skipped'],
            'accuracy': combined_accuracy
        }
    }

    print("\nEvaluation Results:")
    print(f"Total samples: {total_samples}")
    print("\nText Metrics:")
    print(f"  Valid samples: {text_metrics['valid_samples']}")
    print(f"  Correct: {text_metrics['correct']}")
    print(f"  Skipped: {text_metrics['skipped']}")
    print(f"  Accuracy: {text_accuracy:.2f}%")
    print("\nImage Metrics:")
    print(f"  Valid samples: {image_metrics['valid_samples']}")
    print(f"  Correct: {image_metrics['correct']}")
    print(f"  Skipped: {image_metrics['skipped']}")
    print(f"  Accuracy: {image_accuracy:.2f}%")
    print("\nCombined Metrics:")
    print(f"  Valid samples: {combined_metrics['valid_samples']}")
    print(f"  Correct: {combined_metrics['correct']}")
    print(f"  Skipped: {combined_metrics['skipped']}")
    print(f"  Accuracy: {combined_accuracy:.2f}%")
    
    return evaluation_results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate spot difference task results")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to the input JSON file containing model results")
    parser.add_argument("--base_path", type=str, 
                        default="/data/xwl/xwl_data/decode_images",
                        help="Base path for reference images")
    parser.add_argument("--output_json", type=str, 
                        default="evaluation_results.json",
                        help="Path to save evaluation results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run evaluation
    results = evaluate_results(args.input_json, args.base_path)
    
    # Save results
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_json}")