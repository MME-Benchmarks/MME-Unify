import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm

def load_clip_model():
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def compute_text_similarity(model, processor, text1, text2_list):
    if not text1 or not text2_list or not all(text2_list):
        return None
        
    try:
        inputs = processor(
            text=[text1] + text2_list,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        text_features = model.get_text_features(**inputs)
        
        text1_feature = text_features[0]
        text2_features = text_features[1:]
        
        similarities = torch.nn.functional.cosine_similarity(
            text1_feature.unsqueeze(0),
            text2_features,
            dim=1
        )
        
        return similarities.tolist()
    except Exception as e:
        print(f"Error in computing text similarity: {e}")
        return None

def compute_image_similarity(model, processor, image1_path, image2_paths):
    try:
        image1 = Image.open(image1_path).convert('RGB')
        image2_list = [Image.open(path).convert('RGB') for path in image2_paths]
        
        inputs = processor(
            images=[image1] + image2_list,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        image_features = model.get_image_features(**inputs)
        
        image1_feature = image_features[0]
        image2_features = image_features[1:]
        
        similarities = torch.nn.functional.cosine_similarity(
            image1_feature.unsqueeze(0),
            image2_features,
            dim=1
        )
        
        return similarities.tolist()
    except Exception as e:
        print(f"Error in computing image similarity: {e}")
        return None

def evaluate_results(json_path, base_path):
    try:
        model, processor = load_clip_model()
        
        print(f"\nLoading results from {json_path}")
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        total_samples = len(results)
        print(f"Total samples: {total_samples}")
        
        metrics = {
            'text': {
                'total': total_samples,
                'attempted': 0,
                'skipped': 0,
                'correct': 0,
                'failed': 0  
            },
            'image': {
                'total': total_samples,
                'attempted': 0,
                'skipped': 0,
                'correct': 0,
                'failed': 0  
            },
            'combined': {
                'total': total_samples,
                'attempted': 0,
                'skipped': 0,
                'correct': 0,
                'failed': 0  
            }
        }
        
        print("\nDebug - First sample structure:")
        print(json.dumps(results[0], indent=2))
        
        for idx, sample in enumerate(tqdm(results, desc="Evaluating samples")):
            sample_metrics = {
                'text': {'attempted': False, 'correct': False, 'failed': False},
                'image': {'attempted': False, 'correct': False, 'failed': False}
            }
            

            required_fields = {
                'top_level': ['choice', 'answer', 'output', 'data'],
                'output': ['output_explanation', 'output_image']
            }
            
            missing_fields = []
            for field in required_fields['top_level']:
                if field not in sample:
                    missing_fields.append(field)
            
            if 'output' in sample:
                for field in required_fields['output']:
                    if field not in sample['output']:
                        missing_fields.append(f"output.{field}")
            
            if missing_fields:
                print(f"\nSample {idx} missing fields: {missing_fields}")
                continue
            
            try:
                choices = sample['choice']
                output_explanation = sample['output']['output_explanation']
                answer = sample['answer']
                
                if output_explanation and choices:
                    sample_metrics['text']['attempted'] = True
                    similarities = compute_text_similarity(model, processor, output_explanation, choices)
                    if similarities:
                        predicted_idx = similarities.index(max(similarities))
                        predicted_answer = chr(ord('A') + predicted_idx)
                        sample_metrics['text']['correct'] = (predicted_answer == answer)
                    else:
                        sample_metrics['text']['failed'] = True
            except Exception as e:
                print(f"\nError in text evaluation for sample {idx}: {e}")
                sample_metrics['text']['failed'] = True
            
            try:
                output_image = sample['output']['output_image']
                comparison_images = [
                    os.path.join(base_path, sample['data']['edited_image']),
                    os.path.join(base_path, sample['data']['fake_image1']),
                    os.path.join(base_path, sample['data']['fake_image2']),
                    os.path.join(base_path, sample['data']['fake_image3'])
                ]
                
                if output_image and all(os.path.exists(img) for img in comparison_images):
                    sample_metrics['image']['attempted'] = True
                    image_similarities = compute_image_similarity(model, processor, output_image, comparison_images)
                    if image_similarities:
                        sample_metrics['image']['correct'] = (image_similarities.index(max(image_similarities)) == 0)
                    else:
                        sample_metrics['image']['failed'] = True
            except Exception as e:
                print(f"\nError in image evaluation for sample {idx}: {e}")
                sample_metrics['image']['failed'] = True
            
            for task in ['text', 'image']:
                if sample_metrics[task]['attempted']:
                    metrics[task]['attempted'] += 1
                    if sample_metrics[task]['correct']:
                        metrics[task]['correct'] += 1
                elif sample_metrics[task]['failed']:
                    metrics[task]['failed'] += 1
                else:
                    metrics[task]['skipped'] += 1
            
            if sample_metrics['text']['attempted'] and sample_metrics['image']['attempted']:
                metrics['combined']['attempted'] += 1
                if sample_metrics['text']['correct'] and sample_metrics['image']['correct']:
                    metrics['combined']['correct'] += 1
            elif sample_metrics['text']['failed'] or sample_metrics['image']['failed']:
                metrics['combined']['failed'] += 1
            else:
                metrics['combined']['skipped'] += 1
        
        results_dict = {
            'total_samples': total_samples,
            'text_metrics': {
                'total': metrics['text']['total'],
                'attempted': metrics['text']['attempted'],
                'attempted_rate': metrics['text']['attempted'] / total_samples * 100,
                'correct': metrics['text']['correct'],
                'accuracy': metrics['text']['correct'] / total_samples * 100,
                'skipped': metrics['text']['skipped'],
                'skip_rate': metrics['text']['skipped'] / total_samples * 100,
                'failed': metrics['text']['failed'],
                'failure_rate': metrics['text']['failed'] / total_samples * 100
            },
            'image_metrics': {
                'total': metrics['image']['total'],
                'attempted': metrics['image']['attempted'],
                'attempted_rate': metrics['image']['attempted'] / total_samples * 100,
                'correct': metrics['image']['correct'],
                'accuracy': metrics['image']['correct'] / total_samples * 100,
                'skipped': metrics['image']['skipped'],
                'skip_rate': metrics['image']['skipped'] / total_samples * 100,
                'failed': metrics['image']['failed'],
                'failure_rate': metrics['image']['failed'] / total_samples * 100
            },
            'combined_metrics': {
                'total': metrics['combined']['total'],
                'attempted': metrics['combined']['attempted'],
                'attempted_rate': metrics['combined']['attempted'] / total_samples * 100,
                'correct': metrics['combined']['correct'],
                'accuracy': metrics['combined']['correct'] / total_samples * 100,
                'skipped': metrics['combined']['skipped'],
                'skip_rate': metrics['combined']['skipped'] / total_samples * 100,
                'failed': metrics['combined']['failed'],
                'failure_rate': metrics['combined']['failed'] / total_samples * 100
            }
        }
        
        print("\nEvaluation Results:")
        print(f"Total samples: {total_samples}")
        
        for metric_type in ['text', 'image', 'combined']:
            metrics = results_dict[f'{metric_type}_metrics']
            print(f"\n{metric_type.capitalize()} Metrics:")
            print(f"  Attempted: {metrics['attempted']} ({metrics['attempted_rate']:.2f}%)")
            print(f"  Correct: {metrics['correct']} ({metrics['accuracy']:.2f}%)")
            print(f"  Skipped: {metrics['skipped']} ({metrics['skip_rate']:.2f}%)")
            print(f"  Failed: {metrics['failed']} ({metrics['failure_rate']:.2f}%)")
        
        output_path = 'evaluation_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
        
        return results_dict
        
    except Exception as e:
        print(f"Error in evaluation process: {e}")
        return None

if __name__ == "__main__":
    json_path = "/data/xwl/xwl_code/Unify_Benchmark/results/Image_Explaning_and_Editing/MGM-7B/result.json"
    base_path = "/data/xwl/xwl_data/decode_images"
    
    print("\nStarting evaluation...")
    results = evaluate_results(json_path, base_path)
    if not results:
        print("Evaluation failed.")