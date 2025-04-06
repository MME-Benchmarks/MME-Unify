import json
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import Dict, List, Tuple
IMAGE_BASE_PATH="/data/xwl/xwl_data/decode_images"

class Evaluator:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.valid_choices = set(['A', 'B', 'C', 'D'])
        
    def calculate_metrics(self, data: List[Dict]) -> Dict[str, Dict]:
        total_samples = len(data)
        if total_samples == 0:
            return self._create_empty_metrics()
            
        metrics = {
            'choice': {
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
        
        for item in data:
            choice_result = self._evaluate_choice(item)
            self._update_metrics(metrics['choice'], choice_result)
            
            image_result = self._evaluate_image(item)
            self._update_metrics(metrics['image'], image_result)
            
            if choice_result['attempted'] and image_result['attempted']:
                metrics['combined']['attempted'] += 1
                if choice_result['correct'] and image_result['correct']:
                    metrics['combined']['correct'] += 1
            else:
                metrics['combined']['skipped'] += 1

        return self._calculate_final_metrics(metrics, total_samples)
    
    def _create_empty_metrics(self) -> Dict[str, Dict]:
        empty_metrics = {
            'total_samples': 0,
            'choice_metrics': {
                'total': 0, 'attempted': 0, 'skipped': 0,
                'correct': 0, 'accuracy': 0.0, 'attempt_rate': 0.0
            },
            'image_metrics': {
                'total': 0, 'attempted': 0, 'skipped': 0,
                'correct': 0, 'accuracy': 0.0, 'attempt_rate': 0.0
            },
            'combined_metrics': {
                'total': 0, 'attempted': 0, 'skipped': 0,
                'correct': 0, 'accuracy': 0.0, 'attempt_rate': 0.0
            }
        }
        return empty_metrics

    def _evaluate_choice(self, item: Dict) -> Dict[str, bool]:
        result = {'attempted': False, 'correct': False}
        
        pred = item.get('output', {}).get('output_choice', '')
        true = item.get('answer', '')
        
        if pred and pred in self.valid_choices:
            result['attempted'] = True
            result['correct'] = (pred == true)
            
        return result
    
    def _evaluate_image(self, item: Dict) -> Dict[str, bool]:
        result = {'attempted': False, 'correct': False}
        
        output_image = item.get('output', {}).get('output_image', '')
        if not output_image:
            return result
            
        try:
            aux_images = {
                'main': os.path.join(IMAGE_BASE_PATH, item['data']['image_Auxiliary_lines']),
                'neg1': os.path.join(IMAGE_BASE_PATH, item['data']['image_Auxiliary_lines_negative1']),
                'neg2': os.path.join(IMAGE_BASE_PATH, item['data']['image_Auxiliary_lines_negative2']),
                'neg3': os.path.join(IMAGE_BASE_PATH, item['data']['image_Auxiliary_lines_negative3'])
            }
            
            if all(os.path.exists(path) for path in aux_images.values()):
                result['attempted'] = True
                
                similarities = {}
                for name, path in aux_images.items():
                    similarities[name] = self.compute_clip_similarity(output_image, path)
                
                if similarities:
                    result['correct'] = similarities['main'] == max(similarities.values())
                    print(f"Sample similarities: {similarities}")
                    
        except Exception as e:
            print(f"Error evaluating image: {str(e)}")
            
        return result
    
    def _update_metrics(self, metrics: Dict, result: Dict[str, bool]) -> None:
        if result['attempted']:
            metrics['attempted'] += 1
            if result['correct']:
                metrics['correct'] += 1
        else:
            metrics['skipped'] += 1
            
    def _calculate_final_metrics(self, metrics: Dict, total_samples: int) -> Dict:
        results = {
            'total_samples': total_samples,
            'choice_metrics': {
                'total': metrics['choice']['total'],
                'attempted': metrics['choice']['attempted'],
                'skipped': metrics['choice']['skipped'],
                'correct': metrics['choice']['correct'],
                'accuracy': metrics['choice']['correct'] / total_samples * 100,
                'attempt_rate': metrics['choice']['attempted'] / total_samples * 100
            },
            'image_metrics': {
                'total': metrics['image']['total'],
                'attempted': metrics['image']['attempted'],
                'skipped': metrics['image']['skipped'],
                'correct': metrics['image']['correct'],
                'accuracy': metrics['image']['correct'] / total_samples * 100,
                'attempt_rate': metrics['image']['attempted'] / total_samples * 100
            },
            'combined_metrics': {
                'total': metrics['combined']['total'],
                'attempted': metrics['combined']['attempted'],
                'skipped': metrics['combined']['skipped'],
                'correct': metrics['combined']['correct'],
                'accuracy': metrics['combined']['correct'] / total_samples * 100,
                'attempt_rate': metrics['combined']['attempted'] / total_samples * 100
            }
        }
        return results
    
    def compute_clip_similarity(self, image1_path: str, image2_path: str) -> float:
        try:
            if not os.path.exists(image1_path) or not os.path.exists(image2_path):
                return 0.0

            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)

            inputs1 = self.processor(images=image1, return_tensors="pt").to(self.device)
            inputs2 = self.processor(images=image2, return_tensors="pt").to(self.device)

            with torch.no_grad():
                features1 = self.model.get_image_features(**inputs1)
                features2 = self.model.get_image_features(**inputs2)

            similarity = torch.nn.functional.cosine_similarity(features1, features2).item()
            
            return similarity
        except Exception as e:
            print(f"Error computing similarity for {image1_path} and {image2_path}: {str(e)}")
            return 0.0

def main():
    evaluator = Evaluator()
    
    try:
        with open('/data/xwl/xwl_code/Unify_Benchmark/results/Math_Geo/MGM-7B/result.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        return
    
    metrics = evaluator.calculate_metrics(data)
    
    print("\nEvaluation Results:")
    print(f"Total samples: {metrics['total_samples']}")
    
    print("\nChoice Metrics:")
    print(f"  Attempted: {metrics['choice_metrics']['attempted']} ({metrics['choice_metrics']['attempt_rate']:.2f}%)")
    print(f"  Skipped: {metrics['choice_metrics']['skipped']}")
    print(f"  Correct: {metrics['choice_metrics']['correct']}")
    print(f"  Accuracy: {metrics['choice_metrics']['accuracy']:.2f}%")

    print("\nImage Metrics:")
    print(f"  Attempted: {metrics['image_metrics']['attempted']} ({metrics['image_metrics']['attempt_rate']:.2f}%)")
    print(f"  Skipped: {metrics['image_metrics']['skipped']}")
    print(f"  Correct: {metrics['image_metrics']['correct']}")
    print(f"  Accuracy: {metrics['image_metrics']['accuracy']:.2f}%")

    print("\nCombined Metrics:")
    print(f"  Attempted both: {metrics['combined_metrics']['attempted']} ({metrics['combined_metrics']['attempt_rate']:.2f}%)")
    print(f"  Skipped either: {metrics['combined_metrics']['skipped']}")
    print(f"  Both correct: {metrics['combined_metrics']['correct']}")
    print(f"  Combined accuracy: {metrics['combined_metrics']['accuracy']:.2f}%")
    
    output_path = 'evaluation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()