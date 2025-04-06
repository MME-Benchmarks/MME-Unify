import json
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
import os

class VisualCoTEvaluator:
    def __init__(self, result_path, image_base_path="/data/xwl/xwl_data/decode_images"):
        self.result_path = result_path
        self.image_base_path = image_base_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.results = self.load_results()
        
    def load_results(self):
        with open(self.result_path, 'r') as f:
            return json.load(f)
    
    def calculate_clip_similarity(self, image1_path, image2_path):
        """Calculate CLIP-I similarity between two images"""
        try:
            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)
            
            inputs1 = self.processor(images=image1, return_tensors="pt").to(self.device)
            inputs2 = self.processor(images=image2, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image1_features = self.model.get_image_features(**inputs1)
                image2_features = self.model.get_image_features(**inputs2)
                
                image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
                image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)
                
                similarity = (image1_features @ image2_features.T).item()
                
            return similarity
        except Exception as e:
            print(f"Error calculating CLIP similarity: {e}")
            return 0.0
    
    def evaluate_step(self, data_step, output_step, step_idx, gt_images):
        """Evaluate a single step's outputs"""
        results = {
            'action': {
                'attempted': False,
                'correct': False
            },
            'location': {
                'attempted': False,
                'correct': False
            },
            'image': {
                'attempted': False,
                'correct': False
            }
        }
        
        if output_step is None:
            return results
        
        # Evaluate action
        if step_idx < len(data_step['Action']):
            results['action']['attempted'] = output_step.get('output_action') is not None
            results['action']['correct'] = (
                output_step.get('output_action') == data_step['Action'][step_idx]
            )
        
        # Evaluate location
        if step_idx < len(data_step['Coordinate']):
            output_loc = output_step.get('output_location')
            gt_loc = data_step['Coordinate'][step_idx]
            results['location']['attempted'] = output_loc is not None
            results['location']['correct'] = (
                output_loc is not None and 
                len(output_loc) == len(gt_loc) and 
                all(a == b for a, b in zip(output_loc, gt_loc))
            )
        
        # Evaluate image
        output_image = output_step.get('output_image')
        if output_image and gt_images:
            results['image']['attempted'] = True
            similarities = []
            
            for gt_image in gt_images:
                similarity = self.calculate_clip_similarity(output_image, gt_image)
                similarities.append(similarity)
            
            max_similarity_idx = np.argmax(similarities)
            results['image']['correct'] = (max_similarity_idx == step_idx)
            print(f"Step {step_idx} similarities: {similarities}")
        
        return results
    
    def evaluate_sample(self, sample):
        """Evaluate a single sample"""
        data = sample['data']
        outputs = sample.get('outputs', {})
        
        gt_images = []
        step_idx = 0
        while True:
            step_key = f'Step_{step_idx}'
            if step_key not in data:
                break
            gt_images.append(os.path.join(self.image_base_path, data[step_key]))
            step_idx += 1
        
        num_steps = len(data['Action'])
        step_results = []
        
        for i in range(num_steps):
            output_step = outputs.get(f'output_step_{i}')
            step_result = self.evaluate_step(data, output_step, i, gt_images)
            step_results.append(step_result)
        
        all_steps_attempted = all(
            r['action']['attempted'] and r['location']['attempted'] and r['image']['attempted']
            for r in step_results
        )
        
        all_steps_correct = all(
            r['action']['correct'] and r['location']['correct'] and r['image']['correct']
            for r in step_results
        )
        
        return {
            'subcategory': sample['subcategory'],
            'step_results': step_results,
            'all_steps_attempted': all_steps_attempted,
            'all_steps_correct': all_steps_correct
        }
    
    def calculate_accuracies(self):
        """Calculate various accuracy metrics"""
        total_samples = len(self.results)
        
        metrics = {
            'overall': {
                'total_samples': total_samples,
                'step_metrics': defaultdict(lambda: {
                    'total': total_samples,
                    'action': {'attempted': 0, 'correct': 0, 'skipped': 0},
                    'location': {'attempted': 0, 'correct': 0, 'skipped': 0},
                    'image': {'attempted': 0, 'correct': 0, 'skipped': 0}
                }),
                'subcategory_metrics': defaultdict(lambda: {
                    'total': 0,
                    'attempted': 0,
                    'correct': 0,
                    'skipped': 0
                })
            }
        }
        
        # Process all samples
        for sample in self.results:
            evaluation = self.evaluate_sample(sample)
            subcategory = evaluation['subcategory']
            
            # Update subcategory counts
            metrics['overall']['subcategory_metrics'][subcategory]['total'] += 1
            if evaluation['all_steps_attempted']:
                metrics['overall']['subcategory_metrics'][subcategory]['attempted'] += 1
                if evaluation['all_steps_correct']:
                    metrics['overall']['subcategory_metrics'][subcategory]['correct'] += 1
            else:
                metrics['overall']['subcategory_metrics'][subcategory]['skipped'] += 1
            
            # Update step-wise metrics
            for step_idx, step_result in enumerate(evaluation['step_results']):
                step_metrics = metrics['overall']['step_metrics'][f'step_{step_idx}']
                
                for aspect in ['action', 'location', 'image']:
                    if step_result[aspect]['attempted']:
                        step_metrics[aspect]['attempted'] += 1
                        if step_result[aspect]['correct']:
                            step_metrics[aspect]['correct'] += 1
                    else:
                        step_metrics[aspect]['skipped'] += 1
        
        # Calculate final metrics
        results = {
            'overall': {
                'total_samples': total_samples,
                'step_accuracies': {},
                'subcategory_accuracies': {},
                'average_accuracies': {}  
            }
        }
        
        # Calculate step-wise accuracies
        for step, data in metrics['overall']['step_metrics'].items():
            results['overall']['step_accuracies'][step] = {
                aspect: {
                    'total': data['total'],
                    'attempted': aspect_data['attempted'],
                    'attempted_rate': aspect_data['attempted'] / total_samples * 100,
                    'correct': aspect_data['correct'],
                    'accuracy': aspect_data['correct'] / total_samples * 100,
                    'skipped': aspect_data['skipped'],
                    'skip_rate': aspect_data['skipped'] / total_samples * 100
                }
                for aspect, aspect_data in data.items()
                if aspect in ['action', 'location', 'image']
            }
        
        # Calculate subcategory accuracies
        for subcategory, data in metrics['overall']['subcategory_metrics'].items():
            results['overall']['subcategory_accuracies'][subcategory] = {
                'total': data['total'],
                'attempted': data['attempted'],
                'attempted_rate': data['attempted'] / data['total'] * 100 if data['total'] > 0 else 0,
                'correct': data['correct'],
                'accuracy': data['correct'] / data['total'] * 100 if data['total'] > 0 else 0,
                'skipped': data['skipped'],
                'skip_rate': data['skipped'] / data['total'] * 100 if data['total'] > 0 else 0
            }
        
        total_action_steps = 0
        total_location_steps = 0
        total_image_steps = 0
        
        correct_action_steps = 0
        correct_location_steps = 0
        correct_image_steps = 0
        
        for step, data in metrics['overall']['step_metrics'].items():
            total_action_steps += data['total']
            total_location_steps += data['total']
            total_image_steps += data['total']
            
            correct_action_steps += data['action']['correct']
            correct_location_steps += data['location']['correct']
            correct_image_steps += data['image']['correct']
        
        avg_action_accuracy = (correct_action_steps / total_action_steps * 100) if total_action_steps > 0 else 0
        avg_location_accuracy = (correct_location_steps / total_location_steps * 100) if total_location_steps > 0 else 0
        avg_image_accuracy = (correct_image_steps / total_image_steps * 100) if total_image_steps > 0 else 0
        
        overall_avg_accuracy = (avg_action_accuracy + avg_location_accuracy + avg_image_accuracy) / 3
        
        results['overall']['average_accuracies'] = {
            'action': {
                'avg_accuracy': avg_action_accuracy,
                'correct_steps': correct_action_steps,
                'total_steps': total_action_steps
            },
            'location': {
                'avg_accuracy': avg_location_accuracy,
                'correct_steps': correct_location_steps,
                'total_steps': total_location_steps
            },
            'image': {
                'avg_accuracy': avg_image_accuracy,
                'correct_steps': correct_image_steps,
                'total_steps': total_image_steps
            },
            'overall': {
                'avg_accuracy': overall_avg_accuracy
            }
        }
        
        return results

def main():
    evaluator = VisualCoTEvaluator("/data/xwl/xwl_code/Unify_Benchmark/results/VSP/SEED-14B/result.json")
    
    results = evaluator.calculate_accuracies()
    
    print("\nEvaluation Results:")
    print(f"Total samples: {results['overall']['total_samples']}")
    
    print("\nSubcategory Accuracies:")
    for subcategory, metrics in results['overall']['subcategory_accuracies'].items():
        print(f"\n{subcategory}:")
        print(f"  Total samples: {metrics['total']}")
        print(f"  Attempted: {metrics['attempted']} ({metrics['attempted_rate']:.2f}%)")
        print(f"  Correct: {metrics['correct']} ({metrics['accuracy']:.2f}%)")
        print(f"  Skipped: {metrics['skipped']} ({metrics['skip_rate']:.2f}%)")
    
    print("\nStep-wise Accuracies:")
    for step, aspects in results['overall']['step_accuracies'].items():
        print(f"\n{step}:")
        for aspect, metrics in aspects.items():
            print(f"  {aspect.capitalize()}:")
            print(f"    Attempted: {metrics['attempted']} ({metrics['attempted_rate']:.2f}%)")
            print(f"    Correct: {metrics['correct']} ({metrics['accuracy']:.2f}%)")
            print(f"    Skipped: {metrics['skipped']} ({metrics['skip_rate']:.2f}%)")
    
    print("\nOverall Accuracies (All Steps):")
    print(f"  Action Accuracy: {results['overall']['average_accuracies']['action']['avg_accuracy']:.2f}% (Correct Steps: {results['overall']['average_accuracies']['action']['correct_steps']}/{results['overall']['average_accuracies']['action']['total_steps']})")
    print(f"  Location Accuracy: {results['overall']['average_accuracies']['location']['avg_accuracy']:.2f}% (Correct Steps: {results['overall']['average_accuracies']['location']['correct_steps']}/{results['overall']['average_accuracies']['location']['total_steps']})")
    print(f"  Image Accuracy: {results['overall']['average_accuracies']['image']['avg_accuracy']:.2f}% (Correct Steps: {results['overall']['average_accuracies']['image']['correct_steps']}/{results['overall']['average_accuracies']['image']['total_steps']})")
    print(f"  Overall Average Accuracy: {results['overall']['average_accuracies']['overall']['avg_accuracy']:.2f}%")
    
    # Save results
    output_path = 'evaluation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()