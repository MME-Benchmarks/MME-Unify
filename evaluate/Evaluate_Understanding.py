#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

def calculate_accuracy(predictions: List[Dict]) -> Tuple[float, Dict[str, float]]:
    total_correct = 0
    total_count = len(predictions)  
    
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred in predictions:
        category = pred.get('category', 'unknown')
        category_stats[category]['total'] += 1
        
        if pred.get('output') and pred.get('output') == pred.get('answer'):
            total_correct += 1
            category_stats[category]['correct'] += 1
    
    overall_accuracy = total_correct / total_count if total_count > 0 else 0
    
    category_accuracy = {}
    for category, stats in category_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        category_accuracy[category] = accuracy
    
    return overall_accuracy, category_accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate result.')
    parser.add_argument('input_file', help='Path of input file')
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print(f"Can't find the file. '{args.input_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error：'{args.input_file}' Not a vaild file")
        return
    
    overall_acc, category_acc = calculate_accuracy(predictions)
    
    print("\nAccuracy:")
    for category, acc in sorted(category_acc.items()):
        print(f"{category}: {acc:.2%}")

if __name__ == "__main__":
    main()