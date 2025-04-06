import json
import os
from typing import Dict, List, Tuple
import torch
import lpips
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.nn.functional import mse_loss, interpolate
from transformers import CLIPProcessor, CLIPModel
import math
from collections import defaultdict

class ImageEvaluator:
    def __init__(self):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self.clip_model = CLIPModel.from_pretrained("clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("clip-vit-large-patch14")
        self.base_path = "/data/xwl/xwl_data/decode_images"
        
    def load_and_preprocess_image(self, image_path: str, is_output: bool = False) -> torch.Tensor:
        if not image_path:
            return None
            
        full_path = image_path if is_output else os.path.join(self.base_path, image_path.lstrip('/'))
        
        if not os.path.exists(full_path):
            print(f"Warning: Image path does not exist: {full_path}")
            return None
            
        try:
            img = Image.open(full_path).convert('RGB')
            return transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image {full_path}: {str(e)}")
            return None

    def resize_images(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if img1.shape != img2.shape:
            target_size = (
                min(img1.shape[2], img2.shape[2]),
                min(img1.shape[3], img2.shape[3])
            )
            
            img1 = interpolate(img1, size=target_size, mode='bilinear', align_corners=False)
            img2 = interpolate(img2, size=target_size, mode='bilinear', align_corners=False)
            
        return img1, img2

    def calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1, img2 = self.resize_images(img1, img2)
        mse = mse_loss(img1, img2).item()
        if mse == 0:
            return float('inf')
        return 20 * math.log10(1.0 / math.sqrt(mse))

    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1, img2 = self.resize_images(img1, img2)
        with torch.no_grad():
            return self.lpips_model(img1, img2).item()

    def calculate_clip_similarity(self, img: torch.Tensor, text: str = None) -> float:
        with torch.no_grad():
            img_pil = transforms.ToPILImage()(img.squeeze(0))
            
            if text:
                inputs = self.clip_processor(
                    images=img_pil,
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    max_length=77,  
                    truncation=True  
                ).to(self.device)
                
                outputs = self.clip_model(**inputs)
                
                logits_per_image = outputs.logits_per_image
                similarity = logits_per_image[0][0].item()
                return similarity
            else:
                inputs = self.clip_processor(
                    images=img_pil,
                    return_tensors="pt"
                ).to(self.device)

                image_features = self.clip_model.get_image_features(**inputs)
                return image_features

    def evaluate_reconstruction(self, output_path: str, target_path: str) -> Dict:
        output_img = self.load_and_preprocess_image(output_path, is_output=True)
        target_img = self.load_and_preprocess_image(target_path, is_output=False)
        
        if output_img is None or target_img is None:
            return None
            
        return {
            'psnr': self.calculate_psnr(output_img, target_img),
            'lpips': self.calculate_lpips(output_img, target_img)
        }

    def evaluate_editing(self, output_path: str, target_path: str, prompt: str) -> Dict:
        output_img = self.load_and_preprocess_image(output_path, is_output=True)
        target_img = self.load_and_preprocess_image(target_path, is_output=False)
        
        if output_img is None or target_img is None:
            return None
            
        output_img, target_img = self.resize_images(output_img, target_img)
            
        output_features = self.calculate_clip_similarity(output_img)
        target_features = self.calculate_clip_similarity(target_img)
        clip_i = torch.cosine_similarity(output_features, target_features, dim=1).item()
        
        clip_t = self.calculate_clip_similarity(output_img, prompt)
        
        return {
            'clip_i': clip_i,
            'clip_t': clip_t
        }

    def evaluate_generation(self, output_path: str, target_path: str, prompt: str) -> Dict:
        output_img = self.load_and_preprocess_image(output_path, is_output=True)
        target_img = self.load_and_preprocess_image(target_path, is_output=False)
        
        if output_img is None or target_img is None:
            return None
            
        output_img, target_img = self.resize_images(output_img, target_img)
            
        output_features = self.calculate_clip_similarity(output_img)
        target_features = self.calculate_clip_similarity(target_img)
        clip_i = torch.cosine_similarity(output_features, target_features, dim=1).item()
        
        clip_t = self.calculate_clip_similarity(output_img, prompt)
        
        lpips_d = self.calculate_lpips(output_img, target_img)
        
        return {
            'clip_i': clip_i,
            'clip_t': clip_t,
            'lpips_d': lpips_d
        }

def main():
    evaluator = ImageEvaluator()
    
    with open('/data/xwl/xwl_code/Unify_Benchmark/results/Generation/OmniGen/result.json', 'r') as f:
        data = json.load(f)
    
    results = defaultdict(list)
    
    total_samples = len(data)
    processed_samples = 0
    
    for item in data:
        category = item['category']

        panduan = item.get('error', 0)
        if panduan != 0:
            continue


        output_path = item['output']['output_image']

        # Skip if 'output' field is missing or empty
        if not output_path:
            processed_samples += 1
            continue
                    
        try:
            if category == 'Fine-Grained_Image_Reconstruction':
                target_path = item['data']['image']
                result = evaluator.evaluate_reconstruction(output_path, target_path)
                if result:
                    results[category].append(result)
                    
            elif category == 'Text-Image_Editing':
                target_path = item['data']['edited_image']
                prompt = item['Text_Prompt']
                result = evaluator.evaluate_editing(output_path, target_path, prompt)
                if result:
                    results[category].append(result)
                    
            elif category == 'Text-Image_Generation':
                target_path = item['data']['image']
                prompt = item['Text_Prompt']
                result = evaluator.evaluate_generation(output_path, target_path, prompt)
                if result:
                    results[category].append(result)
        except Exception as e:
            print(f"Error processing sample: {e}")
            print(f"Category: {category}")
            print(f"Output path: {output_path}")
        
        processed_samples += 1
        if processed_samples % 100 == 0:
            print(f"Processed {processed_samples}/{total_samples} samples")
    
    print("\nAverage Metrics per Category:")
    print("-" * 50)
    
    for category, category_results in results.items():
        if not category_results:
            continue
            
        print(f"\n{category}:")
        print(f"Total processed samples: {len(category_results)}")
        metrics = {}
        for metric in category_results[0].keys():
            avg_value = np.mean([r[metric] for r in category_results])
            metrics[metric] = avg_value
            print(f"{metric}: {avg_value:.4f}")

if __name__ == "__main__":
    main()
