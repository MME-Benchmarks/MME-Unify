import json
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from scipy import linalg
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


def extract_keyframes(video_path: str, 
                      output_dir: str = None, 
                      num_frames: int = 16, 
                      min_scene_change: float = 20.0,
                      use_uniform_fallback: bool = True) -> List[str]:
    """
    Extract keyframes from an MP4 video. Try to detect scene changes. If insufficient keyframes are detected, use uniform sampling to supplement.

    Args:
    video_path (str): Path to the MP4 file
    output_dir (str, optional): Directory to save keyframes. If None, a folder will be created in the same directory as the video
    num_frames (int, optional): Number of frames to extract, default is 16
    min_scene_change (float, optional): Minimum scene change threshold, the larger the value, the fewer scene changes are detected. Default is 20.0
    use_uniform_fallback (bool, optional): If insufficient keyframes are detected, whether to use uniform sampling to supplement. Default is True

    Returns:
    List[str]: List of extracted keyframe image file paths
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_dir is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(video_dir, f"{video_name}_keyframes")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count == 0:
        raise ValueError(f"The number of video frames is 0: {video_path}")
    
    print(f"Frame rate of video {video_path} : {fps}, total frame count: {frame_count}")
    
    prev_frame = None
    keyframes = []
    keyframe_indices = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            keyframe_path = os.path.join(output_dir, f"keyframe_{frame_idx:04d}.jpg")
            cv2.imwrite(keyframe_path, frame)
            keyframes.append(keyframe_path)
            keyframe_indices.append(frame_idx)
            prev_frame = gray
        else:
            frame_diff = cv2.absdiff(gray, prev_frame)
            mean_diff = np.mean(frame_diff)
            
            if mean_diff > min_scene_change:
                keyframe_path = os.path.join(output_dir, f"keyframe_{frame_idx:04d}.jpg")
                cv2.imwrite(keyframe_path, frame)
                keyframes.append(keyframe_path)
                keyframe_indices.append(frame_idx)
                
            prev_frame = gray
        
        frame_idx += 1
    
    cap.release()
    
    if len(keyframes) < num_frames and use_uniform_fallback:
        print(f"Usage Scenario change If {len(keyframes)} is detected but {num_frames} is insufficient, uniform sampling will be used to replenish them")
        
        keyframes = []
        keyframe_indices = []
        
        cap = cv2.VideoCapture(video_path)
        
        sample_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame in sample_indices:
                keyframe_path = os.path.join(output_dir, f"keyframe_{current_frame:04d}.jpg")
                cv2.imwrite(keyframe_path, frame)
                keyframes.append(keyframe_path)
                keyframe_indices.append(current_frame)
                
            current_frame += 1
                
        cap.release()

    elif len(keyframes) > num_frames:
        print(f"Usage Scenario change If {len(keyframes)} is detected, if {num_frames} is exceeded, the system selects even frames")
        
        selected_indices = np.linspace(0, len(keyframes) - 1, num_frames, dtype=int)
        keyframes = [keyframes[i] for i in selected_indices]
        keyframe_indices = [keyframe_indices[i] for i in selected_indices]
    
    print(f"Finally, {len(keyframes)} keyframes were extracted")
    
    if fps > 0:
        keyframe_times = [idx / fps for idx in keyframe_indices]
        for i, (idx, time) in enumerate(zip(keyframe_indices, keyframe_times)):
            print(f"Key frame {i+1}: frame {idx}, time {time:.2f}s")
    
    return keyframes


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.backbone = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, inception.maxpool1,
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            inception.maxpool2, inception.Mixed_5b,
            inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b,
            inception.Mixed_6c, inception.Mixed_6d,
            inception.Mixed_6e, inception.Mixed_7a,
            inception.Mixed_7b, inception.Mixed_7c
        )
        self.backbone.eval()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=(1, 1))
        return features.reshape(features.shape[0], -1)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone.eval()
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        features = self.backbone(x)
        features = features.reshape(B, T, 2048)
        features = features.mean(dim=1)
        return features

class CLIPMetrics:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained("/data/xwl/xwl_data/.cache/clip-vit-large-patch14").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("/data/xwl/xwl_data/.cache/clip-vit-large-patch14")
        
    def calculate_similarity(self, image_path: str, text_prompt: str) -> float:
        try:
            image = Image.open(image_path).convert('RGB')
            if len(text_prompt.split()) > 70:
                logger.warning(f"Text prompt too long, truncating: {text_prompt[:100]}...")
                text_prompt = ' '.join(text_prompt.split()[:70])
            
            inputs = self.processor(
                images=image,
                text=[text_prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
                return float(similarity[0])
        except Exception as e:
            logger.error(f"Error calculating CLIP similarity for {image_path}: {e}")
            return None

class VideoMetricsCalculator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fvd_extractor = ResNetFeatureExtractor().to(self.device)
        self.inception = InceptionV3FeatureExtractor().to(self.device)
        self.clip_metrics = CLIPMetrics()
        
        # Transform for FVD
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform for FID
        self.inception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        """
        Normalize a set of raw scores to the 0-100 range, where the best (minimum) corresponds to 100 and the worst (maximum) corresponds to 0
        """
        arr = np.array(scores)
        if arr.max() == arr.min():
            return [100.0] * len(scores)
        return (100 * (1 - (arr - 1) / (1000 - 1))).tolist()

    def load_frame_sequence(self, frame_paths: List[str]) -> torch.Tensor:
        frames = []
        for path in frame_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
                frames.append(img_tensor)
            except Exception as e:
                logger.error(f"Error loading frame {path}: {e}")
                continue
        
        if not frames:
            raise ValueError("No frames were successfully loaded")
            
        video_tensor = torch.stack(frames)
        return video_tensor.permute(1, 0, 2, 3)
    
    def load_and_sample_mp4(self, video_path: str, target_frames: int) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Failed to read video: {video_path}")
            
        sample_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frames = []
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame in sample_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
                
            current_frame += 1
            
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames sampled from video: {video_path}")
            
        video_tensor = torch.stack(frames)
        return video_tensor.permute(1, 0, 2, 3)

    def extract_features(self, video_tensor: torch.Tensor) -> np.ndarray:
        video_tensor = video_tensor.to(self.device)
        with torch.no_grad():
            if video_tensor.dim() == 4:
                video_tensor = video_tensor.unsqueeze(0)
            features = self.fvd_extractor(video_tensor)
        return features.cpu().numpy()

    def extract_inception_features(self, images: List[str], batch_size: int = 32) -> np.ndarray:
        features_list = []
        
        for i in range(0, len(images), batch_size):
            batch_paths = images[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.inception_transform(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    logger.error(f"Error loading image {path}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            batch = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                batch_features = self.inception(batch)
                features_list.append(batch_features.cpu().numpy())
        
        if not features_list:
            raise ValueError("No features were extracted")
            
        return np.concatenate(features_list, axis=0)

    def calculate_fid(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        try:
            if real_features.ndim == 1:
                real_features = real_features.reshape(1, -1)
            if gen_features.ndim == 1:
                gen_features = gen_features.reshape(1, -1)
            
            if real_features.shape[0] == 1 or gen_features.shape[0] == 1:
                mu1 = np.mean(real_features, axis=0)
                mu2 = np.mean(gen_features, axis=0)
                return float(np.sum((mu1 - mu2) ** 2))
            
            mu1 = np.mean(real_features, axis=0)
            mu2 = np.mean(gen_features, axis=0)
            
            sigma1 = np.cov(real_features, rowvar=False)
            sigma2 = np.cov(gen_features, rowvar=False)
            
            diff = mu1 - mu2
            covmean = linalg.sqrtm(sigma1.dot(sigma2))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            fid = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
            return float(fid)
            
        except Exception as e:
            logger.error(f"Error in FID calculation: {e}")
            return float('inf')

    def calculate_fvd(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        return self.calculate_fid(real_features, gen_features)

    def process_video_prediction(self, data: List[Dict], mp4_prefix: str) -> Dict:
        i2v_data = [item for item in data if item.get('category') == 'Video prediction']
        
        results = {
            "total_samples": len(i2v_data),
            "processed_samples": 0,
            "skipped_samples": 0,
            "per_frame_fid_scores": []
        }
        
        sample_fvd_scores = [] 
        
        for idx, item in tqdm(enumerate(i2v_data), total=len(i2v_data)):
            try:
                if not self._validate_item(item):
                    results["skipped_samples"] += 1
                    continue
                
                gen_frames = item['output2']
                source_image_path = f"{mp4_prefix.rstrip('/')}/{item['data']['image'].lstrip('/')}"
                video_path = f"{mp4_prefix.rstrip('/')}/{item['data']['video'].lstrip('/')}"
                text_prompt = item.get('Text_Prompt', '')
                
                try:
                    source_features = self.extract_inception_features([source_image_path])
                    frame_fid_scores = []
                    for frame_path in gen_frames:
                        try:
                            frame_features = self.extract_inception_features([frame_path])
                            fid_score = self.calculate_fid(source_features, frame_features)
                            frame_fid_scores.append(fid_score)
                        except Exception as e:
                            logger.error(f"Error calculating FID for frame {frame_path}: {e}")
                            frame_fid_scores.append(None)
                    
                    results.setdefault("per_frame_fid_scores", []).append({
                        "id": item.get('id', 'N/A'),
                        "frame_fid_scores": frame_fid_scores,
                        "mean_fid": np.mean([s for s in frame_fid_scores if s is not None])
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing source image {source_image_path}: {e}")

                try:
                    gen_video = self.load_frame_sequence(gen_frames)
                    real_video = self.load_and_sample_mp4(video_path, len(gen_frames))
                    
                    real_features = self.extract_features(real_video)
                    gen_features = self.extract_features(gen_video)
                    
                    if self._validate_features(real_features, gen_features):
                        sample_fvd = float(self.calculate_fvd(real_features, gen_features))
                        sample_fvd_scores.append(sample_fvd)
                        results["processed_samples"] += 1
                except Exception as e:
                    logger.error(f"Error processing video pair: {e}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                results["skipped_samples"] += 1
                continue
        
        if sample_fvd_scores:
            results["fvd_score"] = float(np.mean(sample_fvd_scores))
            norm_fvd = VideoMetricsCalculator.normalize_scores(sample_fvd_scores)
            results["fvd_normalized"] = float(np.mean(norm_fvd))
        
        return results

    def process_image_to_video(self, data: List[Dict], mp4_prefix: str) -> Dict:
        i2v_data = [item for item in data if item.get('category') == 'Conditional_Image_to_Video_Generation']
        
        results = {
            "total_samples": len(i2v_data),
            "processed_samples": 0,
            "skipped_samples": 0,
            "clip_scores": [],
            "per_frame_fid_scores": []
        }
        
        sample_fvd_scores = []  
        
        for idx, item in tqdm(enumerate(i2v_data), total=len(i2v_data)):
            try:
                if not self._validate_item(item):
                    results["skipped_samples"] += 1
                    continue
                
                gen_frames = item['output2']
                source_image_path = f"{mp4_prefix.rstrip('/')}/{item['data']['image'].lstrip('/')}"
                video_path = f"{mp4_prefix.rstrip('/')}/{item['data']['video'].lstrip('/')}"
                text_prompt = item.get('Text_Prompt', '')
                
                try:
                    source_features = self.extract_inception_features([source_image_path])
                    frame_fid_scores = []
                    for frame_path in gen_frames:
                        try:
                            frame_features = self.extract_inception_features([frame_path])
                            fid_score = self.calculate_fid(source_features, frame_features)
                            frame_fid_scores.append(fid_score)
                        except Exception as e:
                            logger.error(f"Error calculating FID for frame {frame_path}: {e}")
                            frame_fid_scores.append(None)
                    
                    results.setdefault("per_frame_fid_scores", []).append({
                        "id": item.get('id', 'N/A'),
                        "frame_fid_scores": frame_fid_scores,
                        "mean_fid": np.mean([s for s in frame_fid_scores if s is not None])
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing source image {source_image_path}: {e}")
                
                frame_clip_scores = []
                for frame_path in gen_frames:
                    clip_score = self.clip_metrics.calculate_similarity(frame_path, text_prompt)
                    if clip_score is not None:
                        frame_clip_scores.append(clip_score)
                if frame_clip_scores:
                    results.setdefault("clip_scores", []).append({
                        "id": item.get('id', 'N/A'),
                        "frame_scores": frame_clip_scores,
                        "mean_score": np.mean(frame_clip_scores)
                    })
                
                try:
                    gen_video = self.load_frame_sequence(gen_frames)
                    real_video = self.load_and_sample_mp4(video_path, len(gen_frames))
                    
                    real_features = self.extract_features(real_video)
                    gen_features = self.extract_features(gen_video)
                    
                    if self._validate_features(real_features, gen_features):
                        sample_fvd = float(self.calculate_fvd(real_features, gen_features))
                        sample_fvd_scores.append(sample_fvd)
                        results["processed_samples"] += 1
                except Exception as e:
                    logger.error(f"Error processing video pair: {e}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                results["skipped_samples"] += 1
                continue
        
        if sample_fvd_scores:
            results["fvd_score"] = float(np.mean(sample_fvd_scores))
            norm_fvd = VideoMetricsCalculator.normalize_scores(sample_fvd_scores)
            results["fvd_normalized"] = float(np.mean(norm_fvd))
        
        return results

    def process_text_to_video(self, data: List[Dict], mp4_prefix: str) -> Dict:
        t2v_data = [item for item in data if item.get('category') == 'Text-to-Video_Generation']
        
        results = {
            "total_samples": len(t2v_data),
            "processed_samples": 0,
            "skipped_samples": 0,
            "clip_scores": []
        }
        
        sample_fvd_scores = []  
        
        for idx, item in tqdm(enumerate(t2v_data), total=len(t2v_data)):
            try:
                if not self._validate_t2v_item(item):
                    results["skipped_samples"] += 1
                    continue
                
                gen_frames = item['output2']
                mp4_path = f"{mp4_prefix.rstrip('/')}/{item['data']['image'].lstrip('/')}"
                text_prompt = item.get('Text_Prompt', '')
                
                frame_clip_scores = []
                for frame_path in gen_frames:
                    clip_score = self.clip_metrics.calculate_similarity(frame_path, text_prompt)
                    if clip_score is not None:
                        frame_clip_scores.append(clip_score)
                if frame_clip_scores:
                    results.setdefault("clip_scores", []).append({
                        "id": item.get('id', 'N/A'),
                        "frame_scores": frame_clip_scores,
                        "mean_score": np.mean(frame_clip_scores)
                    })
                
                try:
                    gen_video = self.load_frame_sequence(gen_frames)
                    real_video = self.load_and_sample_mp4(mp4_path, len(gen_frames))
                    
                    real_features = self.extract_features(real_video)
                    gen_features = self.extract_features(gen_video)
                    
                    if self._validate_features(real_features, gen_features):
                        sample_fvd = float(self.calculate_fvd(real_features, gen_features))
                        sample_fvd_scores.append(sample_fvd)
                        results["processed_samples"] += 1
                except Exception as e:
                    logger.error(f"Error processing video pair: {e}")
                    results["skipped_samples"] += 1
                    continue
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                results["skipped_samples"] += 1
                continue
        
        if sample_fvd_scores:
            results["fvd_score"] = float(np.mean(sample_fvd_scores))
            norm_fvd = VideoMetricsCalculator.normalize_scores(sample_fvd_scores)
            results["fvd_normalized"] = float(np.mean(norm_fvd))
        
        return results

    def _validate_item(self, item: Dict) -> bool:
        if 'data' not in item:
            return False
        if 'image' not in item['data'] or 'video' not in item['data']:
            return False
        if 'output' not in item or not item['output']:
            return False
        if 'Text_Prompt' not in item:
            return False
        return True

    def _validate_features(self, real_features: np.ndarray, gen_features: np.ndarray) -> bool:
        if real_features.ndim != 2 or gen_features.ndim != 2:
            return False
        if real_features.shape[1] != 2048 or gen_features.shape[1] != 2048:
            return False
        if not (np.isfinite(real_features).all() and np.isfinite(gen_features).all()):
            return False
        return True

    def _validate_t2v_item(self, item: Dict) -> bool:
        if 'data' not in item or 'image' not in item['data']:
            logger.warning("Missing data or video field")
            return False
        if 'output' not in item or not item['output']:
            logger.warning("Missing or empty output field")
            return False
        if 'Text_Prompt' not in item:
            logger.warning("Missing Text_Prompt field")
            return False
        return True

    def process_dataset(self, json_path: str, mp4_prefix: str) -> Dict:
        logger.info(f"Loading data from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        i2v_results = self.process_image_to_video(data, mp4_prefix)
        t2v_results = self.process_text_to_video(data, mp4_prefix)
        vp_results = self.process_video_prediction(data, mp4_prefix)
        
        results = {
            "image_to_video": i2v_results,
            "text_to_video": t2v_results,
            "video_prediction": vp_results
        }
        
        output_path = 'video_metrics_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
            
        return results

def print_results_summary(results: Dict):
    print("\nResults Summary:")
    
    if "image_to_video" in results:
        i2v = results["image_to_video"]
        print("\nImage-to-Video Generation:")
        print(f"Total samples: {i2v['total_samples']}")
        print(f"Processed samples: {i2v['processed_samples']}")
        print(f"Skipped samples: {i2v['skipped_samples']}")
        if 'fvd_score' in i2v and i2v['fvd_score'] is not None:
            print(f"FVD Score: {i2v['fvd_score']:.2f}")
            if 'fvd_normalized' in i2v:
                print(f"FVD Normalized: {i2v['fvd_normalized']:.2f}")
        if i2v.get('clip_scores'):
            mean_clip = np.mean([s['mean_score'] for s in i2v['clip_scores']])
            print(f"Average CLIP Score: {mean_clip:.3f}")
        if i2v.get('per_frame_fid_scores'):
            fid_values = [s['mean_fid'] for s in i2v['per_frame_fid_scores'] if 'mean_fid' in s]
            if fid_values:
                norm_fid = VideoMetricsCalculator.normalize_scores(fid_values)
                print(f"Average FID Score: {np.mean(fid_values):.3f}")
                print(f"FID Normalized: {np.mean(norm_fid):.2f}")

    if "video_prediction" in results:
        vp = results["video_prediction"]
        print("\nVideo Prediction:")
        print(f"Total samples: {vp['total_samples']}")
        print(f"Processed samples: {vp['processed_samples']}")
        print(f"Skipped samples: {vp['skipped_samples']}")
        if 'fvd_score' in vp and vp['fvd_score'] is not None:
            print(f"FVD Score: {vp['fvd_score']:.2f}")
            if 'fvd_normalized' in vp:
                print(f"FVD Normalized: {vp['fvd_normalized']:.2f}")
        if vp.get('per_frame_fid_scores'):
            fid_values = [s['mean_fid'] for s in vp['per_frame_fid_scores'] if 'mean_fid' in s]
            if fid_values:
                norm_fid = VideoMetricsCalculator.normalize_scores(fid_values)
                print(f"Average FID Score: {np.mean(fid_values):.3f}")
                print(f"FID Normalized: {np.mean(norm_fid):.2f}")
    
    if "text_to_video" in results:
        t2v = results["text_to_video"]
        print("\nText-to-Video Generation:")
        print(f"Total samples: {t2v['total_samples']}")
        print(f"Processed samples: {t2v['processed_samples']}")
        print(f"Skipped samples: {t2v['skipped_samples']}")
        if 'fvd_score' in t2v and t2v['fvd_score'] is not None:
            print(f"FVD Score: {t2v['fvd_score']:.2f}")
            if 'fvd_normalized' in t2v:
                print(f"FVD Normalized: {t2v['fvd_normalized']:.2f}")
        if t2v.get('clip_scores'):
            mean_clip = np.mean([s['mean_score'] for s in t2v['clip_scores']])
            print(f"Average CLIP Score: {mean_clip:.3f}")

def main():
    try:
        calculator = VideoMetricsCalculator()
        json_path = '/data/xwl/xwl_code/Unify_Benchmark/results/Generation/CogVideoX-5B/result_with_keyframes.json'
        mp4_prefix = "/data/xwl/xwl_data/decode_images"
        results = calculator.process_dataset(json_path, mp4_prefix)
        print_results_summary(results)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
