#!/usr/bin/env python3
"""
VLM Benchmarking Script for Country Image Dataset
Optimized version with model caching and batch processing
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import logging
import gc
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VLMBenchmark:
    def __init__(self, dataset_path: str, output_dir: str = "benchmark_results", 
                 prompt: str = "Looking at this image, which country do you think this is from?"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.prompt = prompt
        self.results = []
        
        # Model caching
        self.loaded_models = {}
        self.current_model = None
        
    def load_dataset(self) -> Dict[str, List[str]]:
        """Load dataset structure: country_name -> list of image paths"""
        dataset = {}
        
        for country_folder in self.dataset_path.iterdir():
            if country_folder.is_dir():
                country_name = country_folder.name
                image_files = []
                
                # Common image extensions
                extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                
                for ext in extensions:
                    image_files.extend(country_folder.glob(f"*{ext}"))
                    image_files.extend(country_folder.glob(f"*{ext.upper()}"))
                
                if image_files:
                    dataset[country_name] = [str(img) for img in image_files]
                    logger.info(f"Found {len(image_files)} images for {country_name}")
        
        logger.info(f"Loaded dataset with {len(dataset)} countries")
        return dataset
    
    def load_model(self, model_name: str):
        """Load and cache model - only load once per model"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        logger.info(f"Loading {model_name} model...")
        
        if model_name == "git":
            from transformers import AutoProcessor, AutoModelForCausalLM
            model_id = "microsoft/git-base"
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
            
        elif model_name == "moondream":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_id = "vikhyatk/moondream2"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="eager"
            )
            processor = AutoTokenizer.from_pretrained(model_id)
            
        elif model_name == "blip2":
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            model_id = "Salesforce/blip2-opt-2.7b"
            processor = Blip2Processor.from_pretrained(model_id)
            model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
            
        elif model_name == "llava":
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            model_id = "llava-hf/llava-1.5-7b-hf"
            processor = LlavaNextProcessor.from_pretrained(model_id)
            model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"Moved {model_name} to GPU")
        
        self.loaded_models[model_name] = (model, processor)
        logger.info(f"Successfully loaded {model_name}")
        return model, processor
    
    def clear_model_cache(self):
        """Clear model cache to free memory"""
        for model_name, (model, processor) in self.loaded_models.items():
            del model
            del processor
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleared model cache")
    
    def benchmark_image(self, model_name: str, image_path: str) -> Dict[str, Any]:
        """Benchmark single image with specified model"""
        try:
            from PIL import Image
            
            model, processor = self.load_model(model_name)
            image = Image.open(image_path).convert('RGB')
            
            start_time = time.time()
            
            if model_name == "git":
                inputs = processor(images=image, text=self.prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        pixel_values=inputs.pixel_values,
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=50,
                        num_beams=1,
                        do_sample=False
                    )
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
            elif model_name == "moondream":
                if torch.cuda.is_available():
                    image = image.cuda() if hasattr(image, 'cuda') else image
                
                with torch.no_grad():
                    enc_image = model.encode_image(image)
                    response = model.answer_question(enc_image, self.prompt, processor)
                    
            elif model_name == "blip2":
                inputs = processor(images=image, text=self.prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_length=50, num_beams=1)
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
            elif model_name == "llava":
                conversation = [{"role": "user", "content": [{"type": "text", "text": self.prompt}, {"type": "image"}]}]
                prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=image, text=prompt_text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                response = processor.decode(output[0], skip_special_tokens=True)
            
            inference_time = time.time() - start_time
            
            return {
                "model": model_name,
                "response": response.strip(),
                "inference_time": inference_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error with {model_name}: {str(e)}")
            return {
                "model": model_name,
                "response": "",
                "inference_time": 0,
                "success": False,
                "error": str(e)
            }
    
    def run_benchmark(self, models: List[str] = None):
        """Run benchmark on selected models with optimizations"""
        
        if models is None:
            models = ["git"]  # Start with most stable small model
        
        dataset = self.load_dataset()
        
        # Calculate total tests
        total_images = sum([len(images) for images in dataset.values()])
        total_tests = total_images * len(models)
        
        logger.info(f"Starting benchmark with prompt: '{self.prompt}'")
        logger.info(f"Total images: {total_images}, Models: {len(models)}, Total tests: {total_tests}")
        
        test_count = 0
        overall_start = time.time()
        
        # Process one model at a time to optimize memory usage
        for model_name in models:
            logger.info(f"\n{'='*50}")
            logger.info(f"TESTING MODEL: {model_name.upper()}")
            logger.info(f"{'='*50}")
            
            model_start_time = time.time()
            model_test_count = 0
            
            for country, image_paths in dataset.items():
                for image_path in image_paths:
                    test_count += 1
                    model_test_count += 1
                    
                    # Progress indicator
                    if test_count % 50 == 0:
                        elapsed = time.time() - overall_start
                        estimated_total = (elapsed / test_count) * total_tests
                        remaining = estimated_total - elapsed
                        logger.info(f"Progress: {test_count}/{total_tests} ({test_count/total_tests*100:.1f}%) - ETA: {remaining/60:.1f} min")
                    
                    result = self.benchmark_image(model_name, image_path)
                    
                    # Check if response contains correct country name
                    correct_country = False
                    if result["success"] and result["response"]:
                        correct_country = country.lower() in result["response"].lower()
                    
                    # Store results
                    self.results.append({
                        "timestamp": datetime.now().isoformat(),
                        "country": country,
                        "image_path": os.path.basename(image_path),
                        "model": result["model"],
                        "prompt": self.prompt,
                        "response": result["response"],
                        "inference_time": result["inference_time"],
                        "success": result["success"],
                        "error": result.get("error", ""),
                        "correct_country": correct_country
                    })
                    
                    # Minimal console output for speed
                    if model_test_count % 10 == 0:
                        status = "✓" if correct_country else "✗"
                        print(f"  {status} {country}: {result['response'][:50]}...")
                    
                    # Save intermediate results less frequently
                    if test_count % 100 == 0:
                        self.save_results()
            
            # Model summary
            model_time = time.time() - model_start_time
            model_results = [r for r in self.results if r['model'] == model_name]
            model_accuracy = sum(1 for r in model_results if r['correct_country']) / len(model_results) * 100
            avg_inference = sum(r['inference_time'] for r in model_results if r['success']) / len([r for r in model_results if r['success']])
            
            logger.info(f"Model {model_name} completed:")
            logger.info(f"  Accuracy: {model_accuracy:.1f}%")
            logger.info(f"  Avg inference time: {avg_inference:.2f}s")
            logger.info(f"  Total time: {model_time/60:.1f} minutes")
            
            # Clear this model from memory before loading next one
            if len(models) > 1:
                self.clear_model_cache()
        
        self.save_results()
        self.generate_report()
        
        total_time = time.time() - overall_start
        logger.info(f"\nBenchmark completed in {total_time/60:.1f} minutes")
    
    def save_results(self):
        """Save results to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV (faster for large datasets)
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {csv_path}")
    
    def generate_report(self):
        """Generate summary report"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        successful_df = df[df['success'] == True]
        
        report = {
            "summary": {
                "total_tests": len(df),
                "successful_tests": len(successful_df),
                "success_rate": len(successful_df) / len(df) * 100,
                "models_tested": df['model'].unique().tolist(),
                "countries_tested": df['country'].unique().tolist(),
                "prompt_used": self.prompt
            },
            "performance": {}
        }
        
        if len(successful_df) > 0:
            report["performance"] = {
                "avg_inference_time_by_model": successful_df.groupby('model')['inference_time'].mean().to_dict(),
                "success_rate_by_model": df.groupby('model')['success'].mean().to_dict(),
                "accuracy_by_model": successful_df.groupby('model')['correct_country'].mean().to_dict(),
                "accuracy_by_country": successful_df.groupby('country')['correct_country'].mean().to_dict()
            }
        
        report_path = self.output_dir / "benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Prompt: {self.prompt}")
        print(f"Total tests: {report['summary']['total_tests']}")
        print(f"Success rate: {report['summary']['success_rate']:.1f}%")
        print(f"Models tested: {', '.join(report['summary']['models_tested'])}")
        print(f"Countries: {len(report['summary']['countries_tested'])}")
        
        if 'accuracy_by_model' in report['performance']:
            print("\nAccuracy by Model:")
            for model, accuracy in report['performance']['accuracy_by_model'].items():
                print(f"  {model}: {accuracy*100:.1f}%")
            
            print("\nAverage Inference Time:")
            for model, time_val in report['performance']['avg_inference_time_by_model'].items():
                print(f"  {model}: {time_val:.2f}s")
        
        logger.info(f"Report generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLM models on country image dataset")
    parser.add_argument("dataset_path", help="Path to dataset folder containing country subfolders")
    parser.add_argument("--models", nargs="+", default=["git"], 
                       choices=["blip2", "llava", "moondream", "git"],
                       help="Models to benchmark")
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--prompt", default="Looking at this image, which country do you think this is from?",
                       help="Custom prompt to use for all tests")
    
    args = parser.parse_args()
    
    # Performance tips
    print("Performance Optimization Tips:")
    print("- Use GPU if available (CUDA)")
    print("- Test one model at a time for memory efficiency")
    print("- Start with 'git' model (fastest small model)")
    print("- Use --models git for quickest results")
    print()
    
    benchmark = VLMBenchmark(args.dataset_path, args.output_dir, args.prompt)
    benchmark.run_benchmark(models=args.models)

if __name__ == "__main__":
    main()