"""
VLM Benchmarking Script for Country Image Dataset
Simple version with single customizable prompt
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VLMBenchmark:
    def __init__(self, dataset_path: str, output_dir: str = "benchmark_results", 
                 prompt: str = "Looking at this image, which country do you think this is from?"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Single customizable prompt
        self.prompt = prompt
        self.results = []
        
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
    
    def benchmark_blip2_small(self, image_path: str) -> Dict[str, Any]:
        """Benchmark BLIP-2 (smallest VLM option)"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            from PIL import Image
            import torch
            
            # Load model and processor (using smaller variant)
            model_name = "Salesforce/blip2-opt-2.7b"
            processor = Blip2Processor.from_pretrained(model_name)
            model = Blip2ForConditionalGeneration.from_pretrained(model_name)
            
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Process inputs
            inputs = processor(images=image, text=self.prompt, return_tensors="pt")
            
            # Generate response
            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_length=50)
            
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            inference_time = time.time() - start_time
            
            return {
                "model": "blip2-opt-2.7b",
                "response": response.strip(),
                "inference_time": inference_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error with BLIP-2: {str(e)}")
            return {
                "model": "blip2-opt-2.7b",
                "response": "",
                "inference_time": 0,
                "success": False,
                "error": str(e)
            }
    
    def benchmark_llava_small(self, image_path: str) -> Dict[str, Any]:
        """Benchmark LLaVA (small variant)"""
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            from PIL import Image
            import torch
            
            # Use smaller LLaVA model
            model_name = "llava-hf/llava-1.5-7b-hf"
            processor = LlavaNextProcessor.from_pretrained(model_name)
            model = LlavaNextForConditionalGeneration.from_pretrained(model_name)
            
            image = Image.open(image_path).convert('RGB')
            
            # Format prompt for LLaVA
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image"}
                    ]
                }
            ]
            
            prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt_text, return_tensors="pt")
            
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50)
            
            response = processor.decode(output[0], skip_special_tokens=True)
            inference_time = time.time() - start_time
            
            return {
                "model": "llava-1.5-7b",
                "response": response.strip(),
                "inference_time": inference_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error with LLaVA: {str(e)}")
            return {
                "model": "llava-1.5-7b",
                "response": "",
                "inference_time": 0,
                "success": False,
                "error": str(e)
            }
    
    def benchmark_moondream(self, image_path: str) -> Dict[str, Any]:
        """Benchmark Moondream (very small VLM)"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from PIL import Image
            import torch
            
            
            model_id = "vikhyatk/moondream2"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            image = Image.open(image_path).convert('RGB')
            
            start_time = time.time()
            enc_image = model.encode_image(image)
            response = model.answer_question(enc_image, self.prompt, tokenizer)
            inference_time = time.time() - start_time
            
            return {
                "model": "moondream2",
                "response": response.strip(),
                "inference_time": inference_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error with Moondream: {str(e)}")
            return {
                "model": "moondream2",
                "response": "",
                "inference_time": 0,
                "success": False,
                "error": str(e)
            }
    
    def run_benchmark(self, models: List[str] = None):
        """Run benchmark on selected models"""
        
        if models is None:
            models = ["moondream"]  # Start with smallest
        
        dataset = self.load_dataset()
        
        model_functions = {
            "blip2": self.benchmark_blip2_small,
            "llava": self.benchmark_llava_small,
            "moondream": self.benchmark_moondream
        }
        
        total_tests = sum([len(images) for images in dataset.values()]) * len(models)
        
        logger.info(f"Starting benchmark with prompt: '{self.prompt}'")
        logger.info(f"Total tests: {total_tests}")
        
        test_count = 0
        
        for country, image_paths in dataset.items():
            # Use all images in each country folder
            selected_images = image_paths
            
            for image_path in selected_images:
                for model_name in models:
                    if model_name not in model_functions:
                        logger.warning(f"Model {model_name} not supported")
                        continue
                    
                    model_func = model_functions[model_name]
                    test_count += 1
                    
                    logger.info(f"Test {test_count}/{total_tests}: {model_name} on {country}")
                    
                    result = model_func(image_path)
                    
                    # Check if response contains correct country name
                    correct_country = False
                    if result["success"] and result["response"]:
                        # Simple check - country name appears in response (case insensitive)
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
                    
                    # Print result for immediate feedback
                    if result["success"]:
                        status = "✓ CORRECT" if correct_country else "✗ INCORRECT"
                        print(f"  {status}: {result['response'][:100]}...")
                    else:
                        print(f"  ERROR: {result['error']}")
                    
                    # Save intermediate results every 50 tests
                    if test_count % 50 == 0:
                        self.save_results()
        
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        """Save results to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
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
        
        logger.info(f"Report generated: {report_path}")
        
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLM models on country image dataset")
    parser.add_argument("dataset_path", help="Path to dataset folder containing country subfolders")
    parser.add_argument("--models", nargs="+", default=["moondream"], 
                       choices=["blip2", "llava", "moondream"],
                       help="Models to benchmark")
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--prompt", default="Looking at this image, which country do you think this is from?",
                       help="Custom prompt to use for all tests")
    
    args = parser.parse_args()
    
    benchmark = VLMBenchmark(args.dataset_path, args.output_dir, args.prompt)
    benchmark.run_benchmark(models=args.models)

if __name__ == "__main__":
    main()