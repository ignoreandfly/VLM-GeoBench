# 🌍 VLM-GeoBench

**VLM-GeoBench** is a benchmark designed to evaluate Vision-Language Models (VLMs) on real-world geolocation tasks using images from the GeoGuessr game and similar sources. It assesses how well VLMs can localize or identify countries based on visual content and prompt-based natural language reasoning.

> **Note**: The contrastive section of the repository is heavily based on [LR0.FM](https://github.com/shyammarjit/LR0.FM/) — we extend and adapt its structure for geolocation-specific evaluation.


## 🔍 Key Features

- 🗺️ **Diverse Geographical Coverage**: Includes images from a wide range of countries and environments (urban, rural, natural, architectural).
- ✍️ **Prompt Templates**: A rich set of handcrafted templates tailored for geolocation using VLMs.
- 📊 **Evaluation Suite**: Includes zero-shot and few-shot evaluation scripts for popular models.
- 🔄 **Plug-and-Play**: Easy integration with CLIP, BLIP, LLaVA, DINOv2, and more.
- 📁 **Dataset-Agnostic**: Works with both public GeoGuessr-style datasets and custom location-tagged image datasets.

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/VLM-GeoBench.git
cd VLM-GeoBench
python evaluate.py --model clip --dataset geoguessr
