# Vision Language Models as Explainable Classifiers for Skin Lesions
This project trains a vision-language model using reinforcement learning to produce diagnostic classifications with interpretable reasoning for skin lesion images. The model is optimized not just for accuracy, but for explanation quality, robustness to user feedback, and interactive analysis (zoom tool). 

I submitted this to the Terra North Jersey STEM Fair in 2026 and gave a short talk about it at the Bridgewater-Raritan High School AI/ML club. The project won me second alternate for a scholarship to a Kean University summer research program. It finetunes Qwen 3 VL 30b A3b with reinforcement learning to classify skin lesions as benign or malignant and explain rationale using the Tinker platform. My results, samples, and methods can be found in the poster and slideshow. The slideshow also includes ablations and more technical details.

📄 Poster (presented to judges): https://github.com/sr5434/VLMSkinLesionClassifier/blob/main/Poster.pdf

📊 Slideshow (not presented to judges): https://github.com/sr5434/VLMSkinLesionClassifier/blob/main/Slides.pdf

🤖 The trained model can be found here: https://huggingface.co/sr5434/skin-cancer-classifier

## Quick start

1. Download this dataset: https://www.kaggle.com/datasets/tomooinubushi/all-isic-data-20240629 and save its contents to a directory called archive in the root of this repository.

2. Reformat the data, prepare splits, and balance classes:

```python prepare_data.py```

3. Resize images to 256x256:

```python resize_images.py```

4. Generate diverse prompts:

```python prompt_gen.py```

5. Order data in descending order of 4-shot accuracy:

```profile_correctness_reorder.py```

6. Train

```python train.py```

## Key Contributions

- Reinforcement learning with explanation and diagnosis rewards
- Trained to respond to user pushback by defending or revising previous responses
- Provided with a zoom tool for localized visual reasoning
- Taught to adhere to user formatting instructions

## Results

- Accuracy: 82.8%
- F1 score: 84.4%
- p-value: 1.38e-28
- Compared to SFT baseline: +46.1% accuracy improvement
- RL improves explanation quality while maintaining classification performance

## Acknowledgements
I would like to thank Thinking Machines for funding this research. Without their support, this work would not be possible.

## Citation
If you found this work useful, please cite it.
```
@software{Rangwalla_2026,
    title={Vision Language Models as Explainable Classifiers for Skin Lesions},
    url={http://github.com/sr5434},
    author={Rangwalla, Samir},
    year={2026},
    month={Mar}
}
```
