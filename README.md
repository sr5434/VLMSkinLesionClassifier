# Vision Language Models as Explainable Classifiers for Skin Lesions
This is the project I submitted to the Terra North Jersey STEM Fair in 2026. It finetunes Qwen 3 VL 30b A3b with reinforcement learning to classify skin lesions as benign or malignant and explain rationale using the Tinker platform. It is also specifically trained to respond to human pushback by either defending or revising its response, follow specific formatting instructions that a user might give it, and use a zoom tool to take a better look at parts of an image. This technique is flexible and can be easily adapted to other tasks by swapping datasets and modifying the rubric used for grading explanations. The full slides for my project, as well as the poster I presented at the fair, are in this repository as PDFs.

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