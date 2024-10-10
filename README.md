# DCDA
Code and Data for the submission: "**Cross-composition Feature Disentanglement for Compositional Zero-shot Learning**".
> Disentanglement of visual features of primitives (i.e., attributes and objects) has shown exceptional results in Compositional Zero-shot Learning (CZSL). However, due to the feature divergence of an attribute (resp. object) when combined with different objects (resp. attributes), it is challenging to learn disentangled primitive features that are general across different compositions. To this end, we propose the solution of cross-composition feature disentanglement, which takes multiple primitive-sharing compositions as inputs and constrains the disentangled primitive features to be general across these compositions. More specifically, we leverage a compositional graph to define the overall primitive-sharing relationships between compositions, and build a task-specific architecture upon the recently successful large pre-trained vision-language model (VLM) CLIP, with dual cross-composition disentangling adapters (called L-Adapter and V-Adapter) inserted into CLIP’s frozen text and image encoders, respectively. Evaluation on three popular CZSL benchmarks shows that our proposed solution significantly improves the performance of CZSL, and its components have been verified by solid ablation studies. 

## Requirements

The model is developed using PyTorch with environment requirements provided in `requirements.txt`.
## Dataset Preparations
We experiment with three datasets: MIT-States, UT-Zappos, and C-GQA.

Please download datasets to the folder `data` by running the following command.
```
sh utils/download_datasets.sh
```
If you already have setup the datasets, you can use symlink and ensure the following paths exist: `data/<dataset> where <dataset> = {'mit-states', 'ut-zappos', 'cgqa'}.`


## Training

Please replace `<dataset>` with `{mit-states, ut-zappos, cgqa}`. The best hyperparameters are included in the corresponding `.yml` file.

```
python train.py --config configs/<dataset>.yml
```


## Evaluation
We use `<ckpt_location>` to save the best model. Please specify it from `logs/` before evaluating the model. 
```
python evaluate.py --config configs/<dataset>.yml --eval_load_model <ckpt_location>
```
In the open-world setting, we apply a tailored approach for the L-adapter and utilize sharding to optimize memory efficiency.
```
python split_evaluate.py --config configs/<dataset>.yml --eval_load_model <ckpt_location>
```

## Model Variants
In our paper, we develop a series of model variants for evaluating the effectiveness of L-Adapters and V-Adapters, and testing the best location and depth to insert adapters.
 
### Effectiveness of Adapters

- removing L-Adapters, i.e., `w/o L-Adapters` in the paper

  ```
  python train.py --config configs/<dataset>.yml --has_l_adapter False
  ```

- removing V-Adapters, i.e., `w/o V-Adapters` in the paper

  ```
  python train.py --config configs/<dataset>.yml --has_l_adapter False
  ```

- L-Adapters with composition context removed, i.e., `L-Adapter w/o context` in the paper

  ```
  python train.py --config configs/<dataset>.yml --l_adapter_context False
  ```

- V-Adapters with composition context removed, i.e., `V-Adapter w/o context` in the paper

  ```
  python train.py --config configs/<dataset>.yml --v_adapter_context False
  ```

- Removing composition context in both L-Adapters and V-Adapters i.e.,  `L&V-Adapter w/o context` in the paper

  ```
  python train.py --config configs/<dataset>.yml --v_adapter_context False --l_adapter_context False
  ```

### Ablation Studies

- testing Insertion Location of Adapters

  ```
  python train.py --config configs/<dataset>.yml --v_adapter_location <location> --l_adapter_location <location>
  ```

  The choice of `<location>` can be `'in'` or `'out'`.

- testing Insertion Depth of Adapters

  ```
  python train.py --config configs/<dataset>.yml --l_adapter_layers <l_adapter_layers_num> --v_adapter_layers <v_adapter_layers_num>
  ```
  The values of `<l_adapter_layers_num>` can vary from 0 to 12, The values of `<v_adapter_layers_num>` can vary from 0 to 24.

  

