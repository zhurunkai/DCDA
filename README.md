# CDA
Code and Data for the submission: "**Context-aware Decomposing Adapters for Compositional Zero-shot Learning**"
> Pre-trained vision-language models (VLMs) like CLIP have shown promising generalization ability for Compositional Zero-shot Learn- ing (CZSL), which is to predict new compositions of visual prim- itives like attributes and objects. The existing CLIP-based CZSL methods usually learn an entangled representation for a composi- tion, ignoring separate representations of its attribute and object, and are less effective in capturing the feature divergence of an attribute (resp. an object) w.r.t different objects (resp. attributes). In this paper, we fully model intra- and inter-composition inter- actions between attribute and object primitives and build a more task-specific architecture upon CLIP through parameter-efficient adapters. More specifically, we design two kinds of adapters (i.e., L-Adapter and V-Adapter) that are inserted into CLIP’s transformer layers on the language and vision sides, respectively, for decompos- ing the entangled language and vision features for the primitives under different compositions. We therefore name our method as CDA (Context-aware Decomposing Adapters). Evaluations on three popular CZSL benchmarks show that our proposed solution sig- nificantly improves the performance of CZSL, and its components have been verified by solid ablation studies.

## Requirements

The model is developed using PyTorch with environment requirements provided in `requirements.txt`.
## Dataset Preparations
We experiment with three datasets: MIT-States, UT-Zappos, and C-GQA.

Please create a new root folder `DATA_ROOT` for saving the datasets, and download data using the commands in the following script.

```
sh download_datasets.sh
```

## Training

Please replace `<dataset>` with `{mit-states, ut-zappos, cgqa}`. The best hyperparameters are included in the corresponding `yml` file.

```
python train.py --config configs/<dataset>.yml
```


## Evaluation
We use `<ckpt_location>` to save the best model w.r.t the validation set. Please specify it from `configs/` before evaluating the model. 
```
python evaluate.py --config configs/<dataset>.yml --eval_load_model <ckpt_location>
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
  The values of `<l_adapter_layers_num>` can vary from 0 to 24.

  

