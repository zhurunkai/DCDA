# CDA
Code and Data for the submission: "**Context-aware Decomposing Adapters for Compositional Zero-shot Learning**"
> Pre-trained vision-language models (VLMs) like CLIP have shown promising generalization ability for Compositional Zero-shot Learn- ing (CZSL), which is to predict new compositions of visual prim- itives like attributes and objects. The existing CLIP-based CZSL methods usually learn an entangled representation for a composi- tion, ignoring separate representations of its attribute and object, and are less effective in capturing the feature divergence of an attribute (resp. an object) w.r.t different objects (resp. attributes). In this paper, we fully model intra- and inter-composition inter- actions between attribute and object primitives and build a more task-specific architecture upon CLIP through parameter-efficient adapters. More specifically, we design two kinds of adapters (i.e., L-Adapter and V-Adapter) that are inserted into CLIP’s transformer layers on the language and vision sides, respectively, for decompos- ing the entangled language and vision features for the primitives under different compositions. We therefore name our method as CDA (Context-aware Decomposing Adapters). Evaluations on three popular CZSL benchmarks show that our proposed solution sig- nificantly improves the performance of CZSL, and its components have been verified by solid ablation studies.

## Requirements

The model is developed using PyTorch with environment requirements provided in `requirements.txt`.
## Dataset Preparations
We experiment with three datasets: MIT-States, UT-Zappos, and C-GQA.

```
sh download_datasets.sh
```

If you already have setup the datasets, you can use symlink and ensure the following paths exist: `DATA_ROOT/<datasets> where <datasets> = {'mit-states', 'ut-zappos', 'cgqa'}.`

## Training

```
python train.py --config configs/<dataset>.yml
```

You can replace `<dataset>` with `{mit-states, ut-zappos, cgqa}`. The best hyperparameters are included in the paper.

## Evaluation
To evaluate the model, specify the directory where the checkpoint is located in config.

```
python evaluate.py --config configs/<dataset>.yml --eval_load_model <ckpt_location>
```

You can replace `<dataset>` with `{mit-states, ut-zappos, cgqa}`. The best hyperparameters are included in the paper. In addition, replace`<ckpt_location>`with the path of your actual checkpoint.

## Model Illustrations
We also publish the codes of two model variants for effectiveness verification and ablation studies.

### Effectiveness Verification of Adapters

- w/o L-Adapters

  ```
  python train.py --config configs/<dataset>.yml --has_l_adapter False
  ```

- w/o V-Adapters

  ```
  python train.py --config configs/<dataset>.yml --has_l_adapter False
  ```

- L-Adapter w/o context

  ```
  python train.py --config configs/<dataset>.yml --l_adapter_context False
  ```

- V-Adapter w/o context

  ```
  python train.py --config configs/<dataset>.yml --v_adapter_context False
  ```

- L&V-Adapter w/o context

  ```
  python train.py --config configs/<dataset>.yml --v_adapter_context False --l_adapter_context False
  ```

### Ablation Studies

- Insertion Location of Adapters

  ```
  python train.py --config configs/<dataset>.yml --v_adapter_location <location> --l_adapter_location <location>
  ```

  The value of location can be `'in'` or `'out'`.

- Insertion Depth of Adapters

  ```
  python train.py --config configs/<dataset>.yml --l_adapter_layers <l_adapter_layers_num> --v_adapter_layers <v_adapter_layers_num>
  ```

  

