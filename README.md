# ARC-JSD: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation

<p align='center'>
    [<a href="#attributing-response-to-context">tutorial</a>]
    [<a href="https://arxiv.org/abs/2505.16415">paper</a>]
    [<a href="#citation">bib</a>]
</p>


<p align = 'center'>
  <img alt="Attributing context via ARC-JSD" src='images/ARC-JSD.gif' width='75%'/>
</p>

## Attributing Response to Context

Try our ARC-JSD using our demo notebook:
- [ARC-JSD Demo](demos/ARC_JSD.ipynb): a quickstart guide to use ARC-JSD <a target="_blank" href="https://colab.research.google.com/drive/1VCm1W4RK48YyfOc8vq1WQngAI_whKpE9?usp=sharing"><img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> 

## Locating Relevant Attention Heads and MLPs via ARC-JSD

<p align = 'center'>
  <img alt="Attributing context of Attn and MLP via ARC-JSD" src='images/ARC-JSD-attn-mlp.gif' width='75%'/>
</p>

### Libraries Installation
- Pytorch 2.4
- CUDA 12.4
- Python 3.10

```bash
pip install transformers==4.43.3 spacy==3.8.4 numpy==1.26.3 nltk accelerate wheel
pip install flash-attn==2.7.4.post1 --no-build-isolation 
```

### Run ARC-JSD to analyse attention heads and MLPs
```bash
python ARC_JSD.py --model_name Qwen/Qwen2-1.5B-Instruct
```

This will run ARC-JSD on the Qwen2-1.5B-Instruct model and output the attention heads and MLPs that are most relevant to the context. In addition, it will plot the JSD heatmaps for the attention heads and MLPs, and the decoded token heatmaps of each layer.

Currently, ARC-JSD supports the following models:
- Qwen/Qwen2-0.5B-Instruct
- Qwen/Qwen2-1.5B-Instruct
- Qwen/Qwen2-7B-Instruct
- Qwen/Qwen2-72B-Instruct
- Qwen/Qwen2.5-0.5B-Instruct
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct
- Qwen/Qwen2.5-32B-Instruct
- Qwen/Qwen2.5-72B-Instruct
- google/gemma2-2b-it
- google/gemma2-9b-it
- google/gemma2-27b-it


## Citation
If you think ARC-JSD is helpful for your research, please cite our paper:

```bibtex
@article{li2025attributing,
  title={Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation},
  author={Li, Ruizhe and Chen, Chen and Hu, Yuchen and Gao, Yanjun and Wang, Xi and Yilmaz, Emine},
  journal={arXiv preprint arXiv:2505.16415},
  year={2025}
}
```

## Contact
For any questions or issues, please open an issue on GitHub or contact us at `ruizhe.li@abdn.ac.uk`.

## Acknowledgements
Our code is inspired by the following works:
- [ContextCite](https://github.com/MadryLab/context-cite)
- [Retrieval Head](https://github.com/nightdessert/Retrieval_Head)

Our work is supported by the Gemma 2 Academic Program GCP Credit Award from Google.