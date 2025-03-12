# <img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/logo_tight.png" style="vertical-align: -10px;" :height="50px" width="50px"> ShareGPT4V: Improving Large Multi-modal Models with Better Captions

---

🚀🚀🚀 Official implementation of **ShareGPT4V: Improving Large Multi-modal Models with Better Captions** in ECCV 2024.

- **Authors**: [Lin Chen*](https://lin-chen.site), [Jinsong Li*](https://li-jinsong.github.io/), [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en), [Pan Zhang](https://panzhang0212.github.io/), [Conghui He](https://conghui.github.io/), [Jiaqi Wang](https://myownskyw7.github.io/), [Feng Zhao📧](https://scholar.google.com/citations?hl=en&user=r6CvuOUAAAAJ), [Dahua Lin📧](http://dahua.site/)

- **Institutes**: University of Science and Technology of China; Shanghai AI Laboratory
- **Resources**: [[Paper](https://arxiv.org/pdf/2311.12793.pdf)] [[Project Page](https://ShareGPT4V.github.io/)] [[<img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/logo_tight.png" style="vertical-align: -10px;" :height="20px" width="20px">ShareGPT4V Dataset](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)] 
- **Models**: [[ShareGPT4V-7B](https://huggingface.co/Lin-Chen/ShareGPT4V-7B)] [[ShareCaptioner](https://huggingface.co/Lin-Chen/ShareCaptioner)]

## 🤖 Model Zoo

See more details in [ModelZoo.md](docs/ModelZoo.md).

| Name | LLM | Checkpoint | LLaVA-Bench-Wild | MME-perception | MME-cognition | MMBench | MMBench-CN | SEED-image | MM-Vet | QBench | SQA-image | VQA-v2 | VizWiz | GQA | TextVQA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ShareGPT4V-7B | Vicuna-7B | [ShareGPT4V-7B](https://huggingface.co/Lin-Chen/ShareGPT4V-7B) | 72.6 | 1567.4 | 376.4 | 68.8 | 62.2 | 69.7 | 37.6 | 63.4 | 68.4 | 80.6 | 57.2 | 63.3 | 60.4 |
| ShareGPT4V-13B | Vicuna-13B | [ShareGPT4V-13B](https://huggingface.co/Lin-Chen/ShareGPT4V-13B) | 79.9 | 1618.7 | 303.2 | 68.5 | 63.7 | 70.8 | 43.1 | 65.2 | 71.2 | 81.0 | 55.6 | 64.8 | 62.2 |

## Quick Usage
<details>
<summary>Example Code</summary>

```Python
from share4v.model.builder import load_pretrained_model
from share4v.mm_utils import get_model_name_from_path
from share4v.eval.run_share4v import eval_model

model_path = "Lin-Chen/ShareGPT4V-7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```

Check out the details wth the `load_pretrained_model` function in `share4v/model/builder.py`.

You can also use the `eval_model` function in `share4v/eval/run_llava.py` to get the output easily. By doing so, you can use this code on Colab directly after downloading this repository.

``` python
model_path = "Lin-Chen/ShareGPT4V-7B"
prompt = "What is the most common catchphrase of the character on the right?"
image_file = "examples/breaking_bad.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```
</details>

## Install

```bash
git clone https://github.com/InternLM/InternLM-XComposer --depth=1
cd projects/ShareGPT4V
conda create -n share4v python=3.10 -y
conda activate share4v

pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Demo
You can build your local demo by:
```
# run script
python tools/app.py
```

## Evaluation

To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

See [Evaluation.md](docs/Evaluation.md).

## ✒️ Citation
```bibtex
@article{chen2023sharegpt4v,
  title={ShareGPT4V: Improving Large Multi-Modal Models with Better Captions},
  author={Chen, Lin and Li, Jisong and Dong, Xiaoyi and Zhang, Pan and He, Conghui and Wang, Jiaqi and Zhao, Feng and Lin, Dahua},
  journal={arXiv preprint arXiv:2311.12793},
  year={2023}
}
```

## License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
