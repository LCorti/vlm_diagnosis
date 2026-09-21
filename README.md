# DiaVLo: Diagnosing the Behaviours of Vision-Language Models

[**DiaVLo: Diagnosing the Behaviours of Vision-Language Models**](https://arxiv.org/abs/2609.22008)

*Lorenzo Corti*, Jie Yang

To appear in [Findings of the Association for Computational Linguistics: EMNLP 2026](https://2026.emnlp.org/).

## Setup
> TODO

## Contact
For questions regarding DiaVLo, please contact [Lorenzo](https://lcorti.github.io/).

## Citation

If you find this work useful, please cite it:

```bibtex
@inproceedings{corti-2026-diavlo-vlm-diagnosis,
    title = "DiaVLo: Diagnosing Behaviours of Vision-Language Models",
    author = "Corti, Lorenzo and Yang, Jie",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2609.22008",
    abstract = "Vision-language models (VLMs) rely on storing and transferring appropriate information across their sub-components. Verifying that the VLMs exhibit desired behaviours, while avoiding harmful ones, is central to their reliable deployment. Yet, methods that identify VLM behaviours remain scarce. We present DiaVLo, a diagnostic framework that leverages human curation and VLMs' generation capabilities to construct specifications of desired and observed VLM behaviours, surfacing potential misalignments. Beyond this, DiaVLo also provides causal estimates to identify the most influential concepts steering VLM behaviours. We evaluate DiaVLo on several open-source VLMs under both classification and generation conditions. Our experiments show that DiaVLo produces behaviour labels that correlate with model performance and provide context for measured performance. DiaVLo surfaced behaviours that are clearly aligned and misaligned, alongside patterns in how VLMs perceive, organise, and prioritise concepts."
}
```
