<p align="center">
  <a href="#"><img src="assets/bangla_clip_2.PNG" alt="bangla clip"></a>
</p>
<p align="center">
    <em>CLIP (Contrastive Language–Image Pre-training) training code for Bangla.</em>
</p>
<p align="center">
    <em>Live Demo: </em> <a href="https://huggingface.co/spaces/zabir-nabil/bangla-clip">HuggingFace Space</>
</p>

---

#### Installation

* `python >= 3.9`
* `pip install -r requirements.txt`


### Bangla CLIP

<p align="center">
  <a href="#"><img src="assets/clip_bangla.png" alt="bangla clip"></a>
</p>

The model consists of an EfficientNet / ResNet image encoder and a BERT text encoder and was trained on multiple datasets from Bangla image-text domain. To start training,

```console
python train_clip_bangla.py
```
---


### Image Search Demo with Bangla CLIP

 * *Search App Code:* [bangla-image-search](https://github.com/zabir-nabil/bangla-image-search)
 * *Article:* [medium](https://medium.com/@furcifer/a-dead-simple-image-search-engine-for-bangla-using-clip-contrastive-language-image-pre-training-a1f7f50b8419)

### Citation

```
@misc{nazi2024bangla,
  author = {Zabir Al Nazi},
  title = {A dead-simple image search engine for Bangla using CLIP (Contrastive Language–Image Pre-training)},
  howpublished = {\url{https://github.com/zabir-nabil/bangla-CLIP}},
  year = {2024},
}
```


