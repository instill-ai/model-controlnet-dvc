---
Task: ImageToImage
Tags:
  - ImageToImage
  - Image-To-Image
  - controlNet
---

# Model-CONTROLNET-DVC

🔥🔥🔥 Deploy state-of-the-art [Control Net](https://huggingface.co/lllyasviel/sd-controlnet-canny) model in PyTorch format via open-source [VDP](https://github.com/instill-ai/vdp).

Notes:

- This repository contains the Controlnet - Canny Version which corresponds to the ControlNet conditioned on Canny edges and used in combination with [Dtable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as backend.
- Disk Space Requirements: 11G
- GPU Memory Requirements: 15G

**Create Model**

```json
{
    "id": "controlnet-canny-gpu",
    "description": "ControlNet-Canny Version, from Lvmin, is trained to generate image based on your prompts and images.",
    "model_definition": "model-definitions/container",
    "visibility": "VISIBILITY_PUBLIC",
    "region": "REGION_GCP_EUROPE_WEST_4",
    "hardware": "GPU",
    "configuration": {
        "task": "TASK_IMAGE_TO_IMAGE"
    }
}
```

**Inference model**

```json
{
    "task_inputs": [
        {
            "text_to_image": {
                "prompt": "Mona lisa",
                "steps": "50",
                "cfg_scale": "5.5",
                "seed": "1",
                "samples": 1
            }
        }
    ]
}
```
