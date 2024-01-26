# pylint: skip-file
import os

TORCH_GPU_DEVICE_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{TORCH_GPU_DEVICE_ID}"


import traceback

from typing import Dict, List, Union

from torchvision import transforms

import io
import time
import json
import base64
import random
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import cv2

import diffusers


from instill.helpers.const import DataType, ImageToImageInput
from instill.helpers.ray_io import (
    serialize_byte_tensor,
    deserialize_bytes_tensor,
    StandardTaskIO,
)

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    construct_infer_response,
    construct_metadata_response,
    Metadata,
)


@instill_deployment
class ControlNet:
    def __init__(self, model_path: str):
        self.application_name = "_".join(model_path.split("/")[3:5])
        self.deployement_name = model_path.split("/")[4]
        print(f"application_name: {self.application_name}")
        print(f"deployement_name: {self.deployement_name}")
        print(f"torch version: {torch.__version__}")

        print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count() : {torch.cuda.device_count()}")
        print(f"torch.cuda.current_device() : {torch.cuda.current_device()}")
        print(f"torch.cuda.device(0) : {torch.cuda.device(0)}")
        print(f"torch.cuda.get_device_name(0) : {torch.cuda.get_device_name(0)}")

        if model_path[-1] != "/":
            model_path = f"{model_path}/"

        controlnet_canny_path = f"{model_path}sd-controlnet-canny/"
        stable_diffution_path = f"{model_path}stable-diffusion-v1-5/"

        controlnet = diffusers.ControlNetModel.from_pretrained(
            controlnet_canny_path, torch_dtype=torch.float16, use_safetensors=True
        )

        self.pipe = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
            stable_diffution_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        self.pipe.scheduler = diffusers.UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_model_cpu_offload()

    def ModelMetadata(self, req):
        resp = construct_metadata_response(
            req=req,
            inputs=[
                Metadata(
                    name="prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                # TODO: Check Wether `negative_prompt` is needed?
                # model-bakcend supports negative_prompt but not Python-Sdk
                Metadata(
                    name="negative_prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="prompt_image",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="samples",
                    datatype=str(DataType.TYPE_INT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="scheduler",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="steps",
                    datatype=str(DataType.TYPE_INT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="guidance_scale",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[1],
                ),
                Metadata(
                    name="seed",
                    datatype=str(DataType.TYPE_INT64.name),
                    shape=[1],
                ),
                Metadata(
                    name="extra_params",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                Metadata(
                    name="images",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[-1, -1, -1, -1],
                ),
            ],
        )
        return resp

    async def __call__(self, req):
        task_image_to_image_input: ImageToImageInput = (
            StandardTaskIO.parse_task_image_to_image_input(request=req)
        )
        print("----------------________")
        print(task_image_to_image_input)
        print("----------------________")

        print("print(task_image_to_image_input.prompt_image)")
        print(task_image_to_image_input.prompt_image)
        print("-------\n")

        print("print(task_image_to_image_input.prompt)")
        print(task_image_to_image_input.prompt)
        print("-------\n")

        # print("print(task_image_to_image_input.negative_prompt)")
        # print(task_image_to_image_input.negative_prompt)
        # print("-------\n")

        print("print(task_image_to_image_input.steps)")
        print(task_image_to_image_input.steps)
        print("-------\n")

        print("print(task_image_to_image_input.guidance_scale)")
        print(task_image_to_image_input.guidance_scale)
        print("-------\n")

        print("print(task_image_to_image_input.seed)")
        print(task_image_to_image_input.seed)
        print("-------\n")

        print("print(task_image_to_image_input.samples)")
        print(task_image_to_image_input.samples)
        print("-------\n")

        print("print(task_image_to_image_input.extra_params)")
        print(task_image_to_image_input.extra_params)
        print("-------\n")

        if task_image_to_image_input.seed > 0:
            random.seed(task_image_to_image_input.seed)
            np.random.seed(task_image_to_image_input.seed)
            # torch.manual_seed(task_image_to_image_input.seed)
            # if torch.cuda.is_available():
            #     torch.cuda.manual_seed_all(task_image_to_image_input.seed)

        low_threshold = 100
        if "low_threshold" in task_image_to_image_input.extra_params:
            low_threshold = task_image_to_image_input.extra_params["low_threshold"]

        high_threshold = 200
        if "high_threshold" in task_image_to_image_input.extra_params:
            high_threshold = task_image_to_image_input.extra_params["high_threshold"]

        num_inference_steps = task_image_to_image_input.steps
        if "num_inference_steps" in task_image_to_image_input.extra_params:
            num_inference_steps = task_image_to_image_input.extra_params[
                "um_inference_steps"
            ]

        t0 = time.time()
        processed_image = cv2.Canny(
            task_image_to_image_input.prompt_image, low_threshold, high_threshold
        )
        processed_image = processed_image[:, :, None]
        processed_image = np.concatenate(
            [processed_image, processed_image, processed_image], axis=2
        )
        canny_image = Image.fromarray(processed_image)

        outpu_image = self.pipe(
            task_image_to_image_input.prompt,
            image=canny_image,
            num_inference_steps=num_inference_steps,
        ).images[0]

        to_tensor_transform = transforms.ToTensor()
        tensor_image = to_tensor_transform(outpu_image)
        batch_tensor_image = tensor_image.unsqueeze(0).to("cpu").permute(0, 2, 3, 1)
        torch.cuda.empty_cache()

        print(f"Inference time cost {time.time()-t0}s")

        print(f"image: type({type(batch_tensor_image)}):")
        print(f"image: shape: {batch_tensor_image.shape}")

        # task_output = StandardTaskIO.parse_task_text_generation_output(sequences)
        # task_output = np.asarray(batch_tensor_image).tobytes()
        task_output = batch_tensor_image.numpy().tobytes()

        print("Output:")
        # print(task_output)
        print("type(task_output): ", type(task_output))
        response_shape = list(batch_tensor_image.numpy().shape)
        print("batch_tensor_image.numpy().shape:", response_shape)
        print("batch_tensor_image.shape: ", batch_tensor_image.shape)

        return construct_infer_response(
            req=req,
            outputs=[
                Metadata(
                    name="images",
                    datatype=str(DataType.TYPE_FP32.name),
                    # shape=[-1, -1, -1, -1],
                    shape=response_shape,
                )
            ],
            raw_outputs=[task_output],
        )


deployable = InstillDeployable(
    ControlNet,
    # There are two models in this directory,
    # path would be construct inside initialize function
    model_weight_or_folder_name="/",
    use_gpu=True,
)

# # Optional
# deployable.update_max_replicas(2)
# deployable.update_min_replicas(0)
