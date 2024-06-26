# pylint: skip-file
import os

TORCH_GPU_DEVICE_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{TORCH_GPU_DEVICE_ID}"

import time
import random
from PIL import Image

import numpy as np
import torch
import cv2

import diffusers


from instill.helpers.const import DataType, ImageToImageInput
from instill.helpers.ray_io import StandardTaskIO

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    construct_infer_response,
    construct_metadata_response,
    Metadata,
)


@instill_deployment
class ControlNet:
    def __init__(self):
        print(f"torch version: {torch.__version__}")
        print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count() : {torch.cuda.device_count()}")
        print(f"torch.cuda.current_device() : {torch.cuda.current_device()}")
        print(f"torch.cuda.device(0) : {torch.cuda.device(0)}")
        print(f"torch.cuda.get_device_name(0) : {torch.cuda.get_device_name(0)}")

        controlnet = diffusers.ControlNetModel.from_pretrained(
            "sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True
        )

        self.pipe = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None,
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
        print(task_image_to_image_input.prompt_image.shape)
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

        print("canny_image")
        print(canny_image)
        # https://github.com/huggingface/diffusers/blob/ea9dc3fa90c70c7cd825ca2346a31153e08b5367/src/diffusers/pipelines/controlnet/pipeline_controlnet.py#L900
        #  `(batch_size, height, width, num_channels)`
        output_arr = self.pipe(
            task_image_to_image_input.prompt,
            image=canny_image,
            # negative_prompt
            height=canny_image.height,
            width=canny_image.width,
            num_inference_steps=num_inference_steps,
            generator=torch.manual_seed(2),
            output_type="np",
        ).images

        print("Output:")
        output_arr_tansponse = output_arr
        response_shape = list(output_arr_tansponse.shape)
        task_output = output_arr_tansponse.tobytes()
        print("output_arr.shape:", response_shape)

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


entrypoint = InstillDeployable(ControlNet).get_deployment_handle()
