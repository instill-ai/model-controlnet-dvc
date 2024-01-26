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

        t0 = time.time()

        processed_image = cv2.Canny(processed_image, low_threshold, high_threshold)
        processed_image = processed_image[:, :, None]
        processed_image = np.concatenate(
            [processed_image, processed_image, processed_image], axis=2
        )
        canny_image = Image.fromarray(processed_image)

        outpu_image = self.pipe(
            task_image_to_image_input.prompt,
            image=canny_image,
            num_inference_steps=task_image_to_image_input.num_inference_steps,
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
        print("batch_tensor_image.numpy().shape:", batch_tensor_image.numpy().shape)
        print("batch_tensor_image.shape: ", batch_tensor_image.shape)

        return construct_infer_response(
            req=req,
            outputs=[
                Metadata(
                    name="images",
                    datatype=str(DataType.TYPE_FP32.name),
                    # shape=[-1, -1, -1, -1],
                    shape=[1, 1024, 1024, 3],
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


class TritonPythonModel:
    # Reference: https://docs.nvidia.com/launchpad/data-science/sentiment/latest/sentiment-triton-overview.html

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
        Both keys and values are strings. The dictionary keys and values are:
        * model_config: A JSON string containing the model configuration
        * model_instance_kind: A string containing model instance kind
        * model_instance_device_id: A string containing model instance device ID
        * model_repository: Model repository path
        * model_version: Model version
        * model_name: Model name
        """
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])

        # Load the model
        self.logger.log_info(f"[DEBUG] diffusers version: {diffusers.__version__}")
        self.logger.log_info(f"[DEBUG] torch version: {torch.__version__}")

        controlnet_canny_path = str(
            Path(__file__).parent.absolute().joinpath("sd-controlnet-canny")
        )
        stable_diffution_path = str(
            Path(__file__).parent.absolute().joinpath("stable-diffusion-v1-5")
        )

        self.logger.log_info(f"[DEBUG] load model under path: {controlnet_canny_path}")
        self.logger.log_info(f"[DEBUG] load model under path: {stable_diffution_path}")

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

        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "IMAGES")
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        # disable_torch_init()
        responses = []

        for request in requests:
            try:
                # binary data typed back to string
                prompt = [
                    t.decode("UTF-8")
                    for t in pb_utils.get_input_tensor_by_name(request, "PROMPT")
                    .as_numpy()
                    .tolist()
                ]

                # prompt = str(pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0].decode("utf-8"))
                self.logger.log_info(
                    f"[DEBUG] input `prompt` type({type(prompt)}): {prompt}"
                )

                negative_prompt = [
                    t.decode("UTF-8")
                    for t in pb_utils.get_input_tensor_by_name(
                        request, "NEGATIVE_PROMPT"
                    )
                    .as_numpy()
                    .tolist()
                ]
                self.logger.log_info(
                    f"[DEBUG] input `negative_prompt` type({type(negative_prompt)}): {negative_prompt}"
                )

                num_images_per_prompt = [
                    t
                    for t in pb_utils.get_input_tensor_by_name(request, "SAMPLES")
                    .as_numpy()
                    .tolist()
                ][0]
                self.logger.log_info(
                    f"[DEBUG] input `num_images_per_prompt` type({type(num_images_per_prompt)}): {num_images_per_prompt}"
                )

                scheduler = [
                    t.decode("UTF-8")
                    for t in pb_utils.get_input_tensor_by_name(request, "SCHEDULER")
                    .as_numpy()
                    .tolist()
                ][0]
                self.logger.log_info(
                    f"[DEBUG] input `scheduler` type({type(scheduler)}): {scheduler}"
                )

                num_inference_steps = [
                    t
                    for t in pb_utils.get_input_tensor_by_name(request, "STEPS")
                    .as_numpy()
                    .tolist()
                ][0]
                self.logger.log_info(
                    f"[DEBUG] input `num_inference_steps` type({type(num_inference_steps)}): {num_inference_steps}"
                )
                guidance_scale = [
                    t
                    for t in pb_utils.get_input_tensor_by_name(
                        request, "GUIDANCE_SCALE"
                    )
                    .as_numpy()
                    .tolist()
                ][0]
                self.logger.log_info(
                    f"[DEBUG] input `guidance_scale` type({type(guidance_scale)}): {guidance_scale}"
                )

                random_seed = [
                    t
                    for t in pb_utils.get_input_tensor_by_name(request, "SEED")
                    .as_numpy()
                    .tolist()
                ][0]
                self.logger.log_info(
                    f"[DEBUG] input `seed` type({type(random_seed)}): {random_seed}"
                )

                prompt_image = pb_utils.get_input_tensor_by_name(
                    request, "PROMPT_IMAGE"
                ).as_numpy()[0]

                self.logger.log_info(
                    f"[DEBUG] input `PROMPT_IMAGE` type({type(prompt_image)}): {len(prompt_image)}"
                )

                extra_params_str = ""
                if (
                    pb_utils.get_input_tensor_by_name(request, "extra_params")
                    is not None
                ):
                    extra_params_str = str(
                        pb_utils.get_input_tensor_by_name(request, "extra_params")
                        .as_numpy()[0]
                        .decode("utf-8")
                    )
                self.logger.log_info(
                    f"[DEBUG] input `extra_params` type({type(extra_params_str)}): {extra_params_str}"
                )

                extra_params = {}
                # TODO: Add a function handle penalty
                try:
                    extra_params = json.loads(extra_params_str)
                    if "repetition_penalty" in extra_params:
                        self.logger.log_info(
                            "[DEBUG] WARNING `repetition_penalty` would crash transformerparsing faield!"
                        )
                        del extra_params["repetition_penalty"]
                except json.decoder.JSONDecodeError:
                    self.logger.log_info(
                        "[DEBUG] WARNING `extra_params` parsing faield!"
                    )
                    pass

                # random_seed = 0
                # if pb_utils.get_input_tensor_by_name(request, "random_seed") is not None:
                #     random_seed = int(pb_utils.get_input_tensor_by_name(request, "random_seed").as_numpy()[0])
                # self.logger.log_info(f'[DEBUG] input `random_seed` type({type(random_seed)}): {random_seed}')

                if random_seed > 0:
                    random.seed(random_seed)
                    np.random.seed(random_seed)
                    torch.manual_seed(random_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(random_seed)

                success_parsed_image = False
                try:
                    original_image = Image.open(
                        io.BytesIO(prompt_image)
                    )  # for instill vd3p
                    success_parsed_image = True
                except Exception as e:
                    print(e)
                    pass

                if not success_parsed_image:
                    try:
                        original_image = Image.open(
                            io.BytesIO(base64.b64decode(prompt_image))
                        )  # for test script
                        success_parsed_image = True
                    except Exception as e:
                        print(e)
                        pass

                if not success_parsed_image:
                    raise ValueError("Unable to parsed image")

                processed_image = np.array(original_image)

                low_threshold = 100
                high_threshold = 200

                processed_image = cv2.Canny(
                    processed_image, low_threshold, high_threshold
                )
                processed_image = processed_image[:, :, None]
                processed_image = np.concatenate(
                    [processed_image, processed_image, processed_image], axis=2
                )
                canny_image = Image.fromarray(processed_image)

                # # # Define how many steps and what % of steps to be run on each experts (80/20) here
                # # n_steps = 40
                # # high_noise_frac = 0.8
                # if "high_noise_frac" not in extra_params:
                #     extra_params["high_noise_frac"] = 0.8

                t0 = time.time()  # calculate time cost in following function call
                high_noise_frac = 0.8
                # run both experts
                outpu_image = self.pipe(
                    prompt, image=canny_image, num_inference_steps=num_inference_steps
                ).images[0]

                to_tensor_transform = transforms.ToTensor()
                tensor_image = to_tensor_transform(outpu_image)
                batch_tensor_image = (
                    tensor_image.unsqueeze(0).to("cpu").permute(0, 2, 3, 1)
                )
                torch.cuda.empty_cache()

                print(f"image: type({type(batch_tensor_image)}):")
                print(f"image: shape: {batch_tensor_image.shape}")

                tensor_output = [pb_utils.Tensor("IMAGES", batch_tensor_image.numpy())]
                responses.append(pb_utils.InferenceResponse(tensor_output))

            except Exception as e:
                self.logger.log_info(f"Error generating stream: {e}")
                self.logger.log_info(f"{traceback.format_exc()}")

                # error = pb_utils.TritonError(f"Error generating stream: {e}")
                # triton_output_tensor = pb_utils.Tensor(
                #     "IMAGES", np.asarray(["N/A"], dtype=self.output0_dtype)
                # )
                # response = pb_utils.InferenceResponse(
                #     output_tensors=[triton_output_tensor], error=error
                # )
                # responses.append(response)
                # self.logger.log_info("The model did not receive the expected inputs")
                raise e
            return responses

    def finalize(self):
        self.logger.log_info("Cleaning up ...")
