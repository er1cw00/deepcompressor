import os
import json
import torch
from diffusers.utils import load_image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Config
from safetensors.torch import safe_open, load_file
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.pipelines import (
    AutoPipelineForText2Image,
    DiffusionPipeline,
    FluxControlPipeline,
    FluxFillPipeline,
    FluxKontextPipeline,
    SanaPipeline,
)
from accelerate import init_empty_weights

# from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.models.text_encoders.t5_encoder import NunchakuT5EncoderModel

def dump_checkpoint_parameter(safetensor_path):
    try:
        with safe_open(safetensor_path, framework="pt") as f:
            metadata = f.metadata()
            # 有些模型会把类信息存储在 _class_name 字段中
            #print(json.dumps(metadata, indent=4, ensure_ascii=False))
            # class_name = metadata.get("_class_name", "N/A")
            # print(f"Class name in metadata: {metadata}")
            # return "FluxTransformer2DModel" in class_name
            #tensor_name = next(iter())

            for tensor_name in (f.keys()):
                tensor = f.get_tensor(tensor_name)
                #if tensor.dtype == torch.float8_e2m3fn
                print(f"{tensor_name:64} shape:{str(tensor.shape):48} dtype:{tensor.dtype}")     
             #   if tensor_name == 'model.diffusion_model.single_blocks.31.linear1.scale_input' or tensor_name == 'model.diffusion_model.single_blocks.31.linear1.scale_weight':
             #       print(f'  value: {tensor}') 
            return False
    except Exception as e:
        print(f"Error reading {safetensor_path}: {e}")
        return False
    

def load_t5_xxl(config_path, model_path, device, dtype):
    state_dict = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
            
    config = T5Config.from_json_file(config_path) #"t5_config_xxl.json"
    t5_encoder = T5EncoderModel(config)
    
    #state_dict = load_file(model_path) #"")
    #model.load_state_dict(state_dict, strict=False)
    with init_empty_weights():
        t5_encoder = T5EncoderModel(config).to(dtype)
    t5_encoder.eval()
        
    #model.to_empty(device=device)
    t5_encoder.load_state_dict(state_dict, strict=True)
    return t5_encoder

# def check_model(safetensor_path):
#     vae = AutoencoderKL.from_pretrained(
#             "black-forest-labs/FLUX.1-Kontext-dev",
#             subfolder="vae",
#             torch_dtype=torch.bfloat16  # or "auto", or omit
#         )
#     # print(vae) 
#     vae_path = "/home/eric/workspace/AI/sd/temp/deepcompressor/examples/redcraft/vae"
#     text_encoder_path = "/home/eric/workspace/AI/sd/temp/deepcompressor/examples/redcraft/text_encoder"
#     text_encoder_2_path = "/home/eric/workspace/AI/sd/ComfyUI/models/clip/flux.1-dev-clip.safetensors"
#     #text_encoder_2_path = "/home/eric/workspace/AI/sd/ComfyUI/models/clip/flux.1-dev-clip.safetensors"
#     clip_path ="/home/eric/workspace/AI/sd/ComfyUI/models/clip/"
#     # config = "/home/eric/workspace/AI/sd/temp/deepcompressor/examples/redcraft/transformer/config.json" #original_config=config,
#     # print(f'load transformer >> ')
#     #transformer = FluxTransformer2DModel.from_single_file(safetensor_path, local_files_only=True)
#     #transformer = NunchakuFluxTransformer2dModel.from_pretrained(safetensor_path, local_files_only=True)
         
#     text_encoder_2 = load_t5_xxl(os.path.join(clip_path, 't5_config_xxl.json'), os.path.join(clip_path, 't5xxl_fp8_e4m3fn_scaled.safetensors'))

#     #T5EncoderModel.from_single_file(text_encoder_2_path, local_files_only=True,  torch_dtype=dtype)
#     text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, local_files_only=True, torch_dtype=torch.bfloat16)
#     #text_encoder_2 = T5EncoderModel.from_pretrained(t5_path, local_files_only=True, config="t5/flux.1-dev-clip.json" torch_dtype=torch.bfloat16)
#     print(f'load text_encoder_2 >> {text_encoder_2}')
#     # # pipeline = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=dtype)
    
#     # vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True, torch_dtype=torch.bfloat16)
    
#     #pipeline = FluxKontextPipeline.from_single_file(safetensor_path, text_encoder=text_encoder, text_encoder_2=text_encoder_2, vae=vae, torch_dtype=torch.bfloat16)
#     # # pipeline = DiffusionPipeline.from_single_file(
#     # #     safetensor_path
#     # #     torch_dtype=torch.float16  # 根据模型类型决定
#     # # )

#     #transformer = pipeline.transformer  # 或 pipeline.unet.transformer，取决于结构
#     #print(transformer)
#     return False

def show_model_parameter(name, model):
    print(f"show {name} named_parameters ...")
    for name, param in model.named_parameters():
        print(f"{name:<64} shape:{tuple(param.shape)} dtype:{str(param.dtype)}; requires_grad={param.requires_grad}")
    
def load_flux_model(path):
    dtype = torch.bfloat16
    repo = "black-forest-labs/FLUX.1-Kontext-dev"
    clip_path = "/home/eric/workspace/AI/sd/ComfyUI/models/clip"
    text_encoder_2_path = "/home/eric/workspace/AI/sd/ComfyUI/models/clip/awq-int4-flux.1-t5xxl.safetensors"
    #text_encoder_2_path = "/home/eric/workspace/AI/sd/ComfyUI/models/clip/flux.1-dev-clip.safetensors"
    
    print('load vae')
    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
    print("load text_encoder_1 ...")
    text_encoder_1 = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype)
    print("load text_encoder_2 ...")
    #text_encoder_2 = load_t5_xxl(os.path.join(clip_pathss, 't5_config_xxl.json'), os.path.join(clip_path, 't5xxl_fp8_e4m3fn_scaled.safetensors'), 'cuda', dtype)
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(text_encoder_2_path, local_files_only=True, torch_dtype=dtype) 

    print("load pipline ...")
    pipeline = FluxKontextPipeline.from_single_file(path, config=repo, text_encoder=text_encoder_1, text_encoder_2=text_encoder_2, vae=vae, torch_dtype=dtype)
    print(f'pipeline.hf_device_map: {pipeline.hf_device_map}')
    
    pipeline.enable_sequential_cpu_offload()
    pipeline.precision = 'int4'
    
    image = load_image("/home/eric/workspace/AI/sd/temp/nunchaku/examples/robot.png").convert("RGB")

    prompt = "Make Pikachu hold a sign that says 'Nunchaku is awesome', yarn art style, detailed, vibrant colors"
    image = pipeline(image=image, prompt=prompt, num_inference_steps=10, guidance_scale=2.5).images[0]
    image.save("flux-kontext-dev.png")
    # transformer = pipeline.transformer
    
    # print(transformer.__class__.__name__)
    
    # show_model_parameter("transformer", pipeline.transformer)
    # show_model_parameter("vae", pipeline.vae)
    # show_model_parameter("text encoder 1", pipeline.text_encoder)
    # show_model_parameter("text encoder 2", pipeline.text_encoder_2)


# 替换成你的路径
#filepath = '/home/eric/workspace/AI/sd/temp/deepcompressor/examples/redcraft/redcraft-1.safetensors'
#filepath = "/home/eric/workspace/AI/sd/ComfyUI/models/diffusion_models/svdq-int4_r32-flux.1-kontext-dev.safetensors"
filepath= "/root/autodl-tmp/models/redcraft_fp16.safetensors"
dump_checkpoint_parameter(filepath)

# if check_class_name(filepath):
#     print("✔ 包含 FluxTransformer2DModel")
# else:
#     print("✘ 不包含 FluxTransformer2DModel")

