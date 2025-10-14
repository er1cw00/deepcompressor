import torch
import time

from accelerate import dispatch_model, init_empty_weights
from diffusers import FluxKontextPipeline, AutoencoderKL
from diffusers.utils import load_image
from diffusers.models.transformers import FluxTransformer2DModel
from transformers import AutoTokenizer, CLIPTextModel, CLIPConfig
from safetensors.torch import load_file
#from optimum.quanto import QuantizedDiffusersModel, QuantizedTransformersModel

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.models.text_encoders.t5_encoder import NunchakuT5EncoderModel
from nunchaku.utils import get_precision


precision = get_precision()
print(f'precision: {precision}')

flux_config = {
  "attention_head_dim": 128,
  "axes_dims_rope": [
    16,
    56,
    56
  ],
  "guidance_embeds": True,
  "in_channels": 64,
  "joint_attention_dim": 4096,
  "num_attention_heads": 24,
  "num_layers": 19,
  "num_single_layers": 38,
  "out_channels": None,
  "patch_size": 1,
  "pooled_projection_dim": 768
}
flux_config1={'_class_name': 'FluxTransformer2DModel', '_diffusers_version': '0.30.0.dev0', '_name_or_path': '../checkpoints/flux-dev/transformer', 'attention_head_dim': 128, 'guidance_embeds': True, 'in_channels': 64, 'joint_attention_dim': 4096, 'num_attention_heads': 24, 'num_layers': 19, 'num_single_layers': 38, 'patch_size': 1, 'pooled_projection_dim': 768}

# class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
#     base_class = FluxTransformer2DModel

def load_redcraft_transformer():
    #dtype = torch.float8_e4m3fn
    dtype = torch.bfloat16
    # fp8_optimizations = True
    #transformer_path='/home/eric/workspace/AI/sd/ComfyUI/models/unet/redcraft.safetensors'
    #transformer_path = '/home/eric/workspace/AI/sd/temp/deepcompressor/examples/redcraft/redcraft-1.safetensors'
    #transformer = QuantizedFluxTransformer2DModel.from_pretrained(transformer_path).to(dtype)
    transformer_path = '/root/autodl-tmp/models/redcraft/transformer'
    print('load_redcraft_transformer >> from single file' )
    transformer = FluxTransformer2DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16
    )
    print('load_redcraft_transformer << from single file' )
    #state_dict = transformer.state_dict()

    # for key, tensor in state_dict.items():
    #     print(f"{key:60} | {str(tensor.dtype):12} | {tuple(tensor.shape)}")
    # return None
    # ctx = init_empty_weights 
    # transformer = None
    # with ctx():
    #     transformer = FluxTransformer2DModel.from_config(flux_config1, 
    #                                                     low_cpu_mem_usage=True, 
    #                                                     disable_mmap=False, 
    #                                                     torch_dtype=torch.bfloat16).to(torch.bfloat16)
    return transformer

def load_nunchaku_transformer():
    
    transformer_path="/home/eric/workspace/AI/sd/ComfyUI/models/diffusion_models/svdq-fp4_r32-flux.1-kontext-dev.safetensors"
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        transformer_path
    )
    return transformer

def load_kontext_pipeline(transformer):

    dtype = torch.bfloat16

    repo = "black-forest-labs/FLUX.1-Kontext-dev"
    clip_path = "/home/eric/workspace/AI/sd/ComfyUI/models/clip"
    vae_path='/home/eric/workspace/AI/sd/ComfyUI/models/vae/flux-kontext-vae.safetensors'
    
    

    text_encoder_2_path = "/root/autodl-tmp/models/awq-int4-flux.1-t5xxl.safetensors"
    #text_encoder_2_path = "/home/eric/workspace/AI/sd/ComfyUI/models/clip/awq-int4-flux.1-t5xxl.safetensors"

    text_encoder_1 = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype)

    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
        text_encoder_2_path
        #"mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
    )

    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
    #vae = AutoencoderKL.from_pretrained(vae_path)

    
    print("load pipline ...")
    pipeline = FluxKontextPipeline.from_pretrained(repo, transformer=transformer, text_encoder=text_encoder_1, text_encoder_2=text_encoder_2, vae=vae, torch_dtype=dtype)
    print(f'pipeline.hf_device_map: {pipeline.hf_device_map}')
        
    pipeline.enable_sequential_cpu_offload()
    pipeline.precision = precision
    return pipeline

def run_kontext_pipeline(pipeline):

    image = load_image("/root/autodl-tmp/robot.png").convert("RGB")
    prompt = "Make red Pikachu hold a sign that says 'mother is awesome', yarn art style, detailed, red colors"
    image = pipeline(image=image, prompt=prompt, num_inference_steps=10, guidance_scale=2.5).images[0]
    image.save("/root/autodl-tmp/flux-kontext-dev.png")

print('>>>>')
#transformer = load_nunchaku_transformer()
transformer = load_redcraft_transformer()
print('loaded transformer')
#time.sleep(10)
pipeline = load_kontext_pipeline(transformer)
run_kontext_pipeline(pipeline)

# export TORCH_CUDA_ARCH_LIST="12.0"
