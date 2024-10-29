# from huggingface_hub import login
# login()


from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream
import os
import torchvision

# loading the models
device = 'cuda:0'
if_I = IFStageI('IF-I-M-v1.0', device=device)
if_II = IFStageII('IF-II-M-v1.0', device=device)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
t5 = T5Embedder(device="cpu")

# set the prompts
prompt = 'ultra close-up color photo portrait of domestic cat with huge nose in the kitchen'
count = 1   # use the same prompt to generate <how many> images

# save middle steps in output dir
# 定义保存路径
save_dir = "outputs/steps"
os.makedirs(save_dir, exist_ok=True)

# 定义 sample_fn 函数
def save_each_step_image(step_idx, sample):
    # 生成文件名
    filename = os.path.join(save_dir, f"step_{step_idx}.png")
    # 保存图像
    torchvision.utils.save_image(sample, filename)
    return sample  # 返回 sample，不影响后续处理

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=if_III,
    prompt=[prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
        "sample_fn": save_each_step_image,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
    if_III_kwargs={
        "guidance_scale": 9.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
    disable_watermark=True,
)

if_III.show(result['III'], size=14)