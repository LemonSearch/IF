from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream
import os
import torchvision
from word2token import *
import numpy as np
import matplotlib.pyplot as plt

# loading the models
device = 'cuda:0'
if_I = IFStageI('IF-I-M-v1.0', device=device)
if_II = IFStageII('IF-II-M-v1.0', device=device)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
t5 = T5Embedder(device="cpu")

# set the prompts
prompt = 'ultra close-up color photo portrait of angry orange domestic cat'
count = 1   # use the same prompt to generate <how many> images
words_to_tokens = word2token(prompt)
word_idx = 9
word = prompt.split()[word_idx]
token_list = words_to_tokens[word_idx]

# save middle steps in output dir
# 定义保存路径
save_dir = "outputs/steps"
noisy_img_dir = os.path.join(save_dir, "noisy_img")
att_map_dir = os.path.join(save_dir, "att_map", word)
os.makedirs(noisy_img_dir, exist_ok=True)
os.makedirs(att_map_dir, exist_ok=True)

# 定义 sample_fn 函数
def save_each_step_image(step_idx, sample):
    # 生成文件名
    filename = os.path.join(noisy_img_dir, f"step_{step_idx}.png")
    # 保存图像
    torchvision.utils.save_image(sample['sample'][0], filename)     # 这里的0其实是[:count]，同时有几个prompt输入就前几个是图像
    return sample  # 返回 sample，不影响后续处理

# TODO: 保存所有attention map 的数据，在绘制热图的时候获得一个一致的value-color对应关系
# 定义 draw_att_map 函数
def draw_att_map(weight):
    n_heads, width, length = weight.shape
    # Step 1: 提取指定单词的注意力权重
    accumulated_weights = np.sum(weight[:, :, token_list], axis=-1)
    
    # Step 2: 计算H并重塑每个head的注意力图
    H = int(np.sqrt(width))
    attn_weights = accumulated_weights.reshape(n_heads, H, H)
    
    # Step 3: 对12个head结果求平均
    att_map = np.mean(attn_weights, axis=0)
    
    # Step 4: 检测已有文件数，生成唯一文件名
    existing_files = os.listdir(att_map_dir)
    file_count = sum(1 for f in existing_files if f.startswith("att_map"))
    filename = os.path.join(att_map_dir, f'att_map_{file_count}.png')
    
    # Step 5: 使用 matplotlib 绘制热力图并保存
    plt.figure(figsize=(6, 6))
    plt.imshow(att_map, cmap='hot', interpolation='nearest')
    plt.colorbar()  # 添加颜色条以展示数值对应的颜色
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=if_III,
    prompt=[prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
        "sample_fn": save_each_step_image,
        "att_weight_fn": draw_att_map, 
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