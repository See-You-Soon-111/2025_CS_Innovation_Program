from blind_watermark import WaterMark
import PIL
import torchvision
import matplotlib.pyplot as plt
import torch
import kornia
import torch.nn.functional as F
import os


# 初始化水印系统
bwm1 = WaterMark(password_img=1, password_wm=1)
bwm1.read_img('cat.png')
bwm1.read_wm([True, False, True, False, True, False], mode='bit')
bwm1.embed('cat-w.png')
len_wm = len(bwm1.wm_bit)
print(f'Watermark length: {len_wm}')

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 确保张量维度工具函数
def ensure_4d_tensor(x):
    """确保输入为 [B, C, H, W] 格式"""
    if isinstance(x, torch.Tensor):
        if x.dim() == 3:
            return x.unsqueeze(0)
        elif x.dim() == 4:
            return x
    raise ValueError("Input must be 3D or 4D tensor.")





# 定义所有攻击方法
attacks = {
    'none': lambda x: x,
    # 水平翻转
    'flip_h': lambda x: kornia.geometry.transform.hflip(x),
    # 平移（右下角平移20%）
    'translate_20': lambda x: kornia.geometry.transform.translate(
        ensure_4d_tensor(x),
        translation=torch.tensor([[0.2 * x.shape[-1], 0.2 * x.shape[-2]]], device=x.device),
        mode='bilinear'
    ),
    # 平移（右下角平移40%）
    'translate_40': lambda x: kornia.geometry.transform.translate(
        ensure_4d_tensor(x),
        translation=torch.tensor([[0.4 * x.shape[-1], 0.4 * x.shape[-2]]], device=x.device),
        mode='bilinear'
    ),
    # 保留50%中心区域
    'crop_50': lambda x: kornia.geometry.transform.center_crop(
        ensure_4d_tensor(x),
        size=(int(x.shape[-2] * 0.5), int(x.shape[-1] * 0.5))
    ),
    # 缩小到原来的30%
    'resize_03': lambda x: F.interpolate(
        ensure_4d_tensor(x),
        scale_factor=0.3,
        mode='bilinear',
        align_corners=False
    ),
    # 缩小到原来的50%
    'resize_05': lambda x: F.interpolate(
        ensure_4d_tensor(x),
        scale_factor=0.5,
        mode='bilinear',
        align_corners=False
    ),
    # 旋转45度
    'rot_45': lambda x: kornia.geometry.transform.rotate(
        ensure_4d_tensor(x),
        angle=torch.tensor([45.0], device=x.device)
    ),
    # 旋转90 度
    'rot_90': lambda x: kornia.geometry.transform.rotate(
        ensure_4d_tensor(x),
        angle=torch.tensor([90.0], device=x.device)
    ),
    # 高斯模糊
    'blur': lambda x: kornia.filters.gaussian_blur2d(
        ensure_4d_tensor(x),
        kernel_size=(3, 3),
        sigma=(4.0, 4.0)
    ),
    # 降低对比度
    'contrast_05': lambda x: kornia.enhance.adjust_contrast(
        ensure_4d_tensor(x),
        factor=0.5
    ),
    # 增加对比度
    'contrast_20': lambda x: kornia.enhance.adjust_contrast(
        ensure_4d_tensor(x),
        factor=2.0
    ),

}

# 创建输出目录
os.makedirs('attack_results', exist_ok=True)

# 加载带水印的图片
image_w = PIL.Image.open('cat-w.png').convert('RGB')
to_tensor = torchvision.transforms.ToTensor()
image_tensor = to_tensor(image_w).unsqueeze(0).to(torch.float32).to(device)

# 测试所有攻击
for name, attack in attacks.items():
    print(f"\nProcessing attack: {name}")

    try:
        # 应用攻击
        distorted = attack(image_tensor)

        # 确保输出尺寸与输入一致
        if distorted.shape[-2:] != image_tensor.shape[-2:]:
            distorted = F.interpolate(
                distorted,
                size=image_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # 保存攻击后的图片
        to_pil = torchvision.transforms.ToPILImage()
        output_path = f"attack_results/cat_{name}.png"
        to_pil(distorted.squeeze(0).cpu()).save(output_path)
        print(f"Saved: {output_path}")

        # 提取水印
        bwm1 = WaterMark(password_img=1, password_wm=1)
        wm_extract = bwm1.extract(output_path, mode='bit', wm_shape=6)
        print(f"Extracted watermark: {wm_extract}")

        # 显示图片
        plt.figure()
        plt.imshow(to_pil(distorted.squeeze(0).cpu()))
        plt.title(f"Attack: {name}\nWatermark: {wm_extract}")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Error processing {name}: {str(e)}")