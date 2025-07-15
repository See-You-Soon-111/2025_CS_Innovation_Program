# <center>Project02——图片(盲)水印的嵌入与提取以及鲁棒性测试<center>

​		数字水印技术(Digital Watermarking)是通过一定的算法将一些标志性信息直接嵌入到多媒体内容当中，但不影响原来内容的价值和使用，并且不能被人的感知系统察觉或者注意到，只有通过专门的检测器或者阅读器才能提取。

​		本项目基于开源库[blind_watermark](https://github.com/guofei9987/blind_watermark)，实现图片的盲水印嵌入与提取，并对该图片水印的鲁棒性进行测试。

## 1 嵌入水印

​		基于上述库实现图片水印的嵌入如下：

```python
bwm1 = WaterMark(password_img=1, password_wm=1)
bwm1.read_img('cat.png')
bwm1.read_wm([True, False, True, False, True, False], mode='bit')
bwm1.embed('cat-w.png')
len_wm = len(bwm1.wm_bit)
print(f'Watermark length: {len_wm}')
```

## 2 提取水印与鲁棒性测试

​		我们将图片嵌入水印，然后对图片进行一系列操作，再对操作后的图片进行提取水印，若提取水印与原水印一致，则认为对该操作有好的鲁棒性。定义攻击方式如下：

```python
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
```



​		测试结果如下

|    图片操作     | 结果 |
| :-------------: | :--: |
|    原始图片     | 正确 |
|    水平翻转     | 错误 |
|   右下移动20%   | 正确 |
|   右下移动40%   | 正确 |
| 保留50%中心区域 | 正确 |
|     缩小30%     | 正确 |
|     缩小50%     | 正确 |
|    旋转45度     | 错误 |
|    旋转90度     | 错误 |
|    高斯模糊     | 正确 |
|   降低对比度    | 正确 |
|   提高对比度    | 正确 |

​		可以看出，该开源库实现的图片盲水印技术，对于旋转、反转的鲁棒性不佳；对于平移、缩放、模糊、对比度、裁剪的鲁棒性比较好。
