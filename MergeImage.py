import numpy as np
from PIL import Image


# 创建beta和gamma的值
betas = np.arange(0.05, 1, 0.05)
gammas = np.arange(0.05, 1, 0.1)

# 假定读取第一个图片来确定尺寸
sample_beta = betas[0]
sample_gamma = gammas[0]
sample_img_path = './Images/SIR/simulation_beta{:.2f}gamma_{:.2f}.png'.format(sample_beta, sample_gamma)
sample_img = Image.open(sample_img_path)

# 使用样本图片的宽度和高度
img_width, img_height = sample_img.size

# 计算合并后图片的总宽度和高度
total_width = img_width * len(gammas)
total_height = img_height * len(betas)

# 创建一个新的图片，用于合并所有图片
merged_img = Image.new('RGB', (total_width, total_height))

# 加载并合并图片
for i, beta in enumerate(betas):
    for j, gamma in enumerate(gammas):
        # 构建图片路径
        img_path = './Images/SIR/simulation_beta{:.2f}gamma_{:.2f}.png'.format(beta, gamma)
        try:
            # 尝试加载图片
            img = Image.open(img_path)
            # 计算当前图片的粘贴位置
            x_offset = j * img_width
            y_offset = i * img_height
            # 粘贴图片
            merged_img.paste(img, (x_offset, y_offset))
        except IOError:
            print(f"图片 {img_path} 无法加载")

# 如果需要，也可以保存到文件
merged_img.save('merged_image_SIR.png')
