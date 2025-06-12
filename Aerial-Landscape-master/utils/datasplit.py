import os, shutil
from sklearn.model_selection import train_test_split

source_dir = '../Aerial_Landscapes'
train_dir = './Aerial_Landscapes_split/train'
test_dir = './Aerial_Landscapes_split/test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

classes = os.listdir(source_dir)

for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    if not os.path.isdir(cls_path): continue

    images = os.listdir(cls_path)
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))
