import os
import tempfile
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from PIL import Image, ImageFile
from torchvision import datasets
from concurrent.futures import ThreadPoolExecutor

# --- Списки классов (метки словами) ---
MAIN_CLASSES = [
    'antelope', 'badger', 'bat', 'bear', 'bird', 'bison', 'boar', 'cat',
    'cheetah', 'cow', 'coyote', 'crab', 'deer', 'dog', 'dolphin', 'donkey',
    'elephant', 'fish', 'fox', 'frog', 'goat', 'hamster', 'hare', 'hedgehog',
    'hippopotamus', 'horse', 'hyena', 'insect', 'jelly fish', 'kangaroo',
    'koala', 'leopard', 'lion', 'lizard', 'monkey', 'mouse', 'octopus', 'okapi',
    'otter', 'ox', 'panda', 'pig', 'porcupine', 'possum', 'puma', 'raccoon',
    'rat', 'reindeer', 'rhinoceros', 'seahorse', 'seal', 'shark', 'sheep',
    'shrimp', 'snail', 'snake', 'squirrel', 'starfish', 'tiger', 'turtle',
    'whale', 'wolf', 'wombat', 'zebra'
]

CAT_CLASSES = [
    'abyssinian', 'bengal', 'birman', 'bombay', 'british shorthair',
    'egyptian mau', 'himalayan', 'maine coon', 'norwegian forest',
    'oriental shorthair', 'persian', 'ragdoll', 'russian blue',
    'scottish fold', 'siamese', 'sphynx', 'turkish angora'
]

DOG_CLASSES = [
    'american bulldog', 'basset hound', 'beagle', 'border collie', 'boxer',
    'bull terrier', 'chihuahua', 'corgi', 'dachshund', 'doberman',
    'english cocker spaniel', 'german shorthaired', 'great pyrenees',
    'havanese', 'husky', 'japanese chin', 'keeshond', 'leonberger',
    'miniature pinscher', 'newfoundland', 'pomeranian', 'poodle', 'pug',
    'saint bernard', 'samoyed', 'scottish terrier', 'shiba inu',
    'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier'
]

MONKEY_CLASSES = [
    'baboon', 'bald uakari', 'cebus capucinus', 'chimpanzee',
    'colobus monkey', 'common squirrel monkey', 'gibbon',
    'golden snub-nosed monkey', 'gorilla', 'japanese macaque', 'lemur',
    'mantled howler', 'marmoset', 'nilgiri langur', 'orangutan',
    'patas monkey', 'proboscis monkey', 'red-shanked douc', 'spider monkey',
    'tamarin', 'tarsiers', 'vervet monkey'
]

BIRD_CLASSES = [
    'bullfinch bird', 'crow', 'duck', 'eagle', 'flamingo', 'goose',
    'hummingbird', 'owl', 'parrot', 'peacock', 'pelican', 'penguin',
    'pigeon', 'sandpiper', 'sparrow', 'swan', 'titmouse', 'turkey',
    'turkey bird', 'woodpecker'
]

# --- Настройки ---
TOKEN = os.getenv("TELEGRAM_TOKEN")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# --- Трансформации ---
def sharpen_image(image: Image.Image) -> Image.Image:
    return ImageEnhance.Sharpness(image).enhance(2.5)

def apply_clahe(image: Image.Image, clip_limit: float = 1.5) -> Image.Image:
    arr = np.array(image)
    if arr.ndim == 3:
        yuv = cv2.cvtColor(arr, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8)).apply(yuv[:, :, 0])
        arr = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    else:
        arr = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8)).apply(arr)
    return Image.fromarray(arr)

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.Lambda(sharpen_image),
    transforms.Lambda(lambda x: apply_clahe(x, 1.5)),
    transforms.ToTensor(),
])

# --- Общая модель ---
class AnimalNet(nn.Module):
    def __init__(self, in_features, num_classes):
        super(AnimalNet, self).__init__()
        self.reduce_conv = nn.Conv2d(in_features, 128, kernel_size=1)  
        self.bn = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.reduce_conv(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, num_classes, fine_tune=False):
        super(HybridModel, self).__init__()
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        self.features = self.backbone.features

        if not fine_tune:
            for param in self.features.parameters():
                param.requires_grad = False
              
        self.animal_head = AnimalNet(in_features=1280, num_classes=num_classes)

    def forward(self, x):
      x = self.features(x) 
      x = self.animal_head(x) 
      return x

# --- Cat Model ---
class CatNet(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)

class HybridCatModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base_model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base_model.features.children()))
        self.classifier = CatNet(in_features=1280, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# --- Dog Model ---
class DogNet(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)


class HybridDogModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base_model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base_model.features.children()))
        self.classifier = DogNet(in_features=1280, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# --- Monkey Model ---
class MonkeyNet(nn.Module):
    def __init__(self, in_features, num_classes):
        super(MonkeyNet, self).__init__()

        self.reduce_conv = nn.Conv2d(in_features, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.reduce_conv(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class HybridMonkeyModel(nn.Module):
    def __init__(self, num_classes, fine_tune=False):
        super(HybridMonkeyModel, self).__init__()
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        self.features = self.backbone.features

        if not fine_tune:
            for param in self.features.parameters():
                param.requires_grad = False

        self.monkey_head = MonkeyNet(in_features=1280, num_classes=num_classes)

    def forward(self, x):
      x = self.features(x) 
      x = self.monkey_head(x)
      return x

# --- Bird Model ---
class BirdNet(nn.Module):
    def __init__(self, in_features, num_classes):
        super(BirdNet, self).__init__()

        self.reduce_conv = nn.Conv2d(in_features, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.reduce_conv(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class HybridBirdModel(nn.Module):
    def __init__(self, num_classes, fine_tune=False):
        super(HybridBirdModel, self).__init__()
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        self.features = self.backbone.features

        if not fine_tune:
            for param in self.features.parameters():
                param.requires_grad = False

        self.bird_head = BirdNet(in_features=1280, num_classes=num_classes)

    def forward(self, x):
      x = self.features(x)        # признаки из EfficientNet
      x = self.bird_head(x)     # классификация в BirdNet
      return x

# --- Загрузка ---
def load_model(path: str, model_cls: nn.Module, cls_list: list) -> nn.Module:
    num_classes = len(cls_list)
    model = model_cls(num_classes)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

# --- Предсказание ---
def predict(model: nn.Module, img_path: str, cls_list: list) -> dict:
    img = Image.open(img_path).convert('RGB')
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
        probs = F.softmax(out, 1)[0] * 100
        idx = int(probs.argmax())
        top = torch.topk(probs, 3)
    return {
        'pred': cls_list[idx],
        'conf': float(probs[idx]),
        'top3': [(cls_list[i], float(p)) for p, i in zip(top.values, top.indices)]
    }

# --- Путь и модели ---
MODEL_DIR = os.getenv("MODEL_DIR", "models")

main_model   = load_model(
    os.path.join(MODEL_DIR, 'best_model_cpu.pth'),
    HybridModel,
    MAIN_CLASSES
)

cat_model    = load_model(
    os.path.join(MODEL_DIR, 'best_cat_model_cpu.pth'),
    HybridCatModel,
    CAT_CLASSES
)

dog_model    = load_model(
    os.path.join(MODEL_DIR, 'best_dog_model_cpu.pth'),
    HybridDogModel,
    DOG_CLASSES
)

monkey_model = load_model(
    os.path.join(MODEL_DIR, 'best_monkey_model.pth'),
    HybridMonkeyModel,
    MONKEY_CLASSES
)

bird_model   = load_model(
    os.path.join(MODEL_DIR, 'best_bird_model_cpu.pth'),
    HybridBirdModel,
    BIRD_CLASSES
)

# --- Обработчик фото ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text('Думаю...')
    photo = update.message.photo[-1]
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tf:
        await photo.get_file().download_to_drive(tf.name)
        res = predict(main_model, tf.name, MAIN_CLASSES)
        text = f"Вероятность {res['conf']:.1f}% что это {res['pred']}"
        if res['pred'] in ('cat','dog','monkey','bird'):
            smodel, sclist = {
                'cat':    (cat_model,    CAT_CLASSES),
                'dog':    (dog_model,    DOG_CLASSES),
                'monkey': (monkey_model, MONKEY_CLASSES),
                'bird':   (bird_model,   BIRD_CLASSES)
            }[res['pred']]
            sub = predict(smodel, tf.name, sclist)
            text += f"\nПорода: {sub['pred']} ({sub['conf']:.1f}% уверенности)"
        await msg.edit_text(text)

# --- Запуск бота ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', lambda u,c: u.message.reply_text('Привет! Я бот 🐾')))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

