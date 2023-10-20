import torchvision.transforms as T

preprocessing_func = T.Compose(
    [
        T.RandomRotation(45),
        T.ColorJitter(),
        T.RandomHorizontalFlip(),
        T.Resize((224, 224)),
        T.ToTensor()
    ]
)

def preprocess(img):
    return preprocessing_func(img)