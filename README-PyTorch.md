# PyTorch Specific Guidelines
The preprocessing pipeline, if any used, which was used during training time has to be baked in the model's forward method. The testing code simply loads the model for inference and feeds the raw cropped RGB screenshots to the model.

Example ([credit](https://discuss.pytorch.org/t/how-to-add-preprocess-into-the-model/66154/4)):

```python
class MyModel(nn.Module):
    def __init__(self, transform):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 1, 3, 1, 1)
        self.transform = transform
        
    def forward(self, x):
        xs = []
        for x_ in x:
            x_ = self.transform(x_)
            xs.append(x_)
        xs = torch.stack(xs)
        x = self.conv(xs)
        return x

transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
model = MyModel(transform)
x = torch.randn(1, 3, 24, 24)
output = model(x)
```

### Image Classifier
The model consists of only 1 class - person. The model needs to indicate the probability a person is present in the input. Training the model on specific in-game characters from the mentioned contest video game rather than a generic 'person datasets' helps the performance of the model tremendously. If there exists a person in the input, the model outputs values close to 1., else 0. will be the output. The model takes in an `uint8` image of size `(?, 3, H, W)` as input and returns a single `floating-point` number of size `(?, 1)`. The leading `?` in the dimensions stand for batch size. However, while testing it is guaranteed to be 1 as the input video feed is looped frame-by-frame and fed to the model individually. The participants need to make sure the input is converted to `float` inside the model's forward method, else it might throw error because of `uint8` datatype.
