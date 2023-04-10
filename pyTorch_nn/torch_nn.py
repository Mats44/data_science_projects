# Import dependencies
import torch
from PIL import Image
from torch import nn, save, load # neural net
from torch.optim import Adam # optimizer
from torch.utils.data import DataLoader # loading a pyTorch dataset
from torchvision import datasets
from torchvision.transforms import ToTensor # convert images to tensors

# Download MNIST dataset from torch. Include training partition. Transform  images to tensor.
# Images from MNIST have the shape: 1,28,28 and classes 0-9
train = datasets.MNIST(root="data", download=True, train=True, transform =ToTensor() )
dataset = DataLoader(train, 32) # train 32 batches

# Neural Network - Image Classifier.
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10) # 28x28 original image minus 6x6 lost pixels in nn layers.
        )

    def forward(self, x):
        return self.model(x)

# Neural network instance, loss instance, optimizer instance.
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training pipeline
if __name__ == "__main__":
    for epoch in range(10): # train for 10 epochs
        for batch in dataset:
            X,y = batch
            X,y = X.to("cuda"), y.to("cuda")
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Back propagation
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch} loss is {loss.item()}")

    # Save and then load the model weights and biases.
    with open("model_state.pt", "wb") as f: 
        save(clf.state_dict(), f)

    with open("model_state.pt", "rb") as f:
        clf.load_state_dict(load(f))

    # Transform one of the images to a tensor.
    img = Image.open("img_3.jpg")
    img_tensor = ToTensor()(img).unsqueeze(0).to("cuda")

    # Print out the class with the highest calculated probability.
    print( f"The digit in img_3 is: {torch.argmax(clf(img_tensor))}" )
