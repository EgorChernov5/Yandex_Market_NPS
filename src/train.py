from torch import nn
# from torch.optim import Adam


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.linear_block = nn.Sequential(
            nn.Linear(16*300*300, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 7),
        )

    def forward(self, input):
        x = self.conv_block(input)
        output = self.linear_block(x)
        return output


def main():
    # Get the data
    image = "some image"
    # Init the model
    model = BaselineModel()
    # Define the loss function and an optimizer
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    print('The model:')
    print(model)

    print('\n\nJust one layer:')
    print(model.linear_block)

    print('\n\nModel params:')
    for param in model.parameters():
        print(param)

    print('\n\nLayer params:')
    for param in model.linear_block.parameters():
        print(param)

    print(model(image))


if __name__ == "__main__":
    main()
