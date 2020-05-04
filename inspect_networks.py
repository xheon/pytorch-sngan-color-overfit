from generator import Generator
from torchsummary import summary

def main():
    network = Generator(1)

    summary(network, input_data=(1, 1, 1))

if __name__ == '__main__':
    main()