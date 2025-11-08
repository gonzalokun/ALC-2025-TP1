from alc import cargarDataset


def main():
    Xt, Yt, Xv, Yv = cargarDataset("cats_and_dogs")

    print("DIM:")
    print("Xt: ", Xt.shape)
    print("Yt: ", Yt.shape)
    print("Xv: ", Xv.shape)
    print("Yv: ", Yv.shape)

if __name__ == "__main__":
    main()
