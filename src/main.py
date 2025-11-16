from alc import cargarDataset, algoritmo1, algoritmo2, algoritmo3


def main():
    Xt, Yt, Xv, Yv = cargarDataset("./cats_and_dogs")

    print("DIM:")
    print("Xt: ", Xt.shape)
    print("Yt: ", Yt.shape)
    print("Xv: ", Xv.shape)
    print("Yv: ", Yv.shape)

    # W = algoritmo1(Xt, Yt)
    W = algoritmo1(Xt[:, 0:100], Yt[:, 0:100])
    # W = algoritmo2(Xt[: ,0:5], Yt[:, 0:5])
    # W = algoritmo3(Xt[0:5, 0:10], Yt[0:5, 0:10])
    # W = algoritmo3(Xt[0:5, 0:10], Yt[:, 0:10], metodo="GS")

if __name__ == "__main__":
    main()
