import numpy as np
import matplotlib.pyplot as plt


def show_manifold(manifold, imgs, vects, net, n=200):
    list_imgs = list(imgs.items())
    perm = np.arange(len(list_imgs))
    np.random.shuffle(perm)
    points, colors = [], []
    for i in perm[:n]:
        if np.random.randint(0, 2) == 0:
            colors.append(1)
            points.append(list_imgs[i][1][1])
        else:
            colors.append(0)
            id = list_imgs[i][0].lstrip('0')
            vec = net.predict_fun_det([vects[id][1]])[0]
            points.append(vec)

    c_vec = manifold.fit_transform(points)
    plt.scatter(*c_vec.T, c=colors, alpha=0.5, s=50, linewidths=0)
    plt.show()
