import click
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt



def main(v, f, ol=None):
    """ Plots a 3D triangle surface with overlay.
    
    Parameters
    ----------
    v : np.array
        3D (X, Y, Z) point cloud / vertex coordinate array
    f : np.array
        3D (v1, v2, v3) array with face indices
    ol : np.array
        1D OverLay array with size equal to v.shape[0] or f.shape[0]
    """
    
    fig = plt.figure(figsize=(10, 15))
    ax = fig.gca(projection='3d')

    X, Y, Z = v[:, 0], v[:, 1], v[:, 2]
    coll = ax.plot_trisurf(Z, X, Y, triangles=f, antialiased=False, vmin=0, edgecolor='none', linewidth=0)
    ax.view_init(0, 0)
    ax.axis('off')
    if ol is not None:
        if ol.shape[0] == v.shape[0]:
            # Face value = average of vertex values
            ol = ol[f].mean(axis=1)
        
        if ol.shape[0] != f.shape[0]:
            raise ValueError("Overlay has the wrong shape!")

        coll.set_array(ol)


if __name__ == '__main__':
    main()