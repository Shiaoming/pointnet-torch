import numpy as np

vis_ok = False
try:
    import mayavi.mlab

    vis_ok = True
except:
    print("mayavi.mlab not available")


# def pc_viewer(points, point_size=0.02, seg=None, color_map=None):
def pc_viewer(points, seg=None, color_map=None, figure=None, show=True):
    if vis_ok:
        x = points[:, 0]  # x position of point
        y = points[:, 1]  # y position of point
        z = points[:, 2]  # z position of point

        N = x.shape[0]
        scalars = np.arange(N)

        if seg is None:
            if figure != None:
                mayavi.mlab.points3d(x, y, z, scalars, mode="sphere", figure=figure)
            else:
                mayavi.mlab.points3d(x, y, z, scalars, mode="sphere")
        else:
            # construct color of each point
            color = np.random.random((N, 4)).astype(np.uint8)
            color[:, -1] = 255  # No transparency
            for i, color_idx in enumerate(seg):
                color[i, 0:3] = 255 * np.array(color_map[color_idx])  # assign color

            if figure != None:
                nodes = mayavi.mlab.points3d(x, y, z, scalars, mode="sphere", figure=figure)
            else:
                nodes = mayavi.mlab.points3d(x, y, z, scalars, mode="sphere")

            nodes.glyph.color_mode = 'color_by_scalar'
            # Set look-up table and redraw
            nodes.module_manager.scalar_lut_manager.lut.table = color

            # nodes.glyph.scale_mode = 'scale_by_vector'
            # nodes.mlab_source.dataset.point_data.vectors = scalars

        if show:
            mayavi.mlab.show()
    else:
        print("mayavi.mlab not available")
