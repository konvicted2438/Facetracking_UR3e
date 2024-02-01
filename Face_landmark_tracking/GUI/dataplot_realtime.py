import numpy as np
import time
import matplotlib
matplotlib.use('GTKAgg')
from matplotlib import pyplot as plt


class dataplot_realtime:
    def __init__(self, process_variance, measurement_variance):
        self.plot, self.ax = plt.subplot(1,1)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0,360)
        

    def run(niter=1000, doblit=True):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.hold(True)
        rw = randomwalk()
        x, y = rw.next()

        plt.show(False)
        plt.draw()

        if doblit:
        # cache the background
            background = fig.canvas.copy_from_bbox(ax.bbox)

        points = ax.plot(x, y, 'o')[0]
        tic = time.time()

        for ii in xrange(niter):

        # update the xy data
            x, y = rw.next()
            points.set_data(x, y)

            if doblit:
            # restore background
                fig.canvas.restore_region(background)

            # redraw just the points
                ax.draw_artist(points)

            # fill in the axes rectangle
                fig.canvas.blit(ax.bbox)

            else:
            # redraw everything
                fig.canvas.draw()

        plt.close(fig)
        print(str(doblit), niter / (time.time() - tic))