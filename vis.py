import visdom
import time
import numpy as np


class Visualizer(object):
    def __init__(self, config):
        super(Visualizer, self).__init__()

        visdom_port = config.visdom_port
        env = "RCNN_Refine_"+config.date

        self.visdom_port = visdom_port

        vis = visdom.Visdom(port=visdom_port, env=env)
        self.vis = vis

    def show_pointcloud(self, points, title=None, Y=None):
        """
        :param points: pytorch tensor pointcloud
        :param title:
        :param Y:
        :return:
        """
        points = points.squeeze()
        if points.size(-1) == 3:
            points = points.contiguous().data.cpu()
        else:
            points = points.transpose(0, 1).contiguous().data.cpu()

        opts = dict(
            title=title,
            markersize=2,
            xtickmin=-0.7,
            xtickmax=0.7,
            xtickstep=0.3,
            ytickmin=-0.7,
            ytickmax=0.7,
            ytickstep=0.3,
            ztickmin=-0.7,
            ztickmax=0.7,
            ztickstep=0.3)

        if Y is None:
            self.vis.scatter(X=points, win=title, opts=opts)
        else:
            if Y.min() < 1:
                Y = Y - Y.min() + 1
            self.vis.scatter(
                X=points, Y=Y, win=title, opts=opts
            )

    def show_pointclouds(self, points, title=None, Y=None):
        points = points.squeeze()
        assert points.dim() == 2
        for i in range(points.size(0)):
            self.show_pointcloud(points[i], title=title)


    def show_image(self, img, title=None):
        img = img.squeeze()
        self.vis.image(img, win=title, opts=dict(title=title))


    def plot(self, name, y, x, **kwargs):
        '''
        self.plot('loss',1.00)
        '''
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )

    def scatter(self, name, points):
        self.vis.scatter(points,
                         win=name)
        
    def image(self, name, image):
        self.vis.image(image,
                       win=name)
        
    def images(self, name, images):
        self.vis.images(images, win=name)