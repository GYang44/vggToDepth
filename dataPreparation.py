import utils

if __name__ == "__main__":
    color, depth = utils.indexTraininData('D:\Archive\DepthTraining\depthPhoto')
    #color, depth = utils.indexTraininData('/media/gyang/INO1/Archive/DepthTraining/depthPhoto')

    for d, c in zip(depth, color):
        #utils.show_image(utils.load_image(d))
        #utils.show_image(utils.load_image(c))
        img = utils.load_image(d, isGray=True)
        pass
