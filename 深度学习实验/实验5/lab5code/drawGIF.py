import imageio

image_list = ['./piclogs/Generator/wgan-gp_rmsprop/epoch' + str(x) + ".png" for x in range(5, 201, 5)]
gif_name = './piclogs/gifs/wgan-gp_rmsprop.gif'

frames = []
for image_name in image_list:
    frames.append(imageio.imread(image_name))

imageio.mimsave(gif_name, frames, 'GIF', duration=0.3)
