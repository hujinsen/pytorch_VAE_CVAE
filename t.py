import imageio
#/Users/JSen/Documents/pytorch_VAE_CVAE/results/sample_1.png
#/Users/JSen/Documents/pytorch_VAE_CVAE/results/reconstruction_1.png
images = []
for i in range(1, 101):
    one_image = f'./results/reconstruction_{i}.png'
    images.append(imageio.imread(one_image))
imageio.mimsave(f'./results/reconstruction.gif', images, fps=5)