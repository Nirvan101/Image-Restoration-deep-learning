
# Image restoration by in-painting

This is my image-restoration project. The aim is to restore images of old and damaged paintings. This project takes an image in-painting approach to this problem. 

This repository contains multiple models that I constructed to solve this task. The models include context-encoders, GANS, conditional GANS and pixel diffusion.



## Preprocessing:

The dataset is very small. The training dataset contains only 20 images. So, I have used data augmentation using the ImageDataGenerator. The images are all resized to (256,384,3) as this is the average image size in the dataset. They are then divided by 255 to normalize them. 




## Models:

### Context Encoder

I constructed my model based on the paper 'Context Encoders: Feature Learning by Inpainting'  found here https://arxiv.org/pdf/1604.07379.pdf

This model contains an encoder and a decoder. The image is fed into the encoder, it is downsampled into an encoding using Conv layers. The encoding is upsampled using Conv and Upsampling layers. The output is of the same size as the input i.e (256,384,3)
    
I use a mean-square loss function and a sigmoid activation in the output layer. 

The output is then multiplied by 255 to get the final reconstructed output image.

----------------------------------------------------------------------------------------------------
 
### Context Encoder 2

This was my second design for the context encoder. Here I have used a Dense layer to generate the encoding from the encoder. The intuition is that the Dense layer will connect features from different regions of the image together and this will improve the inpainting performance.

-----------------------------------------------------------------------------------------------------

### Conditional GAN

I designed my conditional GAN based on the paper 'SEMI-SUPERVISED LEARNING WITH CONTEXT-CONDITIONAL GENERATIVE ADVERSARIAL NETWORK' found here https://arxiv.org/pdf/1611.06430v1.pdf

The generator receives an image which has it's central region masked out. The generator produces an image of the same size resembling the input image. This output image is passed to a custom keras layer which also receives the input image from the input layer. This custom layer replaces the central part of the original image with the central part of the generated image and this new image is the final output of the generator. Then this is passed to the discriminator. In this way, the model is trained to produce only the central masked region of the image when the remaining region is provided.

-----------------------------------------------------------------------------------------------------

### GAN

This is a regular GAN where the generator has an mse loss function and the discriminator has a binary_cross_entropy loss function. The generator is designed to generate the entire input image back from the encoding. 

-----------------------------------------------------------------------------------------------------


### Pixel diff

This was a script I wrote using the opencv inpainting function. The user has to manually select a rectangular portion of the image and the script will automatically perform inpainting in that region.

